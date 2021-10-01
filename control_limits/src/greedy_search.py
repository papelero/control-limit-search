from .statistical_distance import StatisticalDistance
from .decision_boundary import DecisionBoundary
from .utils import *


class GreedySearch(DecisionBoundary, StatisticalDistance):
    """Greedy search to construct control limits

    :param data: input data
    :param labels: input labels
    :param precision_limits: desired precision of the control limits
    :param length_limits: desired length of the control limits
    :param shape_limits: desired shape of the control limits
    """

    def __init__(self, data, labels, precision_limits, length_limits, shape_limits):
        super(GreedySearch, self).__init__(data, labels, precision_limits)

        self.data = data
        self.labels = labels
        self.precision_limits = precision_limits
        self.length_limits = length_limits
        self.shape_limits = shape_limits

        self.label_nok = self._label_nok()
        self.data_ok = self._data_ok()
        self.min_length = self._min_length()
        self.time_step = self._time_step()
        self.time_steps_limits = self._time_steps_limits()
        self.split_points_limits = self._split_points_limits()
        self.control_limits = self._control_limits()

    @staticmethod
    def define_decision_boundary(all_split_points):
        """Return decision boundary provided the split points

        :param all_split_points: split points
        :return: decision boundary
        """

        decision_boundary = {di: [] for di in range(len(all_split_points[0]))}
        for split_points in all_split_points:
            for di, value in enumerate(split_points):
                decision_boundary[di] += [value]
        return decision_boundary

    def _label_nok(self):
        """Return the label of the faulty distribution

        :return: label of the faulty distribution
        """

        return np.unique(self.labels)[-1]

    def _data_ok(self):
        """Return the true distribution

        :return: data of the true distribution
        """

        return self.data[self.labels == self.label_ok, :]

    def _min_length(self):
        """Return the minimum length of the control limits

        :return: minimum length of the control limits
        """

        return self.length_limits // 2

    def _time_step(self):
        """Return the starting time step

        :return: starting time step
        """

        _, time_step = self.estimate_stat_distance()
        return time_step

    def _time_steps_limits(self):
        """Return the time steps within the minimum length

        :return: time steps within the minimum length
        """

        # Time steps if starting time steps and desired minimum length exceed 0
        if (self.time_step - self.min_length) < 0:
            interval = range(0, (2 * self.min_length))

        # Time steps if starting time steps and desired minimum length exceed length of data
        elif (self.time_step + self.min_length) > self.data.shape[-1]:
            interval = range(self.data.shape[-1] - (2 * self.min_length), self.data.shape[-1])
        else:
            interval = range(self.time_step - self.min_length, self.time_step + self.min_length)
        return [di for di in interval]

    def _split_points_limits(self):
        """Return the split points within the minimum length

        :return: split points with the minimum length
        """

        return [self(t) for t in self.time_steps_limits]

    def _control_limits(self):
        """Return the control limits

        :return: control limits
        """

        if self.shape_limits == 0:
            return self.get_control_limits(self.time_steps_limits, self.split_points_limits)
        else:
            return self.get_control_limits_parallel(self.time_steps_limits, self.split_points_limits)

    def get_control_limits(self, time_steps_limits, split_points_limits):
        """Return the control limits

        :param time_steps_limits: time steps of control limits
        :param split_points_limits: split points of control limits
        :return: control limits
        """

        decision_boundary = self.define_decision_boundary(split_points_limits)
        control_limits = [linear_regression(time_steps_limits, value) for _, value in decision_boundary.items()]
        return dict(zip(range(0, len(control_limits)), control_limits))

    def get_control_limits_parallel(self, time_steps_limits, split_points_limits):
        """Return the control limits parallel

        :param time_steps_limits: time steps of control limits
        :param split_points_limits: split points of control limits
        :return: control limits
        """

        decision_boundary = self.define_decision_boundary(split_points_limits)
        control_limits = [linear_regression(time_steps_limits, value) for _, value in decision_boundary.items()]

        # Calculate the shift for the other line which is the maximum distance between the two lines of the decision
        # boundary
        shift = abs(np.max(np.subtract(np.asarray(decision_boundary[0]), np.asarray(decision_boundary[1]))))

        # Determine which line of the decision boundary is closer to the normal distribution
        dist_to_centroid = []
        for _, value in decision_boundary.items():
            dist_to_centroid += [distance_euclidean(np.median(self.data_ok[:, time_steps_limits], axis=0), value)]
        indices = np.argmin(np.asarray(dist_to_centroid))

        # The other line of the decision boundary is the original line shifted according to the shift
        if indices == 0:
            control_limits[1] = control_limits[indices] + shift
        else:
            control_limits[0] = control_limits[indices] - shift
        return dict(zip(range(0, len(control_limits)), control_limits))

    def labels_and_accuracy(self, time_steps, control_limits):
        """Fit the control limits to the data

        :param time_steps: time steps of the control limits
        :param control_limits: control limits
        :return: labels control limits and precision and recall
        """

        data_limits = self.data[:, time_steps]

        # Set the predicted labels to 1 for the data that is entirely included within the control limits, otherwise set
        # the predicted labels to 2
        labels_limits = np.ones(shape=(self.labels.size,), dtype=np.int8)
        for di in range(len(time_steps)):
            indices_nok = np.where(np.logical_or(data_limits[:, di] < control_limits[0][di],
                                                 data_limits[:, di] > control_limits[1][di]))
            labels_limits[indices_nok] = self.label_nok

        # Calculate the precision and recall as a result of the predicted labels
        precision, recall = precision_and_recall(self.labels, labels_limits)
        return labels_limits, (precision, recall)

    def control_limits_and_score(self, time_steps, split_points):
        """Determine the control limits and the accuracy

        :param time_steps: time steps of the control limits
        :param split_points: split points of the control limits
        :return: control limits and f-beta score
        """

        if self.shape_limits == 0:
            control_limits = self.get_control_limits(time_steps, split_points)
            _, (precision, recall) = self.labels_and_accuracy(time_steps, control_limits)
        else:
            control_limits = self.get_control_limits_parallel(time_steps, split_points)
            _, (precision, recall) = self.labels_and_accuracy(time_steps, control_limits)
        score = self.f_beta_score(precision, recall)
        return control_limits, score

    def search_next(self, search_path, next_time_step):
        """Determine performance at the next time step

        :param search_path: which path to take for the definition of the control limits
        :param next_time_step: next time step
        :return next time steps and next control limits
        """

        # If the search path is 0, then integrate the next time step and split points on the left
        if search_path == 0:
            next_time_steps = [next_time_step, *self.time_steps_limits]
            next_split_points = [self(next_time_step), *self.split_points_limits]

        # If the search path is 1, then integrate the next time step and split points on the right
        else:
            next_time_steps = [*self.time_steps_limits, next_time_step]
            next_split_points = [*self.split_points_limits, self(next_time_step)]

        # Calculate the next f-beta score as a result of integrating the next time steps and split points
        _, next_score = self.control_limits_and_score(next_time_steps, next_split_points)
        return next_time_steps, next_split_points, next_score

    def f_beta_score(self, precision, recall):
        """Estimate the f-beta score

        :param precision: precision control limits
        :param recall: recall control limits
        :return: f-beta score
        """

        beta = get_beta(self.precision_limits)
        return (1 + (beta ** 2)) * (precision * recall / ((beta ** 2) * precision + recall))

    def solve(self):
        """Solve the greedy search

        :return: time steps, control limits and predicted labels
        """

        # Calculate the f-beta score for the control with the minimum length
        _, (precision, recall) = self.labels_and_accuracy(self.time_steps_limits, self.control_limits)
        score = self.f_beta_score(precision, recall)

        # Start the greedy search
        while True:
            next_time_steps = dict(zip(range(0, len(SearchPath)), len(SearchPath) * [None]))
            next_split_points = dict(zip(range(0, len(SearchPath)), len(SearchPath) * [None]))
            next_scores = dict(zip(range(0, len(SearchPath)), len(SearchPath) * [None]))

            # For each search path (i.e., one time step left or one time step right)
            for i in range(len(SearchPath)):

                # Calculate the next time step
                next_time_step = self.time_steps_limits[0] - 1 if i == 0 else self.time_steps_limits[-1] + 1

                # If the next step does not exceed the 0 or the over all length of the data estimate the f-beta score
                if (next_time_step >= 0) and (next_time_step <= (self.data.shape[-1] - 1)):
                    next_time_steps[i], next_split_points[i], next_scores[i] = self.search_next(i, next_time_step)

            # Determine which of the two search paths has a score that is better or equal the original score
            which_path = [i for i, item in next_scores.items() if item is not None and score <= item]

            # If a search path is returned the original time steps and split points are integrated with the next ones
            # and the the original score is updated. Else the time steps, control limits and predicted labels are
            # returned. Note that both search paths are returned, then the left one is considered.
            if which_path:
                self.time_steps_limits = next_time_steps[which_path[0]]
                self.split_points_limits = next_split_points[which_path[0]]
                score = next_scores[which_path[0]]
            else:
                control_limits, _ = self.control_limits_and_score(self.time_steps_limits, self.split_points_limits)
                pred_labels, _ = self.labels_and_accuracy(self.time_steps_limits, control_limits)
                return self.time_steps_limits, control_limits, pred_labels
