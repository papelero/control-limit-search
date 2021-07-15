import numpy as np
from .max_class_separability import StatisticalDistance
from .stat_analysis_limits import StatisticalAnalysis
from .decision_boundary import SplitPointSearch
from ..utils import euc_distance, linear_regression, GreedySearch


class GreedySearchLimits(StatisticalDistance, StatisticalAnalysis):
    """Construct the control limits

    :param array: input data
    :type array: numpy array
    :param labels: input labels
    :type labels: numpy array
    :param precision_limits: user-input precision
    :type precision_limits: float
    :param length_limits: user-input length
    :type length_limits: int
    :param shape_limits: user-input shape
    :type shape_limits: int
    """

    def __init__(self, array, labels, precision_limits, length_limits, shape_limits):
        StatisticalDistance.__init__(self, array, labels)
        StatisticalAnalysis.__init__(self, precision_limits, precision=float(1e-6), recall=float(1e-6))

        self.precision_limits = precision_limits
        self.length_limits = length_limits
        self.shape_limits = shape_limits
        self.start_time_step = self.__start_time_step__()
        self.label_nok = self.__label_nok__()
        self.label_ok = self.__label_ok__()
        self.data_ok = self.__data_ok__()
        self.threshold = self.__threshold__()
        self.time_steps = self.__time_steps__()
        self.split_points = self.__split_points__()
        self.all_split_points = self.__all_split_points__()

    def __repr__(self):
        return f'Control limits(precision={self.precision}, recall={self.recall}, beta={self.beta})'

    def __start_time_step__(self):
        """Return the time step of maximum class separability

        :return: starting time step
        :rtype: int
        """

        return self.extract_start_time_step()

    def __label_ok__(self):
        """Return the label of the true distribution

        :return: label true distribution
        :rtype: int
        """

        return np.unique(self.labels)[0]

    def __label_nok__(self):
        """Return the label of the faulty distribution

        :return: label faulty distribution
        :rtype: int
        """

        return np.unique(self.labels)[-1]

    def __data_ok__(self):
        """Return true data distribution

        :return: true data distribution
        :rtype: numpy array
        """

        return self.array[self.labels == self.label_ok]

    def __threshold__(self):
        """Return threshold length for one side of control limits

        :return: one-sided threshold of control limits
        :rtype: int
        """

        return self.length_limits // 2

    def __time_steps__(self):
        """Return the time steps within the threshold length

        :return: time steps within threshold length
        :rtype: list
        """

        return [idx for idx in range(self.start_time_step - self.threshold, self.start_time_step + self.threshold)]

    def __split_points__(self):
        """Return threshold length for one side of control limits

        :return: one-sided threshold of control limits
        :rtype: SplitPointSearch
        """

        return SplitPointSearch(self.array, self.labels, self.start_time_step, self.precision_limits)

    def __all_split_points__(self):
        """Return split points within the threshold length

        :return: split points within threshold length
        :rtype: list
        """

        all_split_points = list()
        for time_step in self.time_steps:
            self.split_points.update(time_step)
            all_split_points += [self.split_points.fit()]
        return all_split_points

    @staticmethod
    def __define_boundaries(all_split_points):
        """Return low and high decision boundary provided the split points


        :param all_split_points: split points
        :type all_split_points: list
        :return: low and high decision boundary
        :rtype: tuple
        """

        low_dec_boundary = np.empty(shape=(len(all_split_points),))
        high_dec_boundary = np.empty(shape=(len(all_split_points),))
        for idx, value in enumerate(all_split_points):
            low_dec_boundary[idx], high_dec_boundary[idx] = value[0], value[-1]
        return low_dec_boundary, high_dec_boundary

    def construct_limits(self, time_steps, all_split_points):
        """Return the control limits

        :param time_steps: time steps of control limits
        :type time_steps: list
        :param all_split_points: split points of control limits
        :type all_split_points: list
        :return: control limits
        :rtype: dict
        """

        dec_boundaries = self.__define_boundaries(all_split_points)
        limits = [linear_regression(time_steps, dec_boundary) for dec_boundary in dec_boundaries]
        return dict(zip(range(0, len(limits)), limits))

    def construct_limits_parallel(self, time_steps, all_split_points):
        """Return the parallel control limits

        :param time_steps: time steps of control limits
        :type time_steps: list
        :param all_split_points: split points of control limits
        :type all_split_points: list
        :return: parallel control limits
        :rtype: dict
        """

        dec_boundaries = self.__define_boundaries(all_split_points)

        centroid_data_ok = np.median(self.data_ok[:, time_steps], axis=0)
        dist_to_centroid = [euc_distance(centroid_data_ok, dec_boundary) for dec_boundary in dec_boundaries]
        indices = np.argmin(np.asarray(dist_to_centroid)).item()
        offset = abs(np.max(np.asarray(dec_boundaries[0]) - np.asarray(dec_boundaries[1])))

        limits = [linear_regression(time_steps, dec_boundary) for dec_boundary in dec_boundaries]
        if indices == 0:
            limits[1] = limits[indices] + offset
        else:
            limits[0] = limits[indices] - offset
        return dict(zip(range(0, len(limits)), limits))

    def fit_limits(self, time_steps, limits, predict=False):
        """Deploy current control limits on the data

        :param time_steps: time steps of control limits
        :type time_steps: list
        :param limits: control limits
        :type limits: dict
        :param predict: flag for return the labels
        :type predict: bool
        :return: labels predicted based on the control limits
        :rtype: numpy array
        """

        array_limits = self.array[:, time_steps]
        labels_limits = np.ones(shape=(self.labels.size,), dtype=np.int8)
        for idx in range(len(time_steps)):
            indices_nok = np.where(np.logical_or(array_limits[:, idx] < limits[0][idx],
                                                 array_limits[:, idx] > limits[1][idx]))
            labels_limits[indices_nok] = self.label_nok
        self.update_measures(self.labels, labels_limits)
        if predict:
            return labels_limits

    def search_next_time_step(self, path):
        """Investigate performance of the next time step

        :param path: path of investigating
        :type path: int
        :return: next f-beta score, time steps and split points
        :rtype: tuple
        """

        next_time_step = self.time_steps[0] - 1 if path == 0 else self.time_steps[-1] + 1
        self.split_points.update(next_time_step)
        next_time_steps = [next_time_step, *self.time_steps] if path == 0 else [*self.time_steps, next_time_step]
        next_all_split_points = [self.split_points.fit(), *self.all_split_points] if path == 0 else [
            *self.all_split_points, self.split_points.fit()]
        if self.shape_limits == 0:
            next_limits = self.construct_limits(next_time_steps, next_all_split_points)
        else:
            next_limits = self.construct_limits_parallel(next_time_steps, next_all_split_points)
        self.fit_limits(next_time_steps, next_limits)
        next_score = self.f_beta_score()
        return next_score, next_time_steps, next_all_split_points

    def fit(self):
        """Greedy search to define control limits beyond threshold

        :return: f-beta score of the control limits and control limits
        :rtype: tuple
        """

        if self.shape_limits == 0:
            limits = self.construct_limits(self.time_steps, self.all_split_points)
        else:
            limits = self.construct_limits_parallel(self.time_steps, self.all_split_points)
        self.fit_limits(self.time_steps, limits)
        start_score = self.f_beta_score()
        while True:
            next_scores = dict(zip(range(0, len(GreedySearch)), len(GreedySearch) * [None]))
            next_time_steps = dict(zip(range(0, len(GreedySearch)), len(GreedySearch) * [None]))
            next_all_split_points = dict(zip(range(0, len(GreedySearch)), len(GreedySearch) * [None]))

            for path in GreedySearch:
                next_scores[path.value], next_time_steps[path.value], next_all_split_points[
                    path.value] = self.search_next_time_step(path.value)

            indices = [key for key, value in next_scores.items() if start_score <= value]

            if indices:
                self.time_steps = next_time_steps[indices[0]]
                self.all_split_points = next_all_split_points[indices[0]]
            else:
                if self.shape_limits == 0:
                    limits = self.construct_limits(self.time_steps, self.all_split_points)
                else:
                    limits = self.construct_limits_parallel(self.time_steps, self.all_split_points)
                self.fit_limits(self.time_steps, limits)
                end_score = self.f_beta_score()
                return end_score, limits


if __name__ == '__main__':
    from pyts.datasets import load_gunpoint

    x_train, x_test, y_train, y_test = load_gunpoint(return_X_y=True)

    gs = GreedySearchLimits(x_train, y_train, 0.95, 5, 0)
    print(gs)
    score, limits = gs.fit()
    print(score)

