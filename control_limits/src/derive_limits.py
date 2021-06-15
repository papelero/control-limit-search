import numpy as np
from enum import Enum
from .locate_max_separability import StatisticalDistance
from .stat_analysis_limits import StatisticalAnalysis
from .decision_tree import DecisionTree


class SearchRoute(Enum):
    """Signal selecting either left or right side of input series w.r.t. starting point."""

    LEFT = 0
    RIGHT = 1


class DeriveLimits(StatisticalDistance, StatisticalAnalysis):
    def __init__(self, array, labels, precision_limits, length_limits, shape_limits):
        StatisticalDistance.__init__(self, array, labels)
        StatisticalAnalysis.__init__(self, precision_limits, precision=float(1e-6), recall=float(1e-6))

        self.start_time_step = self.__start_time_step__()
        self.label_nok = self.__label_nok__()
        self.label_ok = self.__label_ok__()
        self.data_ok = self.__data_ok__()
        self.length_limits = length_limits
        self.thresh_len_limits = self.__threshold__()
        self.shape_limits = shape_limits
        self.len_limits = int(0)
        self.split_values = list()

    def __repr__(self):
        return f'Specification limits(precision={self.precision}, recall={self.recall}, beta={self.beta})'

    def __start_time_step__(self):
        """Return the time step of maximum class separability.

        Returns:
            int: Starting time step."""

        return self.extract_start_time_step()

    def __label_ok__(self):
        """Return the label of the true distribution.

        Returns:
            int: Label of the true distribution."""

        return np.unique(self.labels)[0]

    def __label_nok__(self):
        """Return the label of the faulty distribution.

        Returns:
            int: Label of the true distribution."""

        return np.unique(self.labels)[-1]

    def __threshold__(self):
        """Return the threshold for the side of the specification limits.

        Returns:
            int: Threshold for one side of the specification limits."""

        return self.length_limits // 2

    def __data_ok__(self):
        """Return true data distribution.

        Returns:
            numpy array: True data distribution."""

        return self.array[self.labels == self.label_ok]

    @staticmethod
    def __linear_regression(time_steps, boundary):
        """Perform linear fitting of the boundary.

        Args:
            time_steps (numpy array): Time steps of the boundary.
            boundary (numpy array): Boundary.

        Returns:
              numpy array: Linear fit of the boundary."""

        linear_fit = np.poly1d(np.polyfit(time_steps, boundary, 1))
        return linear_fit(time_steps)

    @staticmethod
    def __extract_boundaries(split_values):
        """Return the lower and upper boundary given the splits.

        Args:
            split_values (list): Split values.

        Returns:
            tuple: Lower and upper boundary."""

        low_dec_bound, up_dec_bound = np.empty(shape=(len(split_values),)), np.empty(shape=(len(split_values),))
        for idx, value in enumerate(split_values):
            low_dec_bound[idx], up_dec_bound[idx] = value[0], value[-1]
        return low_dec_bound, up_dec_bound

    @staticmethod
    def __euclidean_dist(ref_array, input_array):
        """Return the euclidean distance.

        Args:
            ref_array (numpy array): Reference data.
            input_array (numpy array): Input data.

        Returns:
            float: Euclidean distance."""

        euclidean_dist = float(0)
        for i, j in zip(ref_array, input_array):
            euclidean_dist += np.sqrt((i - j).__pow__(2))
        return euclidean_dist

    def __derive_limits(self, search_route, start_time_step, len_limits, split_values):
        """Define the current specification limits.

        Args:
            search_route (SearchRoute): Search route.
            start_time_step (int): Time step of maximum separability.
            len_limits (int): Length of the limits.
            split_values (list): Split values.

        Returns:
            numpy array: Time steps.
            dict: Decision boundaries."""

        if search_route is SearchRoute.LEFT:
            time_steps = np.arange(start_time_step - len_limits, start_time_step + 1)
        elif search_route is SearchRoute.RIGHT:
            time_steps = np.arange(start_time_step, start_time_step + len_limits + 1)
        else:
            raise NotImplementedError

        boundaries = self.__extract_boundaries(split_values)
        if 1 < time_steps.size:
            boundaries_linear = [self.__linear_regression(time_steps, boundary) for boundary in boundaries]
        else:
            boundaries_linear = boundaries

        return time_steps, dict(zip(range(0, len(boundaries_linear)), boundaries_linear))

    def __define_limits_parallel(self, search_route, start_time_step, len_limits, split_values):
        """Define the current specification limits with parallel hyperplanes.

        Args:
            search_route (SearchRoute): Search route.
            start_time_step (int): Time step of maximum separability.
            len_limits (int): Length of limits.
            split_values (list): Split values.

        Returns:
            numpy array: Time steps.
            dict: Decision boundaries."""

        if search_route is SearchRoute.LEFT:
            time_steps = np.arange(start_time_step - len_limits, start_time_step + 1)
        elif search_route is SearchRoute.RIGHT:
            time_steps = np.arange(start_time_step, start_time_step + len_limits + 1)
        else:
            raise NotImplementedError

        boundaries = self.__extract_boundaries(split_values)

        normal_series = np.median(self.data_ok[:, time_steps], axis=0)
        dist_normal_series = [self.__euclidean_dist(normal_series, boundary) for boundary in boundaries]
        idx_min_dist = np.argmin(np.asarray(dist_normal_series)).item()
        offset = abs(np.max(np.asarray(boundaries[0]) - np.asarray(boundaries[1])))

        if 1 < time_steps.size:
            boundaries_linear = [self.__linear_regression(time_steps, boundary) for boundary in boundaries]
            if idx_min_dist == 0:
                boundaries_linear[1] = boundaries_linear[idx_min_dist] + offset
            else:
                boundaries_linear[0] = boundaries_linear[idx_min_dist] - offset
        else:
            boundaries_linear = boundaries

        return time_steps, dict(zip(range(0, len(boundaries_linear)), boundaries_linear))

    def deploy_limits(self, time_steps, boundaries, *, predict=False):
        """Deploy current control limits on the dataset.

        Args:
            time_steps (numpy array): Time steps interval of control limits.
            boundaries (dict): Decision boundaries.
            predict (bool): Flag for returning prediction.

        Returns:
            numpy array: Predicted labels."""

        limits_series = self.array[:, time_steps]
        limits_labels = np.ones(shape=(self.labels.size,), dtype=np.int8)
        for t in range(time_steps.size):
            indices_nok = np.where((limits_series[:, t] < boundaries[0][t]) | (limits_series[:, t] > boundaries[1][t]))
            limits_labels[indices_nok] = self.label_nok
        self.update_measures(self.labels, limits_labels)

        if predict:
            return limits_labels

    def __define_limits_right(self, time_steps, boundaries):
        """Define the right side of the control limits.

        Args:
            time_steps (numpy array): Time steps.
            boundaries (dict): Decision boundaries.

        Returns:
            numpy array: Time steps.
            dict: Decision boundaries."""

        dt = DecisionTree(self.array, self.labels, self.start_time_step, self.precision_limits)
        f_beta_score = self.f_beta_score()
        self.len_limits += 1
        while True:
            if self.array.shape[-1] <= (self.start_time_step + self.len_limits):
                return time_steps, boundaries

            next_time_step = self.start_time_step + self.len_limits
            dt.update(next_time_step)
            self.split_values.append(dt.fit())
            if self.shape_limits == 0:
                next_time_steps, next_boundaries = self.__derive_limits(SearchRoute.RIGHT,
                                                                        self.start_time_step,
                                                                        self.len_limits,
                                                                        self.split_values)
            else:
                next_time_steps, next_boundaries = self.__define_limits_parallel(SearchRoute.RIGHT,
                                                                                 self.start_time_step,
                                                                                 self.len_limits,
                                                                                 self.split_values)
            self.deploy_limits(next_time_steps, next_boundaries)
            next_f_beta_score = self.f_beta_score()

            if self.thresh_len_limits.__mul__(2) < self.len_limits:
                if next_f_beta_score <= f_beta_score:
                    self.deploy_limits(time_steps, boundaries)
                    return time_steps, boundaries

            f_beta_score = next_f_beta_score
            time_steps, boundaries = next_time_steps, next_boundaries
            self.len_limits += 1

    def __define_limits_left(self):
        """Define the left side of the control limits.

        Returns:
            numpy array: Time steps.
            dict: Decision boundaries."""

        dt = DecisionTree(self.array, self.labels, self.start_time_step, self.precision_limits)
        self.split_values.append(dt.fit())
        time_steps, boundaries = self.__derive_limits(SearchRoute.LEFT,
                                                      self.start_time_step,
                                                      self.len_limits,
                                                      self.split_values)
        self.deploy_limits(time_steps, boundaries)
        f_beta_score = self.f_beta_score()
        self.len_limits += 1
        while True:
            if (self.start_time_step - self.len_limits) < 0:
                self.len_limits -= 1
                return self.__define_limits_right(time_steps,
                                                  boundaries)
            next_time_step = self.start_time_step - self.len_limits
            dt.update(next_time_step)
            self.split_values.append(dt.fit())
            if self.shape_limits == 0:
                next_time_steps, next_boundaries = self.__derive_limits(SearchRoute.LEFT,
                                                                        self.start_time_step,
                                                                        self.len_limits,
                                                                        self.split_values)
            else:
                next_time_steps, next_boundaries = self.__define_limits_parallel(SearchRoute.LEFT,
                                                                                 self.start_time_step,
                                                                                 self.len_limits,
                                                                                 self.split_values)
            self.deploy_limits(next_time_steps, next_boundaries)
            next_f_beta_score = self.f_beta_score()
            if self.thresh_len_limits < self.len_limits:
                if next_f_beta_score <= f_beta_score:
                    self.__update_params_limits()
                    self.deploy_limits(time_steps, boundaries)
                    return self.__define_limits_right(time_steps, boundaries)
            f_beta_score = next_f_beta_score
            time_steps, boundaries = next_time_steps, next_boundaries
            self.len_limits += 1

    def __update_params_limits(self):
        """Update the parameters for right side search."""

        self.len_limits = self.len_limits.__sub__(1)
        self.start_time_step = self.start_time_step.__sub__(self.len_limits)
        self.split_values.pop()
        self.split_values.reverse()

    def derive(self, *, predict=False):
        """Define the specification limits.

        Args:
            predict (bool): Flag for returning prediction.

        Returns:
            numpy array: Predicted labels.
            numpy array: Time steps.
            dict: Decision boundaries."""

        time_steps, boundaries = self.__define_limits_left()
        if predict:
            predicted_labels = self.deploy_limits(time_steps, boundaries, predict=predict)
            return predicted_labels, time_steps, boundaries
        else:
            return time_steps, boundaries
