import numpy as np

from .utils import ShiftSplitPoint


class DecisionBoundary(object):
    """Determine the decision boundary

    :param data: input data
    :param labels: input labels
    :param precision: desired precision for the control limits
    """

    def __init__(self, data, labels, precision):
        super(DecisionBoundary, self).__init__(data, labels)

        self.data = data
        self.labels = labels
        self.precision = precision

        self.min, self.max = self._min_max()
        self.label_ok = self._label_ok()
        self.precision_reversed = self._precision_reversed()

    @staticmethod
    def entropy(data_each_label, data_total):
        """Return the entropy

        :param data_each_label: count of instances for each class
        :param data_total: total number of instances
        :return: entropy
        """

        if data_total != 0:
            fraction_each_label = [i / data_total for i in data_each_label]
            if any(np.asarray(fraction_each_label) == 0):
                return float(0)
            else:
                return sum(-i * np.log2(i) for i in fraction_each_label)
        else:
            return float(0)

    @staticmethod
    def split_points(data):
        """Return all possible split points

        :return: split points
        """

        data_sorted = np.sort(np.unique(data))

        # Starting split point before the data
        split_start = data_sorted[0] - np.abs(np.mean(np.diff(data)))

        # Final split point exceeding the data
        split_end = data_sorted[-1] + np.abs(np.mean(np.diff(data)))

        # Calculate the split point and pad at the start and end with starting and final split point
        split_points = np.add(data_sorted[:-1], np.divide(np.absolute(data_sorted[1:] - data_sorted[:-1]), 2))
        return np.pad(split_points, (1, 1), 'constant', constant_values=(split_start, split_end))

    def _min_max(self):
        """Return minimum and maximum of input data

        :return: minimum and maximum
        """

        return np.min(self.data.flatten()), np.max(self.data.flatten())

    def _label_ok(self):
        """Return label of ok data

        :return: label of ok data
        """

        return np.unique(self.labels)[0]

    def _precision_reversed(self):
        """Return the revered precision value

        :return: reversed precision
        """

        return 1 - self.precision

    def start_entropy(self, data):
        """Return the starting entropy

        :param data: input data considered
        :return: starting entropy
        """

        data_each_label = [len(data[self.labels == label]) for label in np.unique(self.labels)]
        return self.entropy(data_each_label, len(data))

    def next_entropy(self, weights, next_data_each_label):
        """Return the next entropy

        :param weights: weights scaling the entropy of the next nodes
        :param next_data_each_label: count of instances for each class of the next nodes
        :return: weighted next entropy
        """

        next_total_data = [sum(j for j in i) for i in next_data_each_label]
        next_entropy = []
        for idx, weight in enumerate(weights):
            next_entropy += [weight * self.entropy(next_data_each_label[idx], next_total_data[idx])]
        return sum(h for h in next_entropy)

    def optimal_split_point(self, data, precision, *args):
        """Returns the optimal split point

        :param data: input data considered
        :param precision: input precision
        :param args: first split point if available
        :return: optimal split point
        """

        # Determine the starting entropy
        start_entropy = self.start_entropy(data)

        # Filter split points according to the desired precision
        split_points_filtered = self.filter_split_points(data, precision, args)

        # Determine the information gain considering only the spit point available
        info_gain = np.zeros(shape=split_points_filtered.shape)
        for idx, split_point in enumerate(split_points_filtered):
            data_each_label_below, data_each_label_above = [], []

            # Determine the size of the data below/above the split point for each label
            for label in np.unique(self.labels):
                data_each_label_below += [len(data[np.logical_and(self.labels == label, data < split_point)])]
                data_each_label_above += [len(data[np.logical_and(self.labels == label, split_point < data)])]

            # Calculate the weights based on the size of the data
            weights = sum(data_each_label_below) / len(self.labels), sum(data_each_label_above) / len(self.labels)
            next_data_each_label = (data_each_label_below, data_each_label_above)

            # Calculate the entropy of the splitting event
            next_entropy = self.next_entropy(weights, next_data_each_label)

            # Determine the information gain of the splitting event
            info_gain[idx] = start_entropy - next_entropy
        return split_points_filtered[np.argmax(info_gain)]

    def filter_split_points(self, data, precision, *args):
        """Filter the split points based on the provided precision value

        :param data: input data considered
        :param precision: input precision
        :param args: first split point if available
        :return: filtered split points
        """

        split_points = self.split_points(data)
        median_ok = np.median(data[self.labels == self.label_ok])
        threshold = np.quantile(data[self.labels == self.label_ok], precision)

        # Utilize the available split point to determine where the next split point is located. This is mandatory for
        # the case the desired precision for the control limits is 0.5 because the threshold is that case also the
        # median and the distinction is ambiguous.
        if args[0]:
            if args[0] < median_ok:
                return split_points[threshold < split_points]
            else:
                return split_points[split_points < threshold]

        # Determine the next split point using desired precision for the control limits
        else:
            if threshold < median_ok:
                return split_points[split_points < threshold]
            else:
                return split_points[threshold < split_points]

    def shift_optimal_split_point(self, data, split_point, which_split_point):
        """Shift split point to global minimum/maximum if labels below/above decision boundary ok

        :param data: input data considered
        :param split_point: split point
        :param which_split_point: flag to determine which split point to shift
        :return: shifted split point
        """

        if which_split_point == ShiftSplitPoint.BELOW.value:
            if all(self.labels[data < split_point] == self.label_ok):
                return self.min - np.abs(np.mean(np.diff(data)))
            else:
                return split_point
        else:
            if all(self.labels[split_point < data] == self.label_ok):
                return self.max + np.abs(np.mean(np.diff(data)))
            else:
                return split_point

    def __call__(self, time_step):
        """

        :param time_step: current time step
        :return: split points at the current time step
        """

        # Access the local data at the desired time step
        data_time_step = self.data[:, time_step]

        # Determine the two split points
        split_point1 = self.optimal_split_point(data_time_step, self.precision)
        split_point2 = self.optimal_split_point(data_time_step, self.precision_reversed, split_point1)

        # Check if split points can be shifted to the global minimum and maximum and return the output
        if split_point1 < split_point2:
            split_point1_shifted = self.shift_optimal_split_point(data_time_step, split_point1, 0)
            split_point2_shifted = self.shift_optimal_split_point(data_time_step, split_point2, 1)
            return split_point1_shifted, split_point2_shifted
        else:
            split_point1_shifted = self.shift_optimal_split_point(data_time_step, split_point1, 1)
            split_point2_shifted = self.shift_optimal_split_point(data_time_step, split_point2, 0)
            return split_point2_shifted, split_point1_shifted
