# -*- coding: utf-8 -*-

import numpy as np

from enum import Enum


class ShiftSplitPoint(Enum):
    """Flag to determine which split point to consider"""

    BELOW = 0
    ABOVE = 1


def get_entropy(data_labels, data_size):
    """Return the entropy

    :param data_labels: count of instances for each class
    :param data_size: total number of instances
    :return: entropy
    """

    if data_size != 0:
        fraction_each_label = [i / data_size for i in data_labels]
        if any(np.asarray(fraction_each_label) == 0):
            return float(0)
        else:
            return sum(-i * np.log2(i) for i in fraction_each_label)
    else:
        return float(0)


def get_next_entropy(weights, next_data_labels):
    """Return the next entropy

    :param weights: weights scaling the entropy of the next nodes
    :param next_data_labels: count of instances for each class of the next nodes
    :return: weighted next entropy
    """

    next_entropy = []
    next_total_data = [sum(j for j in i) for i in next_data_labels]
    for idx, weight in enumerate(weights):
        next_entropy += [weight * get_entropy(next_data_labels[idx], next_total_data[idx])]
    return sum(h for h in next_entropy)


def get_sp(data):
    """Return all possible split points

    :return: split points
    """

    data_sorted = np.sort(np.unique(data))

    # Start and final split point before the data
    split_start = data_sorted[0] - np.abs(np.mean(np.diff(data)))
    split_end = data_sorted[-1] + np.abs(np.mean(np.diff(data)))

    # Calculate the split point and pad at the start and end with starting and final split point
    sp = np.add(data_sorted[:-1], np.divide(np.absolute(data_sorted[1:] - data_sorted[:-1]), 2))
    return np.pad(sp, (1, 1), 'constant', constant_values=(split_start, split_end))


class DecisionBoundary:
    """Determine the decision boundary"""

    def __init__(self, data, labels, precision):
        self.data = data
        self.labels = labels
        self.precision = precision

        self.min, self.max = np.min(data.flatten()), np.max(data.flatten())
        self.label_ok = np.unique(labels)[0]
        self.precision_reversed = 1 - precision

    def start_entropy(self, data):
        """Return the starting entropy

        :param data: input data considered
        :return: starting entropy
        """

        data_each_label = [len(data[self.labels == label]) for label in np.unique(self.labels)]
        return get_entropy(data_each_label, len(data))

    def optimal_sp(self, data, precision, *args):
        """Returns the optimal split point

        :param data: input data considered
        :param precision: input precision
        :param args: first split point if available
        :return: optimal split point
        """

        # Determine the starting entropy
        start_entropy = self.start_entropy(data)

        # Filter split points according to the desired precision
        sp_filtered = self.filter_sp(data, precision, args)

        # Determine the information gain considering only the spit point available
        info_gain = np.zeros(shape=sp_filtered.shape)
        for idx, sp in enumerate(sp_filtered):
            data_each_label_below, data_each_label_above = [], []

            # Determine the size of the data below/above the split point for each label
            for label in np.unique(self.labels):
                data_each_label_below += [len(data[np.logical_and(self.labels == label, data < sp)])]
                data_each_label_above += [len(data[np.logical_and(self.labels == label, sp < data)])]

            # Calculate the weights based on the size of the data
            weights = sum(data_each_label_below) / len(self.labels), sum(data_each_label_above) / len(self.labels)
            next_data_each_label = (data_each_label_below, data_each_label_above)

            # Calculate the entropy of the splitting event
            next_entropy = get_next_entropy(weights, next_data_each_label)

            # Determine the information gain of the splitting event
            info_gain[idx] = start_entropy - next_entropy
        return sp_filtered[np.argmax(info_gain)]

    def filter_sp(self, data, precision, *args):
        """Filter the split points based on the provided precision value

        :param data: input data considered
        :param precision: input precision
        :param args: first split point if available
        :return: filtered split points
        """

        sp = get_sp(data)
        median_ok = np.median(data[self.labels == self.label_ok])
        threshold = np.quantile(data[self.labels == self.label_ok], precision)

        # Utilize the available split point to determine where the next split point is located.
        if args[0]:
            if args[0] < median_ok:
                return sp[threshold < sp]
            else:
                return sp[sp < threshold]
        else:
            if threshold < median_ok:
                return sp[sp < threshold]
            else:
                return sp[threshold < sp]

    def global_shift(self, data, sp, which_sp):
        """Shift split point to global minimum/maximum if labels below/above decision boundary ok

        :param data: input data considered
        :param sp: split point
        :param which_sp: flag to determine which split point to shift
        :return: shifted split point
        """

        if which_sp == ShiftSplitPoint.BELOW.value:
            if all(self.labels[data < sp] == self.label_ok):
                return self.min - np.abs(np.median(np.diff(data)))
            else:
                return sp
        else:
            if all(self.labels[sp < data] == self.label_ok):
                return self.max + np.abs(np.median(np.diff(data)))
            else:
                return sp

    def __call__(self, ts):
        """

        :param ts: current time step
        :return: split points at the current time step
        """

        # Access the local data at the desired time step
        data_ts = self.data[:, ts]

        # Determine the two split points
        sp1 = self.optimal_sp(data_ts, self.precision)
        sp2 = self.optimal_sp(data_ts, self.precision_reversed, sp1)

        # Check if split points can be shifted to the global minimum and maximum and return the output
        if sp1 < sp2:
            sp1_shifted, sp2_shifted = self.global_shift(data_ts, sp1, 0), self.global_shift(data_ts, sp2, 1)
            return sp1_shifted, sp2_shifted
        else:
            sp1_shifted, sp2_shifted = self.global_shift(data_ts, sp1, 1), self.global_shift(data_ts, sp2, 0)
            return sp2_shifted, sp1_shifted
