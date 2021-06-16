import numpy as np
from enum import Enum


class ShiftSplit(Enum):
    """Signal selecting low or high split value."""

    LOW = 0
    HIGH = 1


class DecisionBoundary:
    """Calculate the decision boundary

    :param array: input data
    :type array: numpy array
    :param labels: input labels
    :type labels: numpy array
    :param time_step: current time-step
    :type time_step: int
    :param precision: current precision
    :type precision: float
    """

    def __init__(self, array, labels, time_step, precision):
        self.array = array
        self.labels = labels
        self.time_step = time_step
        self.precision = precision
        self.array_time_step = self.__array_time_step__()
        self.label_ok = self.__label_ok__()
        self.median_ok = self.__median_ok__()
        self.offset = self.__offset__()
        self.min = self.__min__()
        self.max = self.__max__()
        self.start_entropy = self.__start_entropy__()

    def __repr__(self):
        return f'Decision boundaries(time step={self.time_step}, precision={self.precision})'

    def __array_time_step__(self):
        """Return data at current time step

        :return: data at current time-step
        :rtype: numpy array
        """

        return self.array[:, self.time_step]

    def __offset__(self):
        """Return offset shifting global minimum or maximum

        :return: offset
        :rtype: float
        """

        return np.abs(np.mean(np.diff(self.array.reshape(self.array.shape[0] * self.array.shape[-1], ))))

    def __min__(self):
        """Return global minimum

        :return: global minimum
        :rtype: float
        """

        return np.min(self.array.flatten())

    def __max__(self):
        """Return global maximum

        :return: global maximum
        :rtype: float
        """

        return np.max(self.array.flatten())

    def __label_ok__(self):
        """Return the label of the true distribution

        :return: ok label
        :rtype: int
        """

        return np.unique(self.labels)[0]

    def __median_ok__(self):
        """Return median of ok series at given time step

        :return: median ok data
        :rtype: float
        """

        return np.median(self.array_time_step[self.labels == self.label_ok])

    def __start_entropy__(self):
        """Return the starting entropy

        :return: starting entropy
        :rtype: float
        """

        outcomes_start = [len(self.array_time_step[self.labels == label]) for label in np.unique(self.labels)]
        return self.__get_entropy(outcomes_start, len(self.labels))

    @property
    def time_step(self):
        """Return the current time-step

        :return: current time-step
        :rtype: int
        """

        return self.__time_step

    @time_step.setter
    def time_step(self, value):
        """Set the next time-step.

        :param value: next time-step
        :type value: int
        """

        self.__time_step = value

    @staticmethod
    def __get_entropy(class_count, total_instances):
        """Return the entropy

        :param class_count: count of instances for each class
        :type class_count: list
        :param total_instances: total number of instances
        :type total_instances: int
        :return: entropy
        :rtype: float
        """

        if total_instances != 0:
            class_fraction = [i / total_instances for i in class_count]
            if any(np.asarray(class_fraction) == 0):
                return float(0)
            else:
                return sum(-i * np.log2(i) for i in class_fraction)
        else:
            return float(0)

    def __get_weighted_entropy(self, weights, next_class_count):
        """Return the weighted entropy

        :param weights: weights scaling the entropy of the next nodes
        :type weights: tuple
        :param next_class_count: count of instances for each class of the next nodes
        :type next_class_count: tuple
        :return: weighted next entropy
        :rtype: float
        """

        total_next_split = [sum(j for j in i) for i in next_class_count]
        next_entropy = list()
        for idx, weight in enumerate(weights):
            next_entropy.append(weight * self.__get_entropy(next_class_count[idx], total_next_split[idx]))
        return sum(h for h in next_entropy)


    ### --> continue from here

    def __split_points(self, array):
        """Return all possible split points
        
        :param array: input data
        :type array: numpy array
        :return: split points
        :rtype:numpy array
        """
        
        sorted_array = np.sort(np.unique(array))
        start_split, end_split = sorted_array[0] - self.offset, sorted_array[-1] + self.offset
        split_points = np.add(sorted_array[:-1], np.divide(np.absolute(sorted_array[1:] - sorted_array[:-1]), 2))
        return np.pad(split_points, (1, 1), 'constant', constant_values=(start_split, end_split))

    def __filter_splits(self, array, labels, precision_limits):
        """Return only the split points satisfying the user-input
        
        :param array: input data
        :type array: numpy array
        :param labels: input labels
        :type labels: numpy array
        :param precision_limits: user-input
        :type precision_limits: float
        :return: filtered split points
        :rtype: numpy array
        """

        splits = self.__split_points(array)
        threshold = np.quantile(array[labels == self.label_ok], precision_limits)
        mask = splits < threshold if threshold < self.median_ok else threshold < splits
        return splits[mask]
    
    # here double check of as currently structured is required! 
    def optimal_split_point(self, array, labels, **kwargs):
        """Returns the optimal split point
        
        :param array: input data
        :type array: numpy array
        :param labels: input labels
        :type labels: numpy array
        :return: optimal split point
        :rtype: float
        """
       
        split_points = self.__filter_splits(array, labels, kwargs['precision']) if kwargs else self.__split_points(array)
        info_gain = np.zeros(shape=splits.shape)
        for idx, split in enumerate(split_points):
            class_count_lower, class_count_upper = list(), list()
            for label in np.unique(labels):
                class_count_lower.append(array[(labels == label) & (array < split)].size)
                class_count_upper.append(array[(labels == label) & (split < array)].size)

            weights = sum(class_count_lower) / labels.size, sum(class_count_upper) / labels.size
            next_class_count = (class_count_lower, class_count_upper)

            next_entropy = self.__get_weighted_entropy(weights, next_class_count)
            info_gain[idx] = self.start_entropy - next_entropy
        return split_points[np.argmax(info_gain)]

    def __shift_split_point(self, split, *, pick_split):
        """Shift split point to global minimum or maximum if labels under or over decision boundary ok
        
        :param split: split point
        :type split: float
        :param pick_split: flag to signal if lower or upper split point
        :type pick_split: int
        :return: shifted split point
        :rtype: float
        """
        
        if pick_split == ShiftSplit.LOW.value:
            if all(self.labels[self.array_time_step < split] == self.label_ok):
                return self.min - self.offset
            else:
                return split
        else:
            if all(self.labels[split < self.array_time_step] == self.label_ok):
                return self.max + self.offset
            else:
                return split

    def fit(self):
        """Return the decision boundary
        
        :return: decision boundary 
        :rtype: tuple
        """
       
        split1 = self.optimal_split_point(self.array[:, self.time_step], self.labels, precision=self.precision)
        split2 = self.optimal_split_point(self.array[:, self.time_step], self.labels, precision=(1 - self.precision))
        if split1 < split2:
            return self.__shift_split(split1, pick_split=0), self.__shift_split_point(split2, pick_split=1)
        else:
            return self.__shift_split(split2, pick_split=0), self.__shift_split_point(split1, pick_split=1)

    def update(self, time_step):
        """Update the current time step

        :param time_step: next time-step
        :type time_step: int
        """

        self.time_step = time_step
        self.array_time_step = self.__array_time_step__()
        self.median_ok = self.__median_ok__()
        self.__start_entropy__()
