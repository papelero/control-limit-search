import numpy as np
from enum import Enum


class ShiftSplit(Enum):
    """Signal selecting low or high split value."""

    LOW = 0
    HIGH = 1


class DecisionTree:
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

    @property
    def time_step(self):
        """Return the current time step.

        Returns:
            int: Current time step."""

        return self.__time_step

    @time_step.setter
    def time_step(self, value):
        """Reset the time step.

        Args:
            value (int): New time step.

        Returns:
            int: Updated time step."""

        self.__time_step = value

    def __array_time_step__(self):
        """Return data at current time step.

        Returns:
            numpy array: Data at current time step."""

        return self.array[:, self.time_step]

    def __offset__(self):
        """Return offset for shift of global minimum or maximum.

        Returns:
            float: Offset value."""

        return np.abs(np.mean(np.diff(self.array.reshape(self.array.shape[0] * self.array.shape[-1], ))))

    def __min__(self):
        """Return global minimum.

        Returns:
            float: Global minimum value."""

        return np.min(self.array.flatten())

    def __max__(self):
        """Return global maximum.

        Returns:
            float: Global maximum value."""

        return np.max(self.array.flatten())

    def __label_ok__(self):
        """Return the label of the true distribution.

        Returns:
            int: Label value."""

        return np.unique(self.labels)[0]

    def __median_ok__(self):
        """Return median of ok series at given time step.

        Returns:
            float: Median value."""

        return np.median(self.array_time_step[self.labels == self.label_ok])

    def __start_entropy__(self):
        """Return the starting entropy.

        Returns:
            float: Starting entropy."""

        outcomes_start = [len(self.array_time_step[self.labels == label]) for label in np.unique(self.labels)]
        return self.__get_entropy(outcomes_start, len(self.labels))

    @staticmethod
    def __get_entropy(class_count, total):
        """Return the entropy.

        Args:
            class_count (list): Number of instances for each class.
            total (int): Total number of instances.

        Returns:
            float: Entropy."""

        if total != 0:
            p = [i / total for i in class_count]
            if any(np.asarray(p) == 0):
                return float(0)
            else:
                return sum(-i * np.log2(i) for i in p)
        else:
            return float(0)

    def __get_weighted_entropy(self, weights, next_class_count, *, total_next_entropy=True):
        """Return the weighted entropy.

        Args:
            weights (tuple): Weights scaling the entropy of the next iteration.
            next_class_count (tuple): Class count at the next iteration.

        Returns:
            float, list: Weighted entropy."""

        total_next_split = [sum(j for j in i) for i in next_class_count]
        next_entropy = list()
        for idx, weight in enumerate(weights):
            next_entropy.append(weight * self.__get_entropy(next_class_count[idx], total_next_split[idx]))
        if total_next_entropy:
            return sum(h for h in next_entropy)
        else:
            return next_entropy

    def __splits(self, array):
        """Return the possible split values.

        Args:
            array (numpy array): Input data.

        Returns:
            numpy array: Split values."""

        sorted_array = np.sort(np.unique(array))
        start_split, end_split = sorted_array[0] - self.offset, sorted_array[-1] + self.offset
        splits = np.zeros(shape=sorted_array[:-1].size, dtype=np.float32)
        for idx, arr in enumerate(sorted_array[:-1]):
            splits[idx] = arr + (abs((sorted_array[idx + 1]) - arr) / 2)
        return np.pad(splits, (1, 1), 'constant', constant_values=(start_split, end_split))

    def __filter_splits(self, array, labels, cut_point):
        """Return only the splits within the user-defined precision value:

        Args:
            array (numpy array): Input data.
            labels (numpy array): Input labels.
            cut_point (float): User-defined precision value.

        Returns:
            numpy array: Filtered split values."""

        splits = self.__splits(array)
        threshold = np.quantile(array[labels == self.label_ok], cut_point)
        mask = splits < threshold if threshold < self.median_ok else threshold < splits
        return splits[mask]

    def return_optimum_split(self, array, labels, **kwargs):
        """Returns the optimal split point of the input data.

          Args:
              array (numpy array): Input y at one time step.
              labels (numpy array): Input labels.
              kwargs (float): User-defined precision value.

          Returns:
              float: Optimum split value."""

        splits = self.__filter_splits(array, labels, kwargs['precision']) if kwargs else self.__splits(array)
        info_gain = np.zeros(shape=splits.shape)
        for idx, split in enumerate(splits):
            class_count_lower, class_count_upper = list(), list()
            for label in np.unique(labels):
                class_count_lower.append(array[(labels == label) & (array < split)].size)
                class_count_upper.append(array[(labels == label) & (split < array)].size)

            weights = sum(class_count_lower) / labels.size, sum(class_count_upper) / labels.size
            next_class_count = (class_count_lower, class_count_upper)

            next_entropy = self.__get_weighted_entropy(weights, next_class_count)
            info_gain[idx] = self.start_entropy - next_entropy
        return splits[np.argmax(info_gain)]

    def __shift_split(self, split, *, pick_split):
        """Return global minimum/maximum if labels under/over decision boundary are from true distribution.

        Args:
            split (float): Split value.
            pick_split (int): Flag signaling if lower/upper split value.

        Returns:
            float: Shifted split value."""

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
        """Return second split value.

        Returns:
            float: Second split value."""

        split1 = self.return_optimum_split(self.array[:, self.time_step], self.labels, precision=self.precision)
        split2 = self.return_optimum_split(self.array[:, self.time_step], self.labels, precision=(1 - self.precision))
        if split1 < split2:
            return self.__shift_split(split1, pick_split=0), self.__shift_split(split2, pick_split=1)
        else:
            return self.__shift_split(split2, pick_split=0), self.__shift_split(split1, pick_split=1)

    def update(self, time_step):
        """Update the current time step.

        Args:
            time_step (int): Input time step."""

        self.time_step = time_step
        self.array_time_step = self.__array_time_step__()
        self.median_ok = self.__median_ok__()
        self.__start_entropy__()
