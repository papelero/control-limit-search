import numpy as np


class ExampleSeries:
    def __init__(self, size=25, num_of_series=50, num_of_steps=150):
        self.size = size
        self.num_of_series = num_of_series
        self.num_of_steps = num_of_steps

    def __repr__(self):
        return f'SyntheticSeries(size={self.size}, ' \
               f'num_of_series={self.num_of_series}, ' \
               f'num_of_steps={self.num_of_steps})'

    def __get_normal_series(self):
        """Generate synthetic normal data.

        Returns:
            numpy array: Synthetic normal data."""

        normal_data = np.zeros((self.num_of_series, self.num_of_steps))
        for idx in range(self.num_of_series):
            noise, shift = 1e-2 * np.random.randn(self.num_of_steps), 1e-1 * np.random.randn(1)
            normal_data[idx] = (np.linspace(-1, 1, self.num_of_steps) + noise) + shift
        return normal_data

    def __get_abnormal_series(self):
        """Generate synthetic abnormal data.

            Returns:
                numpy array: Synthetic abnormal data."""

        abnormal_data = np.zeros((self.num_of_series, self.num_of_steps))
        for idx in range(self.num_of_series):
            noise, shift = 1e-2 * np.random.randn(self.num_of_steps), 1e-1 * np.random.randn(1)
            abnormal_data[idx] = (np.sin(np.linspace(0, 20, self.num_of_steps)) + noise) + shift
        return abnormal_data

    def __split_train_test_series(self, normal_series, abnormal_series):
        """Split in train and test series.

        Args:
            normal_series (numpy array): Normal series.
            abnormal_series (numpy array): Abnormal series.

        Returns:
            tuple: train data, test data
        """

        x_train = np.concatenate((normal_series[:self.size], abnormal_series[:self.size]), axis=0)
        x_test = np.concatenate((normal_series[self.size:], abnormal_series[self.size:]), axis=0)
        return x_train, x_test

    def __get_labels(self):
        """Generate labels.

        Returns:
            tuple: train labels, test_labels
        """

        y_train = np.concatenate((np.ones(self.size), np.multiply(np.ones(self.size), 2)), axis=0)
        y_test = np.concatenate((np.ones(self.size), np.multiply(np.ones(self.size), 2)), axis=0)
        return y_train, y_test

    def generate(self, return_X_y=True):
        """Generate synthetic data for example.
        """
        normal_series, abnormal_series = self.__get_normal_series(), self.__get_abnormal_series()

        x_train, x_test = self.__split_train_test_series(normal_series, abnormal_series)
        y_train, y_test = self.__get_labels()

        if return_X_y:
            return x_train, x_test, y_train, y_test
