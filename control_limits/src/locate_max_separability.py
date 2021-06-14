import numpy as np


class StatisticalDistance:
    def __init__(self, array, labels):
        self.array = array
        self.labels = labels

    def __repr__(self):
        return 'Time step of maximum class separability.'

    @staticmethod
    def __empirical_cdf(eval_data, data_single_label):
        """Calculate the empirical CDF.

        Args:
            eval_data (numpy array): Data for evaluation.
            data_single_label (numpy array): Data distribution.

        Returns:
            numpy array: Empirical cumulative distribution function"""

        emp_cdf_total = np.empty(shape=eval_data.size)
        for idx, x in enumerate(eval_data):
            num_of_features, emp_cdf = data_single_label.size, 0
            for d in data_single_label:
                if d <= x:
                    emp_cdf += 1
            emp_cdf /= num_of_features
            emp_cdf_total[idx] = emp_cdf
        return emp_cdf_total

    def compute_stat_distance(self):
        """Returns the Total Variation Distance.

        Returns:
            numpy array: Distance between Ok/Nok distribution across all time steps."""

        distance = np.empty(shape=(self.array.shape[-1],))
        for idx, arr in enumerate(self.array.T):
            eval_data = np.linspace(np.min(arr), np.max(arr), self.array.shape[0])
            emp_cdf = list()
            for label in np.unique(self.labels):
                emp_cdf.append(self.__empirical_cdf(eval_data, arr[self.labels == label]))
            scale_factor = 0.5
            distance[idx] = scale_factor * sum(abs(emp_cdf[0] - emp_cdf[1]))
        return distance

    def extract_start_time_step(self):
        """Returns the time step with largest class separability.

        Returns:
            int: Time step with largest class separability."""

        return np.argmax(self.compute_stat_distance())
