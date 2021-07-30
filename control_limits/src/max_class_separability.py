import numpy as np


class StatisticalDistance:
    """Locate the maximum class separability in time-series data

    :param array: input data
    :type array: numpy array
    :param labels: input labels
    :type labels: input labels
    """

    def __init__(self, array, labels):
        self.array = array
        self.labels = labels

    def __repr__(self):
        return 'Time step of maximum class separability.'

    @staticmethod
    def __empirical_cdf(eval_data, data_single_label):
        """Calculate the empirical CDF

        :param eval_data: data for evaluation
        :type eval_data: numpy array
        :param data_single_label: data distribution
        :type data_single_label: numpy array
        :return: empirical cumulative distribution
        :rtype: numpy array
        """

        emp_cdf_total = np.empty(shape=eval_data.size)
        for idx, x in enumerate(eval_data):
            num_of_features, emp_cdf = data_single_label.size, 0
            for instance in data_single_label:
                if instance <= x:
                    emp_cdf += 1
            emp_cdf /= num_of_features
            emp_cdf_total[idx] = emp_cdf
        return emp_cdf_total

    def compute_distance(self):
        """Calculate the total variation distance

        :return: total variation distance
        :rtype: numpy array
        """

        distance = np.empty(shape=(self.array.shape[-1],))
        for idx, arr in enumerate(self.array.T):
            eval_data = np.linspace(np.min(arr), np.max(arr), self.array.shape[0])
            emp_cdf = list()
            for label in np.unique(self.labels):
                emp_cdf.append(self.__empirical_cdf(eval_data, arr[self.labels == label]))
            scale_factor = 0.5
            distance[idx] = scale_factor * sum(abs(emp_cdf[0] - emp_cdf[1]))
        return distance

    def get_start_time_step(self):
        """Calculate the time-step of largest class separability

        :return: time-step of maximum class separability
        :rtype: int
        """

        return np.argmax(self.compute_distance())
