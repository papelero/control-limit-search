import numpy as np


class StatisticalDistance:
    """Locate the maximum class separability in time-series data

    :param data: input data
    :param labels: input labels
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    @staticmethod
    def empirical_cdf(data_eval, data_each_label):
        """Empirical cumulative distribution function

        :param data_eval: data for evaluation
        :param data_each_label: data distribution for each label
        :return: empirical cumulative distribution function
        """

        emp_cdf_total = np.empty(shape=data_eval.shape)
        for di, x in enumerate(data_eval):
            x_total, emp_cdf = len(data_each_label), float(0)
            for instance in data_each_label:
                if instance <= x:
                    emp_cdf += float(1)
            emp_cdf_total[di] = (emp_cdf / x_total)
        return emp_cdf_total

    def estimate_stat_distance(self):
        """Evaluate statistical distance of data per total variation distance

        :return: total variation distance and time step of maximum class separability
        """

        dist = np.empty(shape=(self.data.shape[-1],))
        for di, instance in enumerate(self.data.T):
            data_eval = np.linspace(np.min(instance), np.max(instance), self.data.shape[0])
            emp_cdf = {label: None for label in np.unique(self.labels)}
            for label in np.unique(self.labels):
                emp_cdf[label] = self.empirical_cdf(data_eval, instance[self.labels == label])
            dist[di] = 0.5 * sum(abs(emp_cdf[np.unique(self.labels)[0]] - emp_cdf[np.unique(self.labels)[-1]]))
        return dist, np.argmax(dist)
