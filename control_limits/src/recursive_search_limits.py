import numpy as np
from sklearn.metrics import confusion_matrix
from .greedy_search_limits import GreedySearchLimits
from ..utils import assert_input, assert_params


class ControlLimits:
    """Recursive search to derive the control limits
    
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

    def __init__(self, array, labels, precision_limits=0.95, length_limits=8, shape_limits=1):
        assert_input(array, labels) and assert_params(array, precision_limits, length_limits, shape_limits)
        self.array = array
        self.labels = labels
        self.precision_limits = precision_limits
        self.length_limits = length_limits
        self.shape_limits = shape_limits
        self.label_ok = self.__label_ok__()
        self.label_nok = self.__label_nok__()
        self.keys = ['fn', 'fp', 'time_steps', 'boundaries']
        self.time_steps = list()
        self.limits = list()
        self.output_fit = self.__output__()
        self.output_eval = self.__output__()

    def __repr__(self):
        return f'Empirical control limits(precision={self.precision_limits},' \
               f'length={self.length_limits},' \
               f'shape={self.shape_limits}).'

    def __label_ok__(self):
        """Return the label of the ok distribution
        
        :return: label ok distribution
        :rtype: int
        """

        return np.unique(self.labels)[0]

    def __label_nok__(self):
        """Return the label of the nok distribution
        
        :return: label nok distribution
        :rtype: int
        """

        return np.unique(self.labels)[-1]

    def __output__(self):
        """Initialize the output
        
        :return: initialized output
        :rtype: dict
        """

        return {key: [] for key in self.keys}

    def __return_fp(self, array, labels, labels_limits):
        """Return the false positive data
        
        :param array: input data
        :type array: numpy array
        :param labels: input labels
        :type labels: numpy array
        :param labels_limits: predicted labels
        :type labels_limits: numpy array
        :return: false positive data
        :rtype: numpy array
        """

        labels_limits_ok = labels_limits[np.where(labels == self.label_ok)[0]]
        indices_fp = np.where(labels_limits_ok == self.label_nok)[0]
        return array[indices_fp, :]

    def __return_fn(self, array, labels, labels_limits):
        """Return the false negative data

        :param array: input data
        :type array: numpy array
        :param labels: input labels
        :type labels: numpy array
        :param labels_limits: predicted labels
        :type labels_limits: numpy array
        :return: false negative data
        :rtype: numpy array
        """

        offset = labels[labels == self.label_ok].size
        labels_limits_nok = labels_limits[np.where(labels == self.label_nok)[0]]
        indices_fn = np.where(labels_limits_nok == self.label_ok)[0]
        return array[indices_fn + offset, :]

    def accuracy(self, array, labels, output):
        """Estimate the accuracy of the control limits
        
        :param array: input data
        :type array: numpy array
        :param labels: input labels
        :type labels: numpy array
        :param output: control limits
        :type output: dict
        :return: accuracy of the control limits
        :rtype: float
        """

        time_steps, boundaries = output['time_steps'], output['boundaries']
        limits_labels = np.ones(shape=labels.shape, dtype=np.int8)
        for idx in range(len(time_steps)):
            limits_series = array[:, time_steps[idx]]
            for t in range(len(time_steps[idx])):
                indices_nok = np.where((limits_series[:, t] < boundaries[idx][0][t]) |
                                       (limits_series[:, t] > boundaries[idx][1][t]))
                limits_labels[indices_nok] = self.label_nok

        tn, fp, fn, tp = confusion_matrix(labels, limits_labels).ravel()
        return round((tn + tp) / (tn + fp + fn + tp), 2)

    def __update(self, array, labels, limits_labels):
        """Apply control limits and return updated data and labels

        :param array: input data
        :type array: numpy array
        :param labels: input labels
        :type labels: numpy array
        :param limits_labels: predicted labels
        :type limits_labels: numpy array
        :return: updated data and labels
        :rtype: tuple
        """

        limits_indices_nok = [idx for idx, label in enumerate(limits_labels) if label == self.label_nok]
        return np.delete(array, limits_indices_nok, axis=0), np.delete(labels, limits_indices_nok)

    def fit(self):
        """Fitting on the training data
        
        :return: training output
        :rtype: dict
        """

        greedy_search = GreedySearchLimits(self.array, self.labels, self.precision_limits,
                                           self.length_limits, self.shape_limits)
        labels_limits, time_steps, limits = greedy_search.fit()
        score, recall = greedy_search.f_beta_score(), greedy_search.recall
        fn, fp = list(), list()
        while True:
            self.time_steps += [time_steps]
            self.limits += [limits]
            if recall == float(1):
                fp += [self.__return_fp(self.array, self.labels, labels_limits)]
                fn += [self.__return_fn(self.array, self.labels, labels_limits)]
                values = [fn, fp, self.time_steps, self.limits, score]
                for key, value in zip(self.keys, values):
                    self.output_fit[key] = value
                return self.output_fit
            else:
                self.array, self.labels = self.__update(self.array, self.labels, labels_limits)
                next_greedy_search = GreedySearchLimits(self.array, self.labels, self.precision_limits,
                                                        self.length_limits, self.shape_limits)
                next_labels_limits, next_time_steps, next_limits = next_greedy_search.fit()
                print(next_time_steps)
                next_score, next_recall = next_greedy_search.f_beta_score(), next_greedy_search.recall
                if next_score <= score:
                    fp += [self.__return_fp(self.array, self.labels, labels_limits)]
                    fn += [self.__return_fn(self.array, self.labels, labels_limits)]
                    values = [fn, fp, self.time_steps, self.limits]
                    for key, value in zip(self.keys, values):
                        self.output_fit[key] = value
                    return self.output_fit
            labels_limits, time_steps, limits = next_labels_limits, next_time_steps, next_limits
            score, recall = next_score, next_recall

    def evaluate(self, test_x, test_y, output_fit):
        """Evaluating on the testing data
        
        :param test_x: test data
        :type test_x: numpy array
        :param test_y: test labels
        :type test_y: numpy array
        :param output_fit: training output
        :return: testing output
        :rtype: dict
        """

        assert_input(test_x, test_y)
        fn, fp = list(), list()
        for idx in range(len(output_fit['time_steps'])):
            greedy_search = GreedySearchLimits(test_x, test_y, self.precision_limits,
                                               self.length_limits, self.shape_limits)
            time_steps, limits = output_fit['time_steps'][idx], output_fit['boundaries'][idx]
            limits_labels = greedy_search.update_and_predict(time_steps, limits, return_labels=True)
            fp += [self.__return_fp(test_x, test_y, limits_labels)]
            fn += [self.__return_fn(test_x, test_y, limits_labels)]
            test_x, test_y = self.__update(test_x, test_y, limits_labels)
        values = [fn, fp, output_fit['time_steps'], output_fit['boundaries']]
        for key, value in zip(self.keys, values):
            self.output_eval[key] = value
        return self.output_eval
