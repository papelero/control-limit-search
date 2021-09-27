import numpy as np

from sklearn.metrics import confusion_matrix
from .greedy_search import GreedySearch
from .utils import get_false_positive, get_false_negative


class ControlLimits(object):
    """Recursive search to derive the control limits

    :param precision_limits: user-input precision
    :param length_limits: user-input length
    :param shape_limits: user-input shape
    """

    def __init__(self, precision_limits, length_limits, shape_limits):
        super(ControlLimits, self).__init__()

        self.precision_limits = precision_limits
        self.length_limits = length_limits
        self.shape_limits = shape_limits

        self.keys = ["false_positive", "false_negative", "time_steps", "control_limits"]

    @staticmethod
    def accuracy(data, labels, output):
        """Return the accuracy of the control limits

        :param data: input data
        :param labels: input labels
        :param output: output of the control limits
        :return: accuracy
        """

        predicted_labels = np.ones(shape=labels.shape, dtype=np.int8)
        for di in range(len(output["time_steps"])):
            data_time_step = data[:, output["time_steps"][di]]
            indices_nok = np.empty(shape=data_time_step.shape)
            for t in range(len(output["time_steps"][di])):
                indices_nok = np.where(np.logical_or(data_time_step[:, t] < output["control_limits"][di][0][t],
                                                     data_time_step[:, t] > output["control_limits"][di][1][t]))
            predicted_labels[indices_nok] = np.unique(labels)[-1]
        tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()
        return round((tn + tp) / (tn + fp + fn + tp), 2)

    @staticmethod
    def update_data_labels(data, labels, predicted_labels):
        """Update the data and labels based on the control limits

        :param data: input data
        :param labels: input labels
        :param predicted_labels: predicted labels
        :return: updated data and labels
        """

        predicted_labels_nok = [di for di, label in enumerate(predicted_labels) if label == np.unique(labels)[-1]]
        return np.delete(data, predicted_labels_nok, axis=0), np.delete(labels, predicted_labels_nok)

    def train(self, data, labels):
        """Training the control limits

        :param data: input data
        :param labels: input labels
        :return: output of training
        """

        train_time_steps, train_control_limits, false_negative, false_positive = [], [], [], []
        search = GreedySearch(data, labels, self.precision_limits, self.length_limits, self.shape_limits)
        time_steps, control_limits, pred_labels = search.solve()
        _, (precision, recall) = search.labels_and_accuracy(time_steps, control_limits)
        score = search.f_beta_score(precision, recall)
        while True:
            train_time_steps += [time_steps]
            train_control_limits += [control_limits]
            if recall == float(1):
                false_negative += [get_false_negative(data, labels, pred_labels)]
                false_positive += [get_false_positive(data, labels, pred_labels)]

                train_output = {key: [] for key in self.keys}
                values = [false_negative, false_positive, train_time_steps, train_control_limits]
                for key, value in zip(self.keys, values):
                    train_output[key] = value
                return train_output
            else:
                data, labels = self.update_data_labels(data, labels, pred_labels)
                search = GreedySearch(data, labels, self.precision_limits, self.length_limits, self.shape_limits)
                next_time_steps, next_control_limits, next_pred_labels = search.solve()
                _, (next_precision, next_recall) = search.labels_and_accuracy(next_time_steps, next_control_limits)
                next_score = search.f_beta_score(next_precision, next_recall)
                if next_score <= score:
                    false_negative += [get_false_negative(data, labels, pred_labels)]
                    false_positive += [get_false_positive(data, labels, pred_labels)]

                    train_output = {key: [] for key in self.keys}
                    values = [false_negative, false_positive, train_time_steps, train_control_limits]
                    for key, value in zip(self.keys, values):
                        train_output[key] = value
                    return train_output
                else:
                    time_steps, control_limits, pred_labels = next_time_steps, next_control_limits, next_pred_labels
                    precision, recall, score = next_precision, next_recall, next_score

    def test(self, data, labels, output):
        """Testing the control limits

        :param data: testing data
        :param labels: testing labels
        :param output: output of training
        :return: output of testing
        """

        false_negative, false_positive = [], []
        search = GreedySearch(data, labels, self.precision_limits, self.length_limits, self.shape_limits)
        for di in range(len(output["time_steps"])):
            predicted_labels, _ = search.labels_and_accuracy(output["time_steps"][di], output["control_limits"][di])
            false_negative += [get_false_negative(data, labels, predicted_labels)]
            false_positive += [get_false_positive(data, labels, predicted_labels)]
            data, labels = self.update_data_labels(data, labels, predicted_labels)

        test_output = {key: [] for key in self.keys}
        values = [false_negative, false_positive, output["time_steps"], output["control_limits"]]
        for key, value in zip(self.keys, values):
            test_output[key] = value
        return test_output
