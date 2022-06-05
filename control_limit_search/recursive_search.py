# -*- coding: utf-8 -*-

import textwrap

from .utils import *
from .greedy_search import GreedySearch


def update(data, labels, pred_labels):
    """Update data & labels according to control limits

    :param data: input data
    :param labels: input labels
    :param pred_labels: predicted labels
    :return: updated data and labels
    """

    indices_labels_nok = [i for i, label in enumerate(pred_labels) if label == np.unique(labels)[-1]]
    return np.delete(data, indices_labels_nok, axis=0), np.delete(labels, indices_labels_nok)


class ControlLimitsSearch:
    """Recursive search to derive the control limits"""

    keys = ["ts", "cl", "fp", "fn"]

    def __init__(self, cl_precision, cl_len, cl_shape):
        self.cl_precision = cl_precision
        self.cl_len = cl_len
        self.cl_shape = cl_shape

    def __repr__(self):
        if self.cl_shape == 0:
            return textwrap.dedent("""
            The control limits are searched using the following parameters:
            - minimum required precision: {0},
            - minimum required length: {1}, 
            - lines control limits not parallel.""").format(self.cl_precision, self.cl_len)
        else:
            return textwrap.dedent("""
            The control limits are searched using the following parameters:
            - minimum required precision: {0},
            - minimum required length: {1}, 
            - lines control limits parallel.""").format(self.cl_precision, self.cl_len)

    @staticmethod
    def performance(data, labels, pred):
        """Performance of the control limits

        :param data: input data
        :param labels: input labels
        :param pred: prediction of the control limits
        :return: precision, recall and f1-score of the control limits
        """

        pred_labels = np.ones(shape=(labels.size,), dtype=np.int32)
        for i in range(len(pred["cl"])):
            data_of_interest = data[:, pred["ts"][i]]
            for t in range(len(pred["ts"][i])):
                indices_nok = np.where(np.logical_or(data_of_interest[:, t] < pred["cl"][i][0][t],
                                                     data_of_interest[:, t] > pred["cl"][i][1][t]))
                pred_labels[indices_nok] = int(np.unique(labels)[-1])
        precision, recall = precision_recall(labels, pred_labels)
        return round(precision, 2), round(recall, 2), round(2 * ((precision * recall) / (precision + recall)), 2)

    def train(self, data, labels):
        """Training the control limits

        :param data: input data
        :param labels: input labels
        :return: output of training
        """

        # Search for the first set of control limits
        cl_search = GreedySearch(data, labels, self.cl_precision, self.cl_len, self.cl_shape)
        ts, cl, pred_labels, precision, recall = cl_search(return_acc=True)
        acc = get_fbeta_score(self.cl_precision, precision, recall)

        # Recursive search
        train_pred = {k: [] for k in self.keys}
        while True:
            train_pred["ts"] += [ts]
            train_pred["cl"] += [cl]

            # If recall is one, return control limits, false positive and negative and updated data/labels
            if recall == 1.0:
                train_pred["fp"] += [false_positives(data, labels, pred_labels)]
                train_pred["fn"] += [false_negatives(data, labels, pred_labels)]
                next_data, next_labels = update(data, labels, pred_labels)
                return train_pred
            else:
                # Update the data and labels and initialize/perform the next greedy search for control limits
                next_data, next_labels = update(data, labels, pred_labels)
                next_cl_search = GreedySearch(next_data, next_labels, self.cl_precision, self.cl_len,
                                              self.cl_shape)
                next_ts, next_cl, next_pred_labels, next_precision, next_recall = next_cl_search(return_acc=True)
                next_acc = get_fbeta_score(self.cl_precision, next_precision, next_recall)

                # If the accuracy of the next control limits is worse than the previous ones return previous ones
                if (next_acc - acc) < 1e-3 or next_acc == float("nan"):
                    train_pred["fp"] += [false_positives(data, labels, pred_labels)]
                    train_pred["fn"] += [false_negatives(data, labels, pred_labels)]
                    return train_pred

                # Update the search variables
                data, labels, acc = next_data, next_labels, next_acc
                ts, cl, pred_labels, precision, recall = next_ts, next_cl, next_pred_labels, next_precision, next_recall

    def test(self, data, labels, train_pred):
        """Testing the control limits

        :param data: testing data
        :param labels: testing labels
        :param train_pred: output of training
        :return: testing prediction
        """

        test_pred = {k: [] for k in self.keys}
        test_pred["ts"], test_pred["cl"] = train_pred["ts"], train_pred["cl"]
        for i in range(len(train_pred["ts"])):
            pred_labels = predict_labels_limits(data, labels, train_pred["ts"][i], train_pred["cl"][i])
            test_pred["fp"] += [false_positives(data, labels, pred_labels)]
            test_pred["fn"] += [false_negatives(data, labels, pred_labels)]
            data, labels = update(data, labels, pred_labels)
        return test_pred
