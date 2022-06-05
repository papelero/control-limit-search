# -*- coding: utf-8 -*-

import numpy as np

from sklearn.metrics import confusion_matrix


def euclidean_distance(x0, x1):
    """Return the euclidean distance

    :param x0: reference data
    :param x1: input data
    :return: euclidean distance
    """

    dist = float(0)
    for i, j in zip(x0, x1):
        dist += np.sqrt((i - j) ** 2)
    return dist


def precision_recall(true_labels, labels_limits):
    """Estimate the precision and recall

    :param true_labels: true labels
    :param labels_limits: predicted labels
    :return: precision and recall
    """

    _, fp, fn, tp = confusion_matrix(true_labels, labels_limits).ravel()
    return tp / (fp + tp), tp / (fn + tp)


def predict_labels_limits(data, labels, ts, control_limits, return_acc=False):
    """Predict the labels of the control limits and optionally return its quality

    Predicted labels set to one for the data that is included within the control limits, otherwise predicted labels
    set to two

    :param data: input data
    :param labels: input labels
    :param ts: time steps of the control limits
    :param control_limits: control limits
    :param return_acc: return only the quality of the predicted labels
    :return: labels control limits and precision and recall
    """

    data_limits = data[:, ts]
    labels_limits = np.ones(shape=(labels.size,), dtype=np.int32)
    for di in range(len(ts)):
        indices_nok = np.where(
            np.logical_or(data_limits[:, di] < control_limits[0][di], data_limits[:, di] > control_limits[1][di]))
        labels_limits[indices_nok] = np.unique(labels)[-1]
    if return_acc:
        precision, recall = precision_recall(labels, labels_limits)
        return precision, recall
    else:
        return labels_limits


def get_beta(precision):
    """Return the beta value

    :param precision: desired precision of the control limits
    :return: beta value for f-beta score
    """

    min_, max_ = 0.0, 1.0
    start_value, end_value = 2.0, -2.0
    scaled_precision = (end_value - start_value) * ((precision - min_) / (max_ - min_)) + start_value
    return 10 ** scaled_precision


def get_fbeta_score(precision_limits, precision, recall):
    """Estimate the f-beta score

    :param precision_limits: user-input precision
    :param precision: precision control limits
    :param recall: recall control limits
    :return: f-beta score
    """

    if precision == 0.0 and recall == 0.0:
        return 0.0
    else:
        beta = get_beta(precision_limits)
        try:
            value = (1 + (beta ** 2)) * (precision * recall / ((beta ** 2) * precision + recall))
        except ZeroDivisionError:
            value = float(0)
        return value


def false_positives(data, labels, predicted_labels):
    """Return the false positive data

    :param data: input data
    :param labels: input labels
    :param predicted_labels: predicted labels of control limits
    :return: false positive data
    """

    idx_labels = list(np.where(labels == np.unique(labels)[0])[0])
    idx_pred_labels = list(np.where(predicted_labels == np.unique(labels)[0])[0])

    false_positive = []
    for idx in idx_labels:
        if idx not in idx_pred_labels:
            false_positive += [data[idx, :]]
    return np.asarray(false_positive)


def false_negatives(data, labels, predicted_labels):
    """Return the false negative data

    :param data: input data
    :param labels: input labels
    :param predicted_labels: predicted labels of control limits
    :return: false negative data
    """

    idx_labels = list(np.where(labels == np.unique(labels)[-1])[0])
    idx_pred_labels = list(np.where(predicted_labels == np.unique(labels)[-1])[0])

    false_negative = []
    for idx in idx_labels:
        if idx not in idx_pred_labels:
            false_negative += [data[idx, :]]
    return np.asarray(false_negative)
