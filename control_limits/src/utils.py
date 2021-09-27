import numpy as np

from enum import Enum
from sklearn.metrics import confusion_matrix


class ShiftSplitPoint(Enum):
    """Flag to determine which split point to consider"""

    BELOW = 0
    ABOVE = 1


class SearchPath(Enum):
    """Flag to determine which split point to consider"""

    LEFT = 0
    RIGHT = 1


def linear_regression(time_steps, decision_boundary):
    """Linear regression on the split points

    :param time_steps: decision boundary time-steps
    :param decision_boundary: decision boundary
    :return: linear regression of the decision boundary
    """

    linear_fit = np.poly1d(np.polyfit(time_steps, decision_boundary, 1))
    return linear_fit(time_steps)


def distance_euclidean(data_ref, data):
    """Return the euclidean distance

    :param data_ref: reference data
    :param data: input data
    :return: euclidean distance
    """

    dist = float(0)
    for i, j in zip(data_ref, data):
        dist += np.sqrt((i - j) ** 2)
    return dist


def precision_and_recall(true_labels, labels_limits):
    """Estimate the precision and recall

    :param true_labels: true labels
    :param labels_limits: predicted labels
    :return: precision and recall
    """

    _, fp, fn, tp = confusion_matrix(true_labels, labels_limits).ravel()
    return tp / (fp + tp), tp / (fn + tp)


def scale_precision(precision_limits, min_scaling, max_scaling, start, end):
    """Scale the precision to the desired interval

    :param precision_limits: desired precision of the control limits
    :param min_scaling: minimum
    :param max_scaling maximum
    :param start: start of the scaling interval
    :param end: end of the scaling interval
    :return: scaled precision
    """

    return (end - start) * ((precision_limits - min_scaling) / (max_scaling - min_scaling)) + start


def get_beta(precision_limits):
    """Return the beta value

    :param precision_limits: desired precision of the control limits
    """

    return 10 ** scale_precision(precision_limits, min_scaling=0.5, max_scaling=1.0, start=2.0, end=-2.0)


def get_false_positive(data, labels, predicted_labels):
    """Return the data false positive

    :param data: input data
    :param labels: input labels
    :param predicted_labels: predicted labels of control limits
    :return: false positive data
    """

    predicted_labels_ok = predicted_labels[np.where(labels == np.unique(labels)[0])][0]
    indices_false_positive = np.where(predicted_labels_ok == np.unique(labels)[-1])[0]
    return data[indices_false_positive, :]


def get_false_negative(data, labels, predicted_labels):
    """Return the data false negative

    :param data: input data
    :param labels: input labels
    :param predicted_labels: predicted labels of control limits
    :return: false negative data
    """

    predicted_labels_nok = predicted_labels[np.where(labels == np.unique(labels)[-1])[0]]
    indices_false_negative = np.where(predicted_labels_nok == np.unique(labels)[0])[0]
    return data[indices_false_negative + len(labels[labels == np.unique(labels)[0]]), :]
