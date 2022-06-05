# -*- coding: utf-8 -*-

import numpy as np

from enum import Enum
from .utils import euclidean_distance, predict_labels_limits, get_fbeta_score
from .statistical_distance import get_stat_dist
from .decision_boundary import DecisionBoundary


class Path(Enum):
    """Flag to determine which split point to consider"""

    LEFT = 0
    RIGHT = 1


def linearize(time_steps, decision_boundary):
    """Linear regression on the split points

    :param time_steps: decision boundary time-steps
    :param decision_boundary: decision boundary
    :return: linear regression of the decision boundary
    """

    linear_fit = np.poly1d(np.polyfit(time_steps, decision_boundary, 1))
    return linear_fit(time_steps)


def get_decision_boundary(sp):
    """Return decision boundary provided the split points

    :param sp: split points
    :return: decision boundary
    """

    decision_boundary = {di: [] for di in range(len(sp[0]))}
    for s in sp:
        for di, value in enumerate(s):
            decision_boundary[di] += [value]
    return decision_boundary


def get_control_limits(ts, split_points):
    """Return the control limits

    :param ts: time steps of control limits
    :param split_points: split points of control limits
    :return: control limits
    """

    decision_boundary = get_decision_boundary(split_points)
    cl = [linearize(ts, value) for _, value in decision_boundary.items()]
    return dict(zip(range(0, len(cl)), cl))


def get_control_limits_parallel(data_ok, ts, sp):
    """Return the control limits parallel

    :param data_ok: data of the ok distribution
    :param ts: time steps of control limits
    :param sp: split points of control limits
    :return: control limits
    """

    decision_boundary = get_decision_boundary(sp)
    cl = [linearize(ts, value) for _, value in decision_boundary.items()]

    # Determine which line of the decision boundary is closer to the normal distribution
    dist = [euclidean_distance(np.median(data_ok[:, ts], axis=0), v) for _, v in decision_boundary.items()]
    idx_min_dist = np.argmin(np.asarray(dist))

    # The other line of the decision boundary is the original line shifted according to the shift
    shift = abs(np.max(np.subtract(np.asarray(decision_boundary[0]), np.asarray(decision_boundary[1]))))
    if idx_min_dist == 0:
        cl[1] = cl[idx_min_dist] + shift
    else:
        cl[0] = cl[idx_min_dist] - shift
    return dict(zip(range(0, len(cl)), cl))


class GreedySearch:
    """Greedy search defining the control limits"""

    search_dict = dict(zip(range(0, len(Path)), len(Path) * [None]))

    def __init__(self, data, labels, precision_cl, length_cl, shape_cl):
        self.decision_boundary = DecisionBoundary(data, labels, precision_cl)

        self.data = data
        self.labels = labels
        self.precision_cl = precision_cl
        self.len_cl = length_cl
        self.shape_cl = shape_cl

        self.label_nok = np.unique(labels)[-1]
        self.data_ok = data[labels == np.unique(labels)[0], :]
        self.min_length = int(np.ceil(self.len_cl / 2))

        self.start_ts = self._start_ts()
        self.ts_limits = self._ts_limits()
        self.sp_limits = self._sp_limits()

    def _start_ts(self):
        """Return the starting time-step

        By definition the starting time-step is the time-step where the statistical distance between the two
        distributions is the largest

        :return: starting time-step
        """

        return get_stat_dist(self.data, self.labels)

    def _ts_limits(self):
        """Return the time steps within the minimum length

        :return: time steps within the minimum length
        """

        if (self.start_ts - self.min_length) < 0:
            interval = range(0, (2 * self.min_length))
        elif (self.start_ts + self.min_length) > self.data.shape[-1]:
            interval = range(self.data.shape[-1] - (2 * self.min_length), self.data.shape[-1])
        else:
            interval = range(self.start_ts - self.min_length, self.start_ts + self.min_length)
        return [i for i in interval]

    def _sp_limits(self):
        """Return the split points within the minimum length

        :return: split points with the minimum length
        """

        return [self.decision_boundary(t) for t in self.ts_limits]

    def get_cl(self, ts, sp, return_acc=False):
        """Determine the control limits and the accuracy

        :param ts: time steps of the control limits
        :param sp: split points of the control limits
        :param return_acc: return only the quality of the predicted labels
        :return: control limits and f-beta score
        """

        if self.shape_cl == 0:
            cl = get_control_limits(ts, sp)
            if return_acc:
                precision, recall = predict_labels_limits(self.data, self.labels, ts, cl, return_acc=True)
                return get_fbeta_score(self.precision_cl, precision, recall)
            else:
                return cl
        else:
            cl = get_control_limits_parallel(self.data_ok, ts, sp)
            if return_acc:
                precision, recall = predict_labels_limits(self.data, self.labels, ts, cl, return_acc=True)
                return get_fbeta_score(self.precision_cl, precision, recall)
            else:
                return cl

    def _get_next_ts(self, search_path, next_time_step):
        """Determine parameters and performance at the next time step

        If the search path is set to zero integrate the next time step and split points on the left, if the search
        path is set one integrate the next time step and split points on the right

        :param search_path: which path to take for the definition of the control limits
        :param next_time_step: next time step
        :return next time steps and next control limits
        """

        if search_path == 0:
            next_ts = [next_time_step, *self.ts_limits]
            next_sp = [self.decision_boundary(next_time_step), *self.sp_limits]
        else:
            next_ts = [*self.ts_limits, next_time_step]
            next_sp = [*self.sp_limits, self.decision_boundary(next_time_step)]
        next_acc = self.get_cl(next_ts, next_sp, return_acc=True)
        return next_ts, next_sp, next_acc

    def __call__(self, return_acc=False):
        """

        :return: time steps, control limits and predicted labels
        """

        acc = self.get_cl(self.ts_limits, self.sp_limits, return_acc=True)
        while True:
            next_acc, next_ts, next_sp = dict(self.search_dict), dict(self.search_dict), dict(self.search_dict)
            for i in range(len(Path)):
                next_time_step = self.ts_limits[0] - 1 if i == 0 else self.ts_limits[-1] + 1
                if (next_time_step >= 0) and (next_time_step <= (self.data.shape[-1] - 1)):
                    next_ts[i], next_sp[i], next_acc[i] = self._get_next_ts(i, next_time_step)

            # Determine which search path has the superior quality measure
            next_path = [k for k, v in next_acc.items() if v is not None and acc <= v]
            if len(next_path) != 0:
                acc = next_acc[next_path[0]]
                self.ts_limits, self.sp_limits = next_ts[next_path[0]], next_sp[next_path[0]]
            else:
                cl = self.get_cl(self.ts_limits, self.sp_limits)
                pred_labels = predict_labels_limits(self.data, self.labels, self.ts_limits, cl)
                if return_acc:
                    precision, recall = predict_labels_limits(self.data, self.labels, self.ts_limits, cl,
                                                              return_acc=True)
                    return self.ts_limits, cl, pred_labels, precision, recall
                else:
                    return self.ts_limits, cl, pred_labels
