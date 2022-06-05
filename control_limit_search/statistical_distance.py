# -*- coding: utf-8 -*-

import numpy as np


def empirical_cdf(data_evaluation, data_labels):
    """Empirical cumulative distribution function

    :param data_evaluation: data for evaluation
    :param data_labels: data distribution for each label
    :return: empirical cumulative distribution function
    """

    emp_cdf_total = np.empty(shape=data_evaluation.shape)
    for di, x in enumerate(data_evaluation):
        x_total, emp_cdf = len(data_labels), float(0)
        for instance in data_labels:
            if instance <= x:
                emp_cdf += float(1)
        emp_cdf_total[di] = (emp_cdf / x_total)
    return emp_cdf_total


def get_stat_dist(data, labels):
    """Locate the maximum class separability in time-series data

    :param data: input data
    :param labels: input labels
    :return: total variation distance and time step of maximum class separability
    """

    # Define the data for evaluating the cumulative distribution
    data_eval = np.linspace(np.min(data), np.max(data), data.shape[0])

    # Compute the statistical distance and return the time-step where the statistical distance is maximum
    dist = np.empty(shape=(data.shape[-1],))
    for di, instance in enumerate(data.T):
        # Compute the empirical cumulative distribution  for each label
        emp_cdf = {label: None for label in np.unique(labels)}
        for label in np.unique(labels):
            emp_cdf[label] = empirical_cdf(data_eval, instance[labels == label])

        # Compute the total variation distance between the ECDFs of each label
        dist[di] = 0.5 * sum(abs(emp_cdf[np.unique(labels)[0]] - emp_cdf[np.unique(labels)[-1]]))
    return np.argmax(dist)
