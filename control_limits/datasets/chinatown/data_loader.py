import os
import arff

import numpy as np


def load_train_data():
    """Loading the train set:

    :return: train set with data and labels
    """

    path = os.getcwd().split("/")[:-1] + ["control_limits"] + ["datasets"] + ["chinatown"]
    path = "/".join(path)
    for item in os.listdir(path):
        if "TRAIN" in item:
            train_set = arff.load(open(os.path.join(path, item), "rt"))
            return np.asarray(train_set["data"])


def load_test_data():
    """Loading the test set:

    :return: test set with data and labels
    """

    path = os.getcwd().split("/")[:-1] + ["control_limits"] + ["datasets"] + ["chinatown"]
    path = "/".join(path)
    for item in os.listdir(path):
        if "TEST" in item:
            test_set = arff.load(open(os.path.join(path, item), "rt"))
            return np.asarray(test_set["data"])


def load_data():
    """Format the train and test data as well as the labels"""

    train_data_raw = load_train_data()
    train_data = np.zeros(shape=(len(train_data_raw), len(train_data_raw[0]) - 1), dtype=np.float32)
    train_labels = np.zeros(shape=(len(train_data_raw), ), dtype=np.int8)
    for di, instance in enumerate(train_data_raw):
        train_data[di], train_labels[di] = instance[:-1], instance[-1]

    test_data_raw = load_test_data()
    test_data = np.zeros(shape=(len(test_data_raw), len(test_data_raw[0]) - 1), dtype=np.float32)
    test_labels = np.zeros(shape=(len(test_data_raw), ), dtype=np.int8)
    for di, instance in enumerate(test_data_raw):
        test_data[di], test_labels[di] = instance[:-1], instance[-1]

    return train_data, test_data, train_labels, test_labels


if __name__ == "__main__":
    load_train_data()
