# -*- coding: utf-8 -*-

import json

import numpy as np

from pathlib import Path


def _from_json(data):
    """Load the data from the JSON file

    :param data: JSON data
    :return: train and test data/labels
    """

    x_train, x_test, y_train, y_test = [], [], [], []
    for k, v in data.items():
        if k == "x_train":
            for i in v:
                x_train += [np.asarray([float(j) for j in i])]
        elif k == "x_test":
            for i in v:
                x_test += [np.asarray([float(j) for j in i])]
        elif k == "y_train":
            y_train += [int(i) for i in v]
        else:
            y_test += [int(i) for i in v]
    return np.asarray(x_train), np.asarray(x_test), np.asarray(y_train), np.asarray(y_test)


def load_data():
    """Load the SonyAIRobot data

    :return: train and test data/labels
    """

    x_train, x_test, y_train, y_test = np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
    path_to_data = Path.cwd() / "datasets" / "sony_ai_robot"
    for i in path_to_data.iterdir():
        if ".json" in str(i):
            with open(path_to_data / i) as file:
                x_train, x_test, y_train, y_test = _from_json(json.load(file))
    return x_train, x_test, y_train, y_test
