import numpy as np
import arff
import os
import pkg_resources


def load_train():
    """From current path load the train y.

    Returns:
        numpy array: Train y and labels.
    """

    for item in pkg_resources.resource_listdir('control_limits', 'datasets/gunpoint_oldversusyoung'):
        if '_TRAIN.arff' in item:
            path = pkg_resources.resource_filename('control_limits', 'datasets/gunpoint_oldversusyoung')
            dataset = arff.load(open(os.path.join(path, item), 'rt'))
            return np.array(dataset['data'])


def load_test():
    """From current path load the test y.

    Returns:
        numpy array: Test y and labels.
    """

    for item in pkg_resources.resource_listdir('control_limits', 'datasets/gunpoint_oldversusyoung'):
        if '_TEST.arff' in item:
            path = pkg_resources.resource_filename('control_limits', 'datasets/gunpoint_oldversusyoung')
            data_set = arff.load(open(os.path.join(path, item), 'rt'))
            return np.array(data_set['data'])


def load_data(return_X_y=True):
    """Reshape the train and test y into two-dimensional arrays and separate y from labels.

    Args:
        return_X_y (bool): Flag to signal whether to return the y and labels.

    Returns:
        numpy array: Train y.
        numpy array: Test y.
        numpy array: Train labels.
        numpy array: Test labels.
    """

    # Train y
    load_train_data = load_train()
    train_data = np.zeros(shape=(len(load_train_data), len(load_train_data[0]) - 1))
    train_labels = np.zeros(shape=(len(load_train_data),), dtype=np.int8)
    for ind in range(len(load_train_data)):
        train_data[ind, :] = load_train_data[ind][:-1]
        train_labels[ind] = load_train_data[ind][-1]

    # Test y
    load_test_data = load_test()
    test_data = np.zeros(shape=(len(load_test_data), len(load_test_data[0]) - 1))
    test_labels = np.zeros(shape=(len(load_test_data),), dtype=np.int8)
    for ind in range(len(load_test_data)):
        test_data[ind, :] = load_test_data[ind][:-1]
        test_labels[ind] = load_test_data[ind][-1]

    if return_X_y:
        return train_data, test_data, train_labels, test_labels