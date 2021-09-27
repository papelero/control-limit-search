import numpy as np

from .synthetic_data_generator import SyntheticDataGenerator


def normal_data_labels():
    """Generate synthetic normal data and respective labels

    :return: normal data and labels
    """

    generator = SyntheticDataGenerator(num_data=100, num_time_steps=200)
    normal_data = generator(data_type=0)
    normal_labels = np.ones(len(normal_data), dtype=np.int8)
    return normal_data, normal_labels


def anomaly_data_labels():
    """Generate synthetic anomaly data and respective labels

    :return: anomaly data and labels
    """

    generator = SyntheticDataGenerator(num_data=20, num_time_steps=200)
    anomaly_data = generator(data_type=1)
    anomaly_labels = np.multiply(np.ones(len(anomaly_data), dtype=np.int8), 2)
    return anomaly_data, anomaly_labels


def load_data():

    normal_data, normal_labels = normal_data_labels()
    anomaly_data, anomaly_labels = anomaly_data_labels()

    normal_split, anomaly_split = len(normal_data) // 2, len(anomaly_data) // 2
    train_data = np.concatenate((normal_data[:-normal_split], anomaly_data[:-anomaly_split]))
    train_labels = np.concatenate((normal_labels[:-normal_split], anomaly_labels[:-anomaly_split]))

    test_data = np.concatenate((normal_data[-normal_split:], anomaly_data[-anomaly_split:]))
    test_labels = np.concatenate((normal_labels[-normal_split:], anomaly_labels[-anomaly_split:]))

    return train_data, test_data, train_labels, test_labels