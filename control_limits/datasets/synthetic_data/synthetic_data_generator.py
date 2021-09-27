import numpy as np

from enum import Enum


class DataType(Enum):
    """Flag to determine which type of data to generate"""

    NORMAL = 0
    ANOMALY = 1


class SyntheticDataGenerator(object):
    """Generate synthetic data

    :param: num_data: number of instance in the data
    :param num_time_steps: number of time steps in the data
    """

    def __init__(self, num_data, num_time_steps):
        assert isinstance(num_data, int) and isinstance(num_time_steps, int)
        self.num_data = num_data
        self.num_time_steps = num_time_steps

        self.added_noise = self.init_noise()

    def init_noise(self):
        """Initialize the added random noise

        :return: random noise scaled by 0.01
        """

        return np.multiply(np.random.randn(self.num_time_steps), 1e-2)

    @property
    def num_data(self):
        """Return the number of instances

        :return: number of instances
        """

        return self._num_data

    @num_data.setter
    def num_data(self, value):
        """Update the number of instances

        :param value: number of instances
        """

        assert isinstance(value, int)
        self._num_data = value

    @property
    def num_time_steps(self):
        """Return the number of time steps

        :return: number of time steps
        """

        return self._num_time_steps

    @num_time_steps.setter
    def num_time_steps(self, value):
        """Update the number of instances

        :param value: number of time steps
        """

        assert isinstance(value, int)
        self._num_time_steps = value

    def added_shift(self):
        """Generate random shift between the data

        :return: random shift
        """

        return np.multiply(np.random.randn(self.num_data), 2.5e-2)

    def __call__(self, data_type):
        """

        :param data_type: type of data to generate
        :return: synthetic data
        """

        if data_type == DataType.NORMAL.value:
            x = np.linspace(-1, 1, self.num_time_steps)
            added_shift = self.added_shift()
            normal_data = np.zeros((self.num_data, self.num_time_steps), dtype=np.float32)
            for di in range(self.num_data):
                normal_data[di] = np.add(np.add(x, self.added_noise), added_shift[di])
            return normal_data
        else:
            x = np.linspace(-5, 5, self.num_time_steps)
            added_shift = self.added_shift()
            anomaly_data = np.zeros((self.num_data, self.num_time_steps), dtype=np.float32)
            for di in range(self.num_data):
                anomaly_data[di] = np.add(np.add(np.sin(x), self.added_noise), added_shift[di])
            return anomaly_data
