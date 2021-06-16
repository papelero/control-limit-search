import numpy as np


def assert_input(array, labels):
    """Assert input series and labels

        :param array: input data
        :type array: numpy array
        :param labels: input labels
        :type labels: numpy array
        """

    if not isinstance(array, np.ndarray) or not isinstance(labels, np.ndarray):
        raise AssertionError
    else:
        if len(array.shape) != 2 or len(labels.shape) != 1:
            raise AssertionError
        else:
            if np.unique(labels).size != 2:
                raise AssertionError


def assert_params(array, precision_limits, len_limits, shape_limits):
    """Assert input parameters
    
    :param array: input data
    :type array: numpy array
    :param precision_limits: user-input precision
    :type precision_limits: float
    :param len_limits: user-input length
    :type len_limits: int
    :param shape_limits: user-input shape
    :type: int
    """

    if precision_limits < 0.5 or precision_limits > 1.0:
        raise AssertionError
    elif len_limits <= 1 or len_limits >= array.shape[-1]:
        raise AssertionError
    elif shape_limits < 0 or shape_limits > 1:
        raise AssertionError
