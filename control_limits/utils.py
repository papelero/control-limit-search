import numpy as np
from enum import Enum


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
        
                
class GreedySearchPath(Enum):
    """Signal selecting the direction of the greedy search"""

    LEFT = 0
    RIGHT = 1


class ShiftSplitPoint(Enum):
    """Signal selecting low or high split point"""

    LOW = 0
    HIGH = 1

    
class ScalePrecision(object):
    """Return the beta for f-beta score

    :param input_min: minimum possible precision
    :type input_min: float
    :param input_max: maximum possible precision
    :type input_max: float
    :param start: start of transformed interval
    :type start: float
    :param end: end of transformed interval
    :type end: float
    """
    def __init__(self, input_min=0.5, input_max=1.0, start=2.0, end=-2.0):
        self.min = input_min
        self.max = input_max
        self.start = start
        self.end = end

    def __call__(self, precision_limits):
        """Return the beta for f-beta score given the precision of the control limits

        :param precision_limits: user input precision
        :type precision_limits: float
        :return: f-beta score
        :rtype: float
        """
        scaling_factor = (self.end - self.start)
        return scaling_factor * ((precision_limits - self.min) / (self.max - self.min)) + self.start


def euc_distance(array_ref, array):
    """Return the euclidean distance

    :param array_ref: reference data
    :type array_ref: numpy array
    :param array: input data
    :type array: numpy array
    :return: euclidean distance
    :rtype: float
    """

    dist = float(0)
    for i, j in zip(array_ref, array):
        dist += np.sqrt((i - j) ** 2)
    return dist


def linear_regression(time_steps, dec_boundary):
    """Linear regression on the split points

    :param time_steps: decision boundary time-steps
    :type time_steps: numpy array
    :param dec_boundary: decision boundary
    :type dec_boundary: numpy array
    :return: linear regression of the decision boundary
    :rtype: numpy array
    """

    linear_fit = np.poly1d(np.polyfit(time_steps, dec_boundary, 1))
    return linear_fit(time_steps)
