import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def assert_input(array, labels):
        """Assert input series and labels.

        Args:
            array (numpy array): Input data.
            labels (numpy array): Input labels."""

        if not isinstance(array, np.ndarray) or not isinstance(labels, np.ndarray):
            raise AssertionError
        else:
            if len(array.shape) != 2 or len(labels.shape) != 1:
                raise AssertionError
            else:
                if np.unique(labels).size != 2:
                    raise AssertionError
                    
def assert_params(array, precision_limits, len_limits, shape_limits):
    """Assert input parameters.

    Args:
        array (numpy array): Input data.
        precision_limits (float): User-input precision.
        len_limits (int): Length of empirical specification limits.
        shape_limits (int): Shape of the empirical specification limits."""

    if precision_limits < 0.5 or precision_limits > 1.0:
        raise AssertionError
    elif len_limits <= 1 or len_limits >= array.shape[-1]:
        raise AssertionError
    elif shape_limits < 0 or shape_limits > 1:
        raise AssertionError
