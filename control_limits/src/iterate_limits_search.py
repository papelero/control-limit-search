import numpy as np
from sklearn.metrics import confusion_matrix
from .derive_limits import DeriveLimits


class ControlLimits:
    def __init__(self, array, labels, precision_limits=0.95, length_limits=8, shape_limits=1):
        self.array = array
        self.labels = labels
        self.precision_limits = precision_limits
        self.length_limits = length_limits
        self.shape_limits = shape_limits
        self.label_ok = self.__label_ok__()
        self.label_nok = self.__label_nok__()
        self.keys = ['fn', 'fp', 'time_steps', 'boundaries']
        self.time_steps = list()
        self.boundaries = list()
        self.output_fit = self.__output__()
        self.output_eval = self.__output__()

    def __repr__(self):
        return f'Empirical control limits(precision={self.precision_limits},' \
               f'length={self.length_limits},' \
               f'shape={self.shape_limits}).'

    def __label_ok__(self):
        """Return the label of the ok distribution.

        Returns:
            int: Label of the true distribution."""

        return np.unique(self.labels)[0]

    def __label_nok__(self):
        """Return the label of the nok distribution.

        Returns:
            int: Label of the faulty distribution."""

        return np.unique(self.labels)[-1]

    def __output__(self):
        """Initialize the output.

        Returns:
            dict: Initialized output dictionary."""

        return {key: [] for key in self.keys}

    @staticmethod
    def __assert_input(array, labels):
        """Assert input series and labels.

        Args:
            array (numpy array): Input data.
            labels (numpy array): Input labels."""

        if not isinstance(array, np.ndarray) or not isinstance(labels, np.ndarray):
            raise TypeError('Input series and labels numpy array.')
        else:
            if len(array.shape) != 2 or len(labels.shape) != 1:
                raise ValueError('Input series 2D, input labels 1D')
            else:
                if np.unique(labels).size != 2:
                    raise ValueError('Input labels binary')

    @staticmethod
    def __assert_params(array, precision_limits, len_limits, shape_limits):
        """Assert input parameters.

        Args:
            array (numpy array): Input data.
            precision_limits (float): User-input precision.
            len_limits (int): Length of empirical specification limits.
            shape_limits (int): Shape of the empirical specification limits."""

        if precision_limits < 0.5 or precision_limits > 1.0:
            raise ValueError('Precision between 0.5 and 1.')
        elif len_limits <= 1 or len_limits >= array.shape[-1]:
            raise ValueError('Length of limits larger than 1 and smaller than total time steps.')
        elif shape_limits < 0 or shape_limits > 1:
            raise ValueError('Shape 0 if not parallel, 1 if parallel.')

    def __return_fp(self, array, labels, predicted_labels):
        """Return the false positive.

        Args:
            array: Input data.
            labels: Input labels.
            predicted_labels: Predicted labels.

        Returns:
            numpy array: False positive series."""

        predicted_ok = predicted_labels[np.where(labels == self.label_ok)[0]]
        indices_fp = np.where(predicted_ok == self.label_nok)[0]
        return array[indices_fp, :]

    def __return_fn(self, array, labels, predicted_labels):
        """Return the false negative.

        Args:
            array: Input data.
            labels: Input labels.
            predicted_labels: Predicted labels.

        Returns:
            numpy array: False negative series."""

        offset = labels[labels == self.label_ok].size
        predicted_nok = predicted_labels[np.where(labels == self.label_nok)[0]]
        indices_fn = np.where(predicted_nok == self.label_ok)[0]
        return array[indices_fn + offset, :]

    def accuracy(self, array, labels, output):
        """Return the accuracy of the defined specification limits.

        Args:
            array (numpy array): Input data.
            labels (numpy array): Input labels.
            output (dict): Empirical control limits.

        Returns:
            float: Accuracy."""

        time_steps, boundaries = output['time_steps'], output['boundaries']
        limits_labels = np.ones(shape=labels.shape, dtype=np.int8)
        for idx in range(len(time_steps)):
            limits_series = array[:, time_steps[idx]]
            for t in range(time_steps[idx].size):
                indices_nok = np.where((limits_series[:, t] < boundaries[idx][0][t]) |
                                       (limits_series[:, t] > boundaries[idx][1][t]))
                limits_labels[indices_nok] = self.label_nok

        tn, fp, fn, tp = confusion_matrix(labels, limits_labels).ravel()
        return round((tn + tp) / (tn + fp + fn + tp), 2)

    def fit(self):
        """Fit the empirical control limits on train_data.

        Returns:
            dict: Fitting results."""

        self.__assert_input(self.array, self.labels)
        self.__assert_params(self.array, self.precision_limits, self.length_limits, self.shape_limits)

        limits = DeriveLimits(self.array, self.labels, self.precision_limits, self.length_limits, self.shape_limits)
        predicted_labels, time_steps, boundaries = limits.derive(predict=True)
        f_beta_score, recall = limits.f_beta_score(), limits.recall
        fn, fp = list(), list()
        while True:
            self.__store_limits(time_steps, boundaries)
            if recall == float(1):
                fp += [self.__return_fp(self.array, self.labels, predicted_labels)]
                fn += [self.__return_fn(self.array, self.labels, predicted_labels)]
                values = [fn, fp, self.time_steps, self.boundaries, f_beta_score]
                for key, value in zip(self.keys, values):
                    self.output_fit[key] = value
                return self.output_fit
            else:
                self.array, self.labels = self.__deploy(self.array, self.labels, predicted_labels)
                next_limits = DeriveLimits(self.array, self.labels, self.precision_limits, self.length_limits,
                                           self.shape_limits)
                next_labels_spec_limits, next_time_steps, next_boundaries = next_limits.derive(predict=True)
                next_f_beta_score, next_recall = next_limits.f_beta_score(), next_limits.recall
                if next_f_beta_score <= f_beta_score:
                    fp += [self.__return_fp(self.array, self.labels, predicted_labels)]
                    fn += [self.__return_fn(self.array, self.labels, predicted_labels)]
                    values = [fn, fp, self.time_steps, self.boundaries]
                    for key, value in zip(self.keys, values):
                        self.output_fit[key] = value
                    return self.output_fit
            predicted_labels, time_steps, boundaries = next_labels_spec_limits, next_time_steps, next_boundaries
            f_beta_score, recall = next_f_beta_score, next_recall

    def evaluate(self, test_x, test_y, output):
        """Evaluate the defined empirical control limits on the test data.

        Args:
            test_x (numpy array): Test data.
            test_y (numpy array): Test labels.
            output (dict): Output fitting.

        Returns:
            dict: Output evaluation."""

        self.__assert_input(test_x, test_y)

        fn, fp = list(), list()
        for idx in range(len(output['time_steps'])):
            limits = DeriveLimits(test_x, test_y, self.precision_limits, self.length_limits, self.shape_limits)
            time_steps, boundaries = output['time_steps'][idx], output['boundaries'][idx]
            predicted_labels = limits.deploy_limits(time_steps, boundaries, predict=True)
            fp += [self.__return_fp(test_x, test_y, predicted_labels)]
            fn += [self.__return_fn(test_x, test_y, predicted_labels)]
            test_x, test_y = self.__deploy(test_x, test_y, predicted_labels)
        values = [fn, fp, output['time_steps'], output['boundaries']]
        for key, value in zip(self.keys, values):
            self.output_eval[key] = value
        return self.output_eval

    def __store_limits(self, time_steps, boundaries):
        """Add the next specification limits defined.

        Args:
            time_steps (numpy array): Time steps.
            boundaries (dict): Decision boundaries."""

        self.time_steps += [time_steps]
        self.boundaries += [boundaries]

    def __deploy(self, array, labels, limits_labels):
        """Update the data and labels according to prediction of specification limits.

        Args:
            array (numpy array): Input array
            labels (numpy array): Input labels.
            limits_labels (numpy array): Predicted labels.

        Returns:
            tuple: Updated array and labels"""

        limits_indices_nok = [idx for idx, label in enumerate(limits_labels) if label == self.label_nok]
        return np.delete(array, limits_indices_nok, axis=0), np.delete(labels, limits_indices_nok)
