from sklearn.metrics import confusion_matrix


def scale_precision_input(precision_limits, input_min=0.5, input_max=1.0, start_interval=2.0, end_interval=-2.0):
    """Scale user-input to derive beta for F-beta score.

    Args:
        precision_limits (float): User-input.
        input_min (float): Minimum possible precision.
        input_max (float): Maximum possible precision.
        start_interval (float): Start of transformed interval.
        end_interval (float): End of transformed interval.

    Returns:
        float: Scaled user-input."""

    scaling_factor = (end_interval - start_interval)
    return scaling_factor * ((precision_limits - input_min) / (input_max - input_min)) + start_interval


class StatisticalAnalysis:
    def __init__(self, precision_limits, precision, recall):
        self.precision_limits = precision_limits
        self.precision = precision
        self.recall = recall
        self.beta = self.__beta__()

    def __repr__(self):
        return f'F-beta(precision: {self.precision}, recall: {self.recall}, beta: {self.beta})'

    @property
    def precision(self):
        """Return the precision.

        Returns:
            float: Precision."""

        return self.__precision

    @precision.setter
    def precision(self, value):
        """Update the precision."""

        self.__precision = value

    @property
    def recall(self):
        """Return the recall.

        Returns:
            float: Recall."""

        return self.__recall

    @recall.setter
    def recall(self, value):
        """Update the recall."""

        self.__recall = value

    def __beta__(self):
        return 10 ** scale_precision_input(self.precision_limits)

    def update_metrics(self, true_labels, labels_limits):
        """Update the metrics given the control limits.

        Args:
            true_labels (numpy array): True labels.
            labels_limits (numpy array): Predicted labels per control limits."""

        _, fp, fn, tp = confusion_matrix(true_labels, labels_limits).ravel()
        self.precision = tp / (fp + tp)
        self.recall = tp / (fn + tp)

    def f_beta_score(self):
        """Return the F-beta score.

        Returns:
            float: F-beta score."""

        term1 = self.precision * self.recall
        term2 = (self.beta.__pow__(2) * self.precision) + self.recall
        return (1 + self.beta.__pow__(2)) * (term1 / term2)
