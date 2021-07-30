from sklearn.metrics import confusion_matrix
from ..utils import ScalePrecision


class StatisticalAnalysis:
    """Compute measure of success for the control limits

    :param precision_limits: user input precision
    :type precision_limits: float
    :param precision: precision
    :type precision: float
    :param recall: recall
    :type recall: float
    """

    def __init__(self, precision_limits, precision, recall):
        self.precision_limits = precision_limits
        self.precision = precision
        self.recall = recall
        self.beta = self.__beta__()

    def __repr__(self):
        return f'F-beta(precision: {self.precision}, recall: {self.recall}, beta: {self.beta})'

    def __beta__(self):
        """Return beta for calculating the f-beta score

        :return: beta
        :rtype: float
        """

        scaling = ScalePrecision()
        return 10 ** scaling(self.precision_limits)

    @property
    def precision(self):
        """Return the precision

        :return: precision of the control limits
        :rtype: float
        """

        return self.__precision

    @precision.setter
    def precision(self, value):
        """Update the precision

        :param value: new precision value
        :type value: float
        """

        self.__precision = value

    @property
    def recall(self):
        """Return the recall

        :return: recall of the control limits
        :rtype: float
        """

        return self.__recall

    @recall.setter
    def recall(self, value):
        """Update the recall

        :param value: new recall value
        :type value: float
        """

        self.__recall = value

    def update_measures(self, labels_true, labels_limits):
        """Update the success measures given the control limits

        :param labels_true: true labels
        :type labels_true: numpy array
        :param labels_limits: true labels
        :type labels_limits: numpy array
        """

        _, fp, fn, tp = confusion_matrix(labels_true, labels_limits).ravel()
        self.precision = tp / (fp + tp)
        self.recall = tp / (fn + tp)

    def f_beta_score(self):
        """Return the f-beta score

        :return: f-beta score
        :rtype: float
        """

        term1 = self.precision * self.recall
        term2 = ((self.beta ** 2) * self.precision) + self.recall
        return (1 + (self.beta ** 2)) * (term1 / term2)
