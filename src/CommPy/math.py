from math import log2


class Formulas:
    """
    This module contains various formulas related to communication theory.
    """

    @staticmethod
    def shannon_entropy(probabilities):
        """
        Calculate the Shannon entropy of a probability distribution.

        :param probabilities: A list of probabilities.
        :return: The Shannon entropy.
        """

        return -sum(p * log2(p) for p in probabilities if p > 0)