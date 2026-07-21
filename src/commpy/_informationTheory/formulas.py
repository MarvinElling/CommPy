"""General mathematical formulas for communication theory."""

from math import log2

import numpy as np
from numpy.typing import ArrayLike, NDArray


def shannon_entropy(probabilities: list[float]) -> float:
    """Calculate the Shannon entropy of a probability distribution.

    :param probabilities: A list of probabilities.
    :return: The Shannon entropy.
    """
    return -sum(p * log2(p) for p in probabilities if p > 0)


def binary_entropy(p: float) -> float:
    """Shannon entropy (bits) of a Bernoulli(`p`) source: `-p*log2(p) - (1-p)*log2(1-p)`."""
    if p <= 0 or p >= 1:
        return 0.0
    return shannon_entropy([p, 1 - p])


def mutual_information(joint_prob: ArrayLike) -> float:
    """Mutual information `I(X;Y)` of a joint probability distribution.

    Args:
        joint_prob: 2D array `P[x, y] = p(X=x, Y=y)`.

    Returns:
        Mutual information in bits.
    """
    p_xy = np.asarray(joint_prob, dtype=np.float64)
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    denom = p_x * p_y
    # marginals are sums of non-negative terms, so p_x*p_y > 0 wherever p_xy > 0
    included = p_xy > 0
    ratio = np.where(included, p_xy / np.where(included, denom, 1.0), 1.0)
    terms = np.where(included, p_xy * np.log2(ratio), 0.0)
    return float(np.sum(terms))


def channel_capacity_bsc(p: float) -> float:
    """Capacity (bits/use) of a Binary Symmetric Channel with crossover probability `p`."""
    return 1.0 - binary_entropy(p)


def channel_capacity_awgn(snr_linear: float) -> float:
    """Shannon capacity (bits/complex channel use) of an AWGN channel at linear SNR `snr_linear`."""
    return float(np.log2(1.0 + snr_linear))


def channel_capacity_dmc(
    transition_matrix: ArrayLike, max_iter: int = 1000, tol: float = 1e-12,
) -> tuple[float, NDArray[np.float64]]:
    """Capacity of a discrete memoryless channel via the Blahut-Arimoto algorithm.

    Iteratively finds the input distribution maximizing mutual information;
    cross-validated in the test suite against the closed-form BSC capacity
    (`channel_capacity_bsc`) and against a brute-force grid search on an
    asymmetric (Z-)channel.

    Args:
        transition_matrix: `Q[x, y] = p(Y=y | X=x)`, each row summing to 1.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance on the input distribution (max abs change).

    Returns:
        `(capacity, optimal_input_distribution)`, capacity in bits.
    """
    q = np.asarray(transition_matrix, dtype=np.float64)
    n_x = q.shape[0]
    p = np.full(n_x, 1.0 / n_x)

    for _ in range(max_iter):
        marg = p @ q
        included = q > 0
        ratio = np.where(included, q / np.where(included, marg[None, :], 1.0), 1.0)
        logr = np.where(included, np.log2(ratio), 0.0)
        c = np.exp2(np.sum(q * logr, axis=1))
        p_new = p * c / np.sum(p * c)
        converged = np.max(np.abs(p_new - p)) < tol
        p = p_new
        if converged:
            break

    marg = p @ q
    included = q > 0
    ratio = np.where(included, q / np.where(included, marg[None, :], 1.0), 1.0)
    logr = np.where(included, np.log2(ratio), 0.0)
    capacity = float(np.sum(p[:, None] * q * logr))
    return capacity, p


__all__ = [
    'binary_entropy',
    'channel_capacity_awgn',
    'channel_capacity_bsc',
    'channel_capacity_dmc',
    'mutual_information',
    'shannon_entropy',
]
