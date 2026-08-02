"""Information-theoretic capacity of a MIMO channel.

For a channel `H` known at the receiver with equal power split across the
`n_tx` transmit antennas, the (deterministic) capacity is

    C = log2 det( I_{n_rx} + (snr / n_tx) H H^H )   [bits per channel use].

Averaging that over the fading distribution gives the ergodic capacity, which
grows roughly like `min(n_tx, n_rx)` -- the spatial-multiplexing gain that makes
MIMO attractive.
"""

import numpy as np
from numpy.typing import ArrayLike

from commpy._mimo.channel import rayleigh_channel_matrix


def mimo_capacity(channel: ArrayLike, snr_db: float) -> float:
    """Deterministic MIMO capacity (bits/use) for a known channel, equal power.

    Args:
        channel: Channel matrix `H`, shape `(n_rx, n_tx)`.
        snr_db: Total SNR in dB (`snr / n_tx` is allocated to each transmit
            antenna).

    Returns:
        The capacity in bits per channel use.
    """
    H = np.asarray(channel, dtype=np.complex128)
    n_rx, n_tx = H.shape
    snr_linear = 10.0 ** (snr_db / 10.0)
    gram = np.eye(n_rx) + (snr_linear / n_tx) * (H @ H.conj().T)
    _, log_abs_det = np.linalg.slogdet(gram)  # gram is Hermitian positive-definite
    return float(log_abs_det / np.log(2.0))


def ergodic_mimo_capacity(
    n_rx: int,
    n_tx: int,
    snr_db: float,
    *,
    n_trials: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """Ergodic MIMO capacity: the mean of `mimo_capacity` over Rayleigh fading.

    Args:
        n_rx: Number of receive antennas.
        n_tx: Number of transmit antennas.
        snr_db: Total SNR in dB.
        n_trials: Number of independent channel realizations to average.
        rng: Optional `np.random.Generator` for reproducibility.

    Returns:
        The average capacity in bits per channel use.
    """
    if rng is None:
        rng = np.random.default_rng()
    capacities = [
        mimo_capacity(rayleigh_channel_matrix(n_rx, n_tx, rng=rng), snr_db)
        for _ in range(n_trials)
    ]
    return float(np.mean(capacities))


__all__ = ['ergodic_mimo_capacity', 'mimo_capacity']
