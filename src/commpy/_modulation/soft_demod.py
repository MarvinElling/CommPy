"""Soft-decision (LLR) demodulation, shared by any `Modulator` subclass."""

import numpy as np
from numpy.typing import NDArray


def compute_llr(
    symbols: NDArray[np.complex128],
    constellation: NDArray[np.complex128],
    bit_labels: NDArray[np.int64],
    noise_var: float,
) -> NDArray[np.float64]:
    """Compute max-log approximate LLRs for each bit of each received symbol.

    For bit position `j`:
        `LLR_j = (1/noise_var) * (min_{s: b_j=1} |y-s|^2 - min_{s: b_j=0} |y-s|^2)`

    A positive LLR favors bit 0 (`L = log(P(b=0|y) / P(b=1|y))`), matching the
    common convention used e.g. as Viterbi soft-decision input.

    Args:
        symbols: Received complex symbols, shape `(N,)`.
        constellation: Constellation points, shape `(M,)`.
        bit_labels: Bit labels per constellation point, shape `(M, bits_per_symbol)`.
        noise_var: Noise variance per complex dimension.

    Returns:
        LLRs, shape `(N * bits_per_symbol,)`, in the same bit ordering as
        `Modulator.demodulate`'s output.
    """
    bits_per_symbol = bit_labels.shape[1]
    dists = np.abs(symbols[:, None] - constellation[None, :])**2  # (N, M)

    llrs = np.empty((symbols.shape[0], bits_per_symbol), dtype=np.float64)
    for j in range(bits_per_symbol):
        is_bit_zero = bit_labels[:, j] == 0
        min_dist_zero = np.min(dists[:, is_bit_zero], axis=1)
        min_dist_one = np.min(dists[:, ~is_bit_zero], axis=1)
        llrs[:, j] = (min_dist_one - min_dist_zero) / noise_var

    return llrs.reshape(-1)


__all__ = ['compute_llr']
