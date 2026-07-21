"""Gray code utilities, used to build Gray-labeled constellations."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def binary_to_gray(n: ArrayLike) -> NDArray[np.int64] | int:
    """Convert standard binary integer(s) to their Gray code equivalent."""
    n_arr = np.asarray(n, dtype=np.int64)
    result = n_arr ^ (n_arr >> 1)
    return int(result) if result.ndim == 0 else result


def gray_to_binary(g: ArrayLike) -> NDArray[np.int64] | int:
    """Convert Gray code integer(s) back to standard binary."""
    g_arr = np.asarray(g, dtype=np.int64)
    mask = g_arr.copy()
    result = g_arr.copy()
    while np.any(mask):
        mask = mask >> 1
        result = result ^ mask
    return int(result) if result.ndim == 0 else result


def gray_code_sequence(bits_per_symbol: int) -> NDArray[np.int64]:
    """Return the Gray code sequence of length `2**bits_per_symbol`.

    `gray_code_sequence(k)[i]` is the Gray code for binary value `i`, so
    consecutive entries always differ in exactly one bit.
    """
    n = np.arange(1 << bits_per_symbol, dtype=np.int64)
    return np.asarray(binary_to_gray(n))


__all__ = ['binary_to_gray', 'gray_code_sequence', 'gray_to_binary']
