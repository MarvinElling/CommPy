"""Abstract base for constellation-based digital modulators."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from commpy._modulation.soft_demod import compute_llr


class Modulator(ABC):
    """Base class for constellation-based digital modulators.

    Subclasses need only implement `_build_constellation()`, returning the
    constellation points (unit average energy) and their bit labels;
    `modulate`, `demodulate`, and `soft_demodulate` are then generic,
    vectorized, and correct for any input container (list, tuple, or
    ndarray) by construction.
    """

    M: int
    bits_per_symbol: int
    constellation: NDArray[np.complex128]  # shape (M,)
    bit_labels: NDArray[np.int64]  # shape (M, bits_per_symbol), each row 0/1

    def __init__(self, m: int) -> None:
        """Build the constellation for an M-ary scheme.

        Args:
            m: Constellation size; must be a power of two >= 2.

        Raises:
            ValueError: If `m` is not a power of two >= 2.
        """
        if m < 2 or (m & (m - 1)) != 0:
            msg = f'M must be a power of two >= 2, got {m}.'
            raise ValueError(msg)
        self.M = m
        self.bits_per_symbol = int(np.log2(m))
        self.constellation, self.bit_labels = self._build_constellation()

        weights = 1 << np.arange(self.bits_per_symbol - 1, -1, -1)
        self._weights = weights
        label_values = self.bit_labels @ weights
        self._label_to_index = np.empty(self.M, dtype=np.int64)
        self._label_to_index[label_values] = np.arange(self.M)

    @abstractmethod
    def _build_constellation(self) -> tuple[NDArray[np.complex128], NDArray[np.int64]]:
        """Return `(constellation, bit_labels)` for this scheme.

        `constellation` has shape `(M,)` with unit average energy.
        `bit_labels` has shape `(M, bits_per_symbol)`; `bit_labels[i]` are the
        bits (MSB first) mapped to `constellation[i]`. Every integer in
        `[0, M)` must appear exactly once among the rows' binary values.
        """

    def modulate(self, bits: ArrayLike) -> NDArray[np.complex128]:
        """Map a bitstream onto constellation symbols.

        Args:
            bits: Bit sequence (list, tuple, or ndarray), length a multiple
                of `bits_per_symbol`.

        Returns:
            Complex baseband symbols.

        Raises:
            ValueError: If `len(bits)` is not a multiple of `bits_per_symbol`.
        """
        bits_arr = np.asarray(bits, dtype=np.int64)
        if bits_arr.size % self.bits_per_symbol != 0:
            msg = f'Bitstream length must be a multiple of {self.bits_per_symbol}.'
            raise ValueError(msg)
        groups = bits_arr.reshape(-1, self.bits_per_symbol)
        indices = groups @ self._weights
        return self.constellation[self._label_to_index[indices]]

    def demodulate(self, symbols: ArrayLike) -> NDArray[np.int64]:
        """Hard-decision demodulate symbols to bits via nearest-neighbor decision.

        Args:
            symbols: Received complex baseband symbols.

        Returns:
            Demodulated bit sequence.
        """
        symbols_arr = np.asarray(symbols, dtype=np.complex128)
        dists = np.abs(symbols_arr[:, None] - self.constellation[None, :])**2
        nearest = np.argmin(dists, axis=1)
        return np.asarray(self.bit_labels[nearest].reshape(-1))

    def soft_demodulate(self, symbols: ArrayLike, noise_var: float) -> NDArray[np.float64]:
        """Compute max-log approximate bit LLRs for received symbols.

        Args:
            symbols: Received complex baseband symbols.
            noise_var: Noise variance per complex dimension (as produced by
                e.g. `Channels.awgn`).

        Returns:
            LLRs, one per transmitted bit, in the same bit ordering as
            `demodulate`'s output. A positive LLR favors bit 0.
        """
        symbols_arr = np.asarray(symbols, dtype=np.complex128)
        return compute_llr(symbols_arr, self.constellation, self.bit_labels, noise_var)


__all__ = ['Modulator']
