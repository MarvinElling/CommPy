"""Generic M-ary Phase Shift Keying (M-PSK) modulator."""

import numpy as np
from numpy.typing import NDArray

from commpy._modulation.base import Modulator
from commpy._modulation.gray import gray_code_sequence


class MPSKModulator(Modulator):
    """Generic M-PSK modulator with Gray-coded bit mapping.

    Constellation points sit on the unit circle at angles `2*pi*i/M`; adjacent
    points differ by exactly one bit.
    """

    def _build_constellation(self) -> tuple[NDArray[np.complex128], NDArray[np.int64]]:
        indices = np.arange(self.M)
        angles = 2 * np.pi * indices / self.M
        constellation = np.exp(1j * angles).astype(np.complex128)  # |s| == 1, already unit energy

        gray_labels = gray_code_sequence(self.bits_per_symbol)
        shifts = np.arange(self.bits_per_symbol - 1, -1, -1)
        bit_labels = ((gray_labels[:, None] >> shifts) & 1).astype(np.int64)
        return constellation, bit_labels


__all__ = ['MPSKModulator']
