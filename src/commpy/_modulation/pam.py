"""Generic M-ary Pulse Amplitude Modulation (M-PAM) modulator."""

import numpy as np
from numpy.typing import NDArray

from commpy._modulation.base import Modulator
from commpy._modulation.gray import gray_code_sequence


class MPAMModulator(Modulator):
    """Generic M-PAM modulator: equally spaced real levels, Gray-coded and unit-energy."""

    def _build_constellation(self) -> tuple[NDArray[np.complex128], NDArray[np.int64]]:
        levels = np.arange(self.M, dtype=np.float64) * 2 - (self.M - 1)  # ..., -3, -1, 1, 3, ...
        levels = levels / np.sqrt(np.mean(levels**2))  # normalize to unit average energy
        constellation = levels.astype(np.complex128)

        gray_labels = gray_code_sequence(self.bits_per_symbol)
        shifts = np.arange(self.bits_per_symbol - 1, -1, -1)
        bit_labels = ((gray_labels[:, None] >> shifts) & 1).astype(np.int64)
        return constellation, bit_labels


__all__ = ['MPAMModulator']
