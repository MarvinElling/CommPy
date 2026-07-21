"""Generic square M-ary QAM (M-QAM) modulator."""

import numpy as np
from numpy.typing import NDArray

from commpy._modulation.base import Modulator
from commpy._modulation.gray import gray_code_sequence


class MQAMModulator(Modulator):
    """Generic square M-QAM modulator (M in {4, 16, 64, 256, ...}).

    Independently Gray-coded PAM on the I and Q axes, normalized to unit
    average energy. Non-square ("cross") constellations such as 32-QAM are
    out of scope; `M` must be a power of four.
    """

    def __init__(self, m: int) -> None:
        """Build the constellation for a square M-QAM scheme.

        Args:
            m: Constellation size; must be a power of four (4, 16, 64, ...).

        Raises:
            ValueError: If `m` is not a power of four.
        """
        is_power_of_two = m >= 2 and (m & (m - 1)) == 0
        bits_per_symbol = m.bit_length() - 1 if is_power_of_two else -1
        if not is_power_of_two or bits_per_symbol % 2 != 0:
            msg = f'M-QAM requires M to be a power of four (4, 16, 64, 256, ...), got {m}.'
            raise ValueError(msg)
        super().__init__(m)

    def _build_constellation(self) -> tuple[NDArray[np.complex128], NDArray[np.int64]]:
        side = round(np.sqrt(self.M))
        bits_per_axis = self.bits_per_symbol // 2

        pam_levels = np.arange(side, dtype=np.float64) * 2 - (side - 1)  # ..., -3, -1, 1, 3, ...
        gray_axis = gray_code_sequence(bits_per_axis)

        i_grid, q_grid = np.meshgrid(np.arange(side), np.arange(side), indexing='ij')
        i_idx = i_grid.ravel()
        q_idx = q_grid.ravel()

        constellation = (pam_levels[i_idx] + 1j * pam_levels[q_idx]).astype(np.complex128)
        constellation = constellation / np.sqrt(np.mean(np.abs(constellation)**2))

        shifts = np.arange(bits_per_axis - 1, -1, -1)
        i_bits = (gray_axis[i_idx][:, None] >> shifts) & 1
        q_bits = (gray_axis[q_idx][:, None] >> shifts) & 1
        bit_labels = np.concatenate([i_bits, q_bits], axis=1).astype(np.int64)

        return constellation, bit_labels


__all__ = ['MQAMModulator']
