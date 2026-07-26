"""Ready-made quasi-cyclic LDPC codes built by protograph lifting.

Modern standardized LDPC codes (5G-NR, 802.11n/ac, DVB-S2) are all *quasi-
cyclic*: a small integer "base graph" is lifted by a factor `z` into the full
parity-check matrix via circulant-shifted identity blocks (see `qc_lift`). This
module ships one worked rate-1/2 base graph and a convenience constructor, both
as a demonstration of that mechanism and as a ready code for coded-BER studies.

The full 5G-NR base graphs (BG1/BG2) follow the identical construction with
larger, standardized shift tables; adding them is a drop-in extension of the
same `qc_lift` path.
"""

import numpy as np
from numpy.typing import NDArray

from commpy._channelCoding.ldpc.code import LDPCCode

# A (column-weight-3, row-weight-6) protograph: 3 x 6 base graph -> rate ~1/2.
# Every entry is a non-negative circulant shift, so each lifted column has
# weight 3 and each lifted row weight 6.
RATE_ONE_HALF_BASE: NDArray[np.int64] = np.array(
    [
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 0],
        [2, 4, 0, 5, 1, 3],
    ],
    dtype=np.int64,
)


def rate_one_half_ldpc(z: int = 8) -> LDPCCode:
    """Construct a rate-~1/2 quasi-cyclic LDPC code of length `6 * z`.

    Lifts `RATE_ONE_HALF_BASE` by `z` (see `qc_lift`), giving a code of length
    `n = 6 * z` with `3 * z` parity checks.

    Args:
        z: Lifting factor; `>= 1`. The default `z = 8` gives a length-48 code.

    Returns:
        The constructed `LDPCCode`.
    """
    return LDPCCode.from_base_graph(RATE_ONE_HALF_BASE, z)


__all__ = ['RATE_ONE_HALF_BASE', 'rate_one_half_ldpc']
