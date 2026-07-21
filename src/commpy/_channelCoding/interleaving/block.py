"""Block interleaver: write rows, read columns."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BlockInterleaver:
    """Matrix (row-write, column-read) interleaver over fixed-size blocks of `rows * cols`."""

    def __init__(self, rows: int, cols: int) -> None:
        """Construct a `rows x cols` block (de)interleaver."""
        self.rows = rows
        self.cols = cols
        self.block_size = rows * cols

    def interleave(self, data: ArrayLike) -> NDArray[np.generic]:
        """Write `data` into the matrix row-wise, read it out column-wise.

        Raises:
            ValueError: If `len(data) != rows * cols`.
        """
        arr = np.asarray(data)
        if arr.size != self.block_size:
            err = f'data must have length {self.block_size}, got {arr.size}.'
            raise ValueError(err)
        return arr.reshape(self.rows, self.cols).T.reshape(-1)

    def deinterleave(self, data: ArrayLike) -> NDArray[np.generic]:
        """Invert `interleave`.

        Raises:
            ValueError: If `len(data) != rows * cols`.
        """
        arr = np.asarray(data)
        if arr.size != self.block_size:
            err = f'data must have length {self.block_size}, got {arr.size}.'
            raise ValueError(err)
        return arr.reshape(self.cols, self.rows).T.reshape(-1)


__all__ = ['BlockInterleaver']
