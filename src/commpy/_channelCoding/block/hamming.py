"""Binary Hamming(2**m - 1, 2**m - 1 - m) single-error-correcting block code."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


class HammingCode:
    """Binary Hamming(2**m - 1, 2**m - 1 - m) code: corrects any single-bit error.

    Uses the classic systematic construction where parity bits sit at the
    power-of-two positions (1, 2, 4, ...) of the (1-indexed) codeword and data
    bits fill the rest. This makes the parity-check matrix `H`'s columns at
    parity positions exactly the standard basis vectors, so the syndrome
    directly encodes the 1-indexed error position (0 if none) with no table
    lookup needed to decode.
    """

    def __init__(self, m: int) -> None:
        """Construct Hamming(2**m - 1, 2**m - 1 - m).

        Args:
            m: Number of parity bits; m >= 2.

        Raises:
            ValueError: If `m < 2`.
        """
        if not isinstance(m, int) or m < 2:
            msg = 'm must be an integer >= 2.'
            raise ValueError(msg)
        self.m = m
        self.n = (1 << m) - 1
        self.k = self.n - m

        positions = np.arange(1, self.n + 1)
        # H[i, j] = bit i of (1-indexed) position j + 1. Columns at power-of-two
        # positions are exactly the standard basis vectors e_i.
        self.H = np.array([(positions >> i) & 1 for i in range(m)], dtype=np.uint8)

        self.parity_positions = [1 << i for i in range(m)]
        self.data_positions = [int(p) for p in positions if int(p) not in self.parity_positions]

        self.G = np.zeros((self.k, self.n), dtype=np.uint8)
        for t, dpos in enumerate(self.data_positions):
            self.G[t, dpos - 1] = 1
            for i, ppos in enumerate(self.parity_positions):
                self.G[t, ppos - 1] = self.H[i, dpos - 1]

    def encode(self, message: ArrayLike) -> NDArray[np.uint8]:
        """Systematically encode a length-`k` message into a length-`n` codeword.

        Raises:
            ValueError: If `len(message) != k`.
        """
        msg = np.asarray(message, dtype=np.uint8)
        if msg.size != self.k:
            err = f'message must have length {self.k}, got {msg.size}.'
            raise ValueError(err)
        return (msg @ self.G) % 2

    def decode(self, received: ArrayLike) -> tuple[NDArray[np.uint8], NDArray[np.uint8], int]:
        """Syndrome-decode a length-`n` received word, correcting any single-bit error.

        Args:
            received: Received codeword, length `n`.

        Returns:
            `(message, corrected_codeword, error_position)`: the recovered
            `k`-bit message, the corrected `n`-bit codeword, and the
            1-indexed bit position that was corrected (0 if no error was
            detected).

        Raises:
            ValueError: If `len(received) != n`.
        """
        r = np.asarray(received, dtype=np.uint8) % 2
        if r.size != self.n:
            err = f'received must have length {self.n}, got {r.size}.'
            raise ValueError(err)
        syndrome = (self.H @ r) % 2
        weights = 1 << np.arange(self.m)
        error_pos = int(syndrome @ weights)

        corrected = r.copy()
        if error_pos != 0:
            corrected[error_pos - 1] ^= 1

        message = corrected[np.array(self.data_positions) - 1]
        return message, corrected, error_pos


__all__ = ['HammingCode']
