"""The user-facing `LDPCCode` class: encode, and belief-propagation decode."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from commpy._channelCoding.ldpc.decoder import bp_decode
from commpy._channelCoding.ldpc.matrices import (
    build_tanner,
    make_gallager,
    parity_check_to_generator,
    qc_lift,
)


class LDPCCode:
    """A binary LDPC code defined by a sparse parity-check matrix `H`.

    The code is the null space of `H` over GF(2). At construction `H` is reduced
    to a systematic generator matrix, so `encode` is a single GF(2) mat-vec and
    the `k` information bits sit at fixed codeword positions (`info_positions`).
    `decode` runs iterative belief propagation on soft channel LLRs -- the
    modern soft-decision path that consumes `Modulator.soft_demodulate` output
    directly.

    Construct either directly from a matrix (`LDPCCode(H)`), from Gallager's
    regular ensemble (`LDPCCode.from_gallager`), or from a quasi-cyclic
    protograph (`LDPCCode.from_base_graph`).
    """

    def __init__(self, H: ArrayLike) -> None:
        """Build an LDPC code from a parity-check matrix.

        Args:
            H: Parity-check matrix, shape `(m, n)`, 0/1 valued. Redundant
                (linearly dependent) rows are permitted; the true dimension
                `k = n - rank(H)` is used.

        Raises:
            ValueError: If `H` is not 2-dimensional.
        """
        Hm = np.asarray(H, dtype=np.uint8) % 2
        if Hm.ndim != 2:
            msg = f'H must be a 2-D array, got {Hm.ndim} dimension(s).'
            raise ValueError(msg)
        self.H = Hm
        self.m = int(Hm.shape[0])
        self.n = int(Hm.shape[1])

        G, info_positions, parity_positions, rank = parity_check_to_generator(Hm)
        self.G = G
        self.info_positions = info_positions
        self.parity_positions = parity_positions
        self.rank = rank
        self.k = self.n - rank
        self._tanner = build_tanner(Hm)

    @classmethod
    def from_gallager(
        cls, n: int, w_c: int, w_r: int, *, rng: np.random.Generator | None = None,
    ) -> 'LDPCCode':
        """Construct a regular `(w_c, w_r)` LDPC code (Gallager ensemble).

        See `make_gallager` for the construction and its constraints (notably
        `n` must be a multiple of `w_r`).
        """
        return cls(make_gallager(n, w_c, w_r, rng=rng))

    @classmethod
    def from_base_graph(cls, base_matrix: ArrayLike, z: int) -> 'LDPCCode':
        """Construct a quasi-cyclic LDPC code by lifting a protograph base matrix.

        See `qc_lift` for the base-matrix convention (`-1` for a zero block,
        `s >= 0` for a circulant right-shift by `s`).
        """
        return cls(qc_lift(base_matrix, z))

    @property
    def rate(self) -> float:
        """Code rate `k / n`."""
        return self.k / self.n

    def encode(self, message: ArrayLike) -> NDArray[np.uint8]:
        """Systematically encode a length-`k` message into a length-`n` codeword.

        Args:
            message: Information bits, length `k`.

        Returns:
            The length-`n` codeword (`uint8`); `H @ codeword % 2 == 0`.

        Raises:
            ValueError: If `len(message) != k`.
        """
        msg = np.asarray(message, dtype=np.uint8) % 2
        if msg.size != self.k:
            err = f'message must have length {self.k}, got {msg.size}.'
            raise ValueError(err)
        return np.asarray((msg @ self.G) % 2, dtype=np.uint8)

    def extract_message(self, codeword: ArrayLike) -> NDArray[np.uint8]:
        """Extract the `k` information bits from a length-`n` (systematic) codeword."""
        c = np.asarray(codeword, dtype=np.uint8)
        return np.asarray(c[self.info_positions], dtype=np.uint8)

    def decode(
        self,
        llr: ArrayLike,
        *,
        max_iter: int = 50,
        method: str = 'sum-product',
        normalization: float = 0.75,
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8], int]:
        """Belief-propagation decode channel LLRs into the transmitted message.

        Args:
            llr: Channel LLRs, length `n`, `L = log(P(0)/P(1))` (positive favors
                bit 0), as produced by `Modulator.soft_demodulate`.
            max_iter: Maximum number of belief-propagation iterations.
            method: `'sum-product'` (exact) or `'min-sum'` (approximate).
            normalization: Scaling factor for `'min-sum'` (ignored otherwise).

        Returns:
            `(message, codeword, iterations)`: the decoded `k`-bit message, the
            full `n`-bit decoded codeword, and the number of BP iterations run
            (stops early once a valid codeword is found).

        Raises:
            ValueError: If `len(llr) != n` (see `bp_decode`).
        """
        codeword, _, iterations = bp_decode(
            self._tanner, llr, max_iter=max_iter, method=method, normalization=normalization,
        )
        return self.extract_message(codeword), codeword, iterations


__all__ = ['LDPCCode']
