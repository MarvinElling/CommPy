"""The user-facing `TurboCode`: parallel-concatenated RSC codes with iterative decoding.

A rate-1/3 turbo code transmits the systematic bits plus the parity of two RSC
encoders -- the second fed an interleaved copy of the message. Decoding iterates
two BCJR decoders that exchange *extrinsic* information through the interleaver
until they agree, the scheme that first approached the Shannon limit.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from commpy._channelCoding.turbo.bcjr import bcjr_decode
from commpy._channelCoding.turbo.rsc import RSCTrellis, rsc_encode


class TurboCode:
    """A rate-1/3 parallel-concatenated (turbo) code.

    Two identical RSC constituent codes encode the message and an interleaved
    copy of it; the codeword is `[systematic | parity_1 | parity_2]` (length
    `3k`). `decode` runs iterative log-MAP (BCJR) decoding on soft channel LLRs
    (`Modulator.soft_demodulate` output), passing extrinsic information between
    the two constituent decoders through the interleaver.
    """

    def __init__(  # noqa: PLR0913 -- each parameter configures a distinct, independent code design choice
        self,
        k: int,
        *,
        constraint_length: int = 3,
        feedback: int = 0o7,
        feedforward: int = 0o5,
        interleaver: ArrayLike | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Construct a turbo code for length-`k` messages.

        Args:
            k: Information block length.
            constraint_length: RSC constraint length.
            feedback: RSC feedback polynomial (see `RSCTrellis`).
            feedforward: RSC feedforward (parity) polynomial.
            interleaver: Optional explicit interleaver permutation of `range(k)`;
                if omitted a pseudo-random permutation is drawn from `rng`.
            rng: Optional `np.random.Generator` seeding the default interleaver
                (defaults to a fixed seed so a code is reproducible).

        Raises:
            ValueError: If `k < 1`, or `interleaver` is not a permutation of
                `range(k)`.
        """
        if k < 1:
            msg = f'k must be >= 1, got {k}.'
            raise ValueError(msg)
        self.k = int(k)
        self.n = 3 * self.k
        self.trellis = RSCTrellis(constraint_length, feedback, feedforward)

        if interleaver is not None:
            perm = np.asarray(interleaver, dtype=np.int64)
            if perm.shape != (self.k,) or sorted(perm.tolist()) != list(range(self.k)):
                msg = f'interleaver must be a permutation of range({self.k}).'
                raise ValueError(msg)
        else:
            generator = rng if rng is not None else np.random.default_rng(0)
            perm = generator.permutation(self.k)
        self.interleaver = perm
        self.deinterleaver = np.argsort(perm)

    @property
    def rate(self) -> float:
        """Code rate `k / n` (= 1/3)."""
        return self.k / self.n

    def encode(self, message: ArrayLike) -> NDArray[np.uint8]:
        """Encode a length-`k` message into a length-`3k` codeword.

        Args:
            message: Information bits, length `k`.

        Returns:
            The codeword `[systematic | parity_1 | parity_2]` (`uint8`).

        Raises:
            ValueError: If `len(message) != k`.
        """
        msg = np.asarray(message, dtype=np.uint8) % 2
        if msg.size != self.k:
            err = f'message must have length {self.k}, got {msg.size}.'
            raise ValueError(err)
        parity1 = rsc_encode(self.trellis, msg)
        parity2 = rsc_encode(self.trellis, msg[self.interleaver])
        return np.concatenate([msg, parity1, parity2])

    def decode(
        self, llr: ArrayLike, *, iterations: int = 8,
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8], int]:
        """Iteratively decode channel LLRs into the transmitted message.

        Args:
            llr: Channel LLRs, length `3k`, `L = log(P(0)/P(1))` (positive favors
                bit 0), ordered `[systematic | parity_1 | parity_2]`.
            iterations: Number of turbo (BCJR-exchange) iterations.

        Returns:
            `(message, codeword, iterations)`: the decoded `k`-bit message, the
            re-encoded `3k`-bit codeword, and the number of iterations run.

        Raises:
            ValueError: If `len(llr) != 3k` or `iterations < 1`.
        """
        channel = np.asarray(llr, dtype=np.float64)
        if channel.size != self.n:
            msg = f'llr must have length {self.n}, got {channel.size}.'
            raise ValueError(msg)
        if iterations < 1:
            msg = f'iterations must be >= 1, got {iterations}.'
            raise ValueError(msg)

        sys_llr = channel[:self.k]
        parity1_llr = channel[self.k:2 * self.k]
        parity2_llr = channel[2 * self.k:]

        apriori = np.zeros(self.k, dtype=np.float64)
        extrinsic1 = np.zeros(self.k, dtype=np.float64)
        for _ in range(iterations):
            extrinsic1 = bcjr_decode(self.trellis, apriori, sys_llr, parity1_llr)
            apriori2 = extrinsic1[self.interleaver]
            extrinsic2 = bcjr_decode(
                self.trellis, apriori2, sys_llr[self.interleaver], parity2_llr,
            )
            apriori = extrinsic2[self.deinterleaver]

        total_llr = sys_llr + extrinsic1 + apriori
        message = (total_llr < 0).astype(np.uint8)
        return message, self.encode(message), iterations


__all__ = ['TurboCode']
