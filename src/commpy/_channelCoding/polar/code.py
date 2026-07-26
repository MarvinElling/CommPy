"""The user-facing `PolarCode` class: encode, and SC / CRC-aided SCL decode."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from commpy._channelCoding.block.crc import CRC
from commpy._channelCoding.polar.construction import frozen_mask
from commpy._channelCoding.polar.decoder import polar_transform, scl_decode


def _crc_bits(crc: CRC, info_bits: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Compute the CRC of a bit vector (MSB-first byte packing) as its bit expansion."""
    packed = np.packbits(info_bits)
    value = crc.compute(packed.tobytes())
    width = crc.config.width
    return np.array([(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)


class PolarCode:
    """A binary polar code of length `N = 2**n` and dimension `k`.

    Information bits ride the `k` most reliable synthetic bit-channels (chosen by
    `construction`); the rest are frozen to zero. `encode` applies the polar
    (Kronecker) transform; `decode` runs successive-cancellation list decoding
    on soft channel LLRs (`Modulator.soft_demodulate` output).

    Passing a `CRC` enables CRC-aided SCL (CA-SCL): a CRC of the message is
    appended before encoding and, at decode time, used to pick the surviving
    list path whose message checks out -- the decoder used by 5G-NR polar codes.
    The CRC occupies its own `crc.config.width` reliable channels, so the code
    needs `k + crc.config.width <= N`.
    """

    def __init__(  # noqa: PLR0913 -- each parameter configures a distinct, independent design choice
        self,
        block_length: int,
        k: int,
        *,
        crc: CRC | None = None,
        construction: str = 'gaussian',
        design_snr_db: float = 1.0,
        frozen: ArrayLike | None = None,
    ) -> None:
        """Construct a polar code.

        Args:
            block_length: Codeword length `N`; must be a power of two.
            k: Number of information bits.
            crc: Optional `CRC` for CRC-aided SCL decoding (its width bits are
                appended to the message before encoding).
            construction: Frozen-set design, `'gaussian'` or `'bhattacharyya'`.
            design_snr_db: Design-point SNR in dB for the construction.
            frozen: Optional explicit boolean frozen mask (length `N`,
                `True` = frozen); overrides `construction` when given.

        Raises:
            ValueError: If `N` is not a power of two, `k < 1`, or
                `k + crc_width > N`.
        """
        if block_length < 1 or (block_length & (block_length - 1)) != 0:
            msg = f'block_length must be a power of two, got {block_length}.'
            raise ValueError(msg)
        if k < 1:
            msg = f'k must be >= 1, got {k}.'
            raise ValueError(msg)

        self.n = int(block_length)
        self.k = int(k)
        self.crc = crc
        self._crc_width = crc.config.width if crc is not None else 0
        self._n_free = self.k + self._crc_width
        if self._n_free > self.n:
            msg = f'k + crc_width = {self._n_free} exceeds block length {self.n}.'
            raise ValueError(msg)

        if frozen is not None:
            self.frozen = np.asarray(frozen, dtype=bool)
            if self.frozen.shape != (self.n,):
                msg = f'frozen mask must have length {self.n}, got {self.frozen.size}.'
                raise ValueError(msg)
        else:
            self.frozen = frozen_mask(
                self.n, self._n_free, method=construction, design_snr_db=design_snr_db,
            )
        self.info_positions = np.flatnonzero(~self.frozen)

    @property
    def rate(self) -> float:
        """Code rate `k / N`."""
        return self.k / self.n

    def encode(self, message: ArrayLike) -> NDArray[np.uint8]:
        """Encode a length-`k` message into a length-`N` codeword.

        Args:
            message: Information bits, length `k`.

        Returns:
            The length-`N` codeword (`uint8`).

        Raises:
            ValueError: If `len(message) != k`.
        """
        msg = np.asarray(message, dtype=np.uint8) % 2
        if msg.size != self.k:
            err = f'message must have length {self.k}, got {msg.size}.'
            raise ValueError(err)

        payload = msg
        if self.crc is not None:
            payload = np.concatenate([msg, _crc_bits(self.crc, msg)])

        u = np.zeros(self.n, dtype=np.uint8)
        u[self.info_positions] = payload
        return polar_transform(u)

    def _select_path(
        self, paths: list[tuple[float, NDArray[np.uint8], NDArray[np.uint8]]],
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8], int]:
        """Pick the decoded path: CRC-passing best metric if a CRC is set, else best metric."""
        for rank, (_, u_hat, codeword) in enumerate(paths):
            payload = u_hat[self.info_positions]
            message = payload[:self.k]
            if self.crc is None:
                return message, codeword, rank
            if np.array_equal(payload[self.k:], _crc_bits(self.crc, message)):
                return message, codeword, rank
        # No path satisfied the CRC: fall back to the best-metric path.
        _, u_hat, codeword = paths[0]
        return u_hat[self.info_positions][:self.k], codeword, 0

    def decode(
        self, llr: ArrayLike, *, list_size: int = 8,
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8], int]:
        """Decode channel LLRs into the transmitted message via SC / CA-SCL.

        Args:
            llr: Channel LLRs, length `N`, `L = log(P(0)/P(1))` (positive favors
                bit 0), as produced by `Modulator.soft_demodulate`.
            list_size: SCL survivor-list size; `1` is plain successive
                cancellation, larger values (with a `CRC`) give CA-SCL.

        Returns:
            `(message, codeword, path_rank)`: the decoded `k`-bit message, the
            full `N`-bit decoded codeword, and the rank (0 = best metric) of the
            selected survivor path.

        Raises:
            ValueError: If `len(llr) != N` (see `scl_decode`).
        """
        paths = scl_decode(llr, self.frozen, list_size)
        return self._select_path(paths)


__all__ = ['PolarCode']
