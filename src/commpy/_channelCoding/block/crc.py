"""Cyclic Redundancy Check (CRC), the parametrized ("Rocksoft") model.

Supports any width >= 8 via the standard reflected-input/reflected-output
parametrization used by the CRC RevEng catalogue (`poly`, `init`, `refin`,
`refout`, `xorout`). Two computation paths are provided: `compute_bitwise`
(a direct, transparent implementation of the CRC definition -- one bit at a
time) and the table-driven `CRC` class (one XOR + table lookup per byte).
Both are cross-validated against each other and against `zlib.crc32` /
`binascii.crc_hqx` in the test suite.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CRCConfig:
    """Parameters of a CRC algorithm, in the standard "Rocksoft model" form."""

    width: int
    poly: int
    init: int
    xorout: int
    refin: bool
    refout: bool


def _reflect_bits(value: int, n_bits: int) -> int:
    """Reverse the lowest `n_bits` bits of `value`."""
    result = 0
    for _ in range(n_bits):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def compute_bitwise(data: bytes, config: CRCConfig) -> int:
    """Compute a CRC one bit at a time -- the direct reference implementation.

    Raises:
        ValueError: If `config.width < 8`.
    """
    if config.width < 8:
        msg = 'CRC width must be at least 8.'
        raise ValueError(msg)
    width = config.width
    mask = (1 << width) - 1
    top_bit = 1 << (width - 1)
    crc = config.init & mask
    for raw_byte in data:
        in_byte = _reflect_bits(raw_byte, 8) if config.refin else raw_byte
        crc ^= in_byte << (width - 8)
        for _ in range(8):
            crc = ((crc << 1) ^ config.poly) & mask if crc & top_bit else (crc << 1) & mask
    if config.refout:
        crc = _reflect_bits(crc, width)
    return crc ^ config.xorout


def _build_table(config: CRCConfig) -> tuple[int, ...]:
    """Precompute the 256-entry per-byte CRC table for `config`."""
    width = config.width
    mask = (1 << width) - 1
    top_bit = 1 << (width - 1)
    table = []
    for byte in range(256):
        crc = byte << (width - 8)
        for _ in range(8):
            crc = ((crc << 1) ^ config.poly) & mask if crc & top_bit else (crc << 1) & mask
        table.append(crc & mask)
    return tuple(table)


class CRC:
    """A CRC algorithm instance: holds a config and its precomputed table.

    Use the `crc8`/`crc16_xmodem`/`crc32` classmethods for common presets, or
    construct directly from a `CRCConfig` for a custom polynomial.
    """

    def __init__(self, config: CRCConfig) -> None:
        """Build a CRC instance and precompute its lookup table.

        Raises:
            ValueError: If `config.width < 8`.
        """
        if config.width < 8:
            msg = 'CRC width must be at least 8.'
            raise ValueError(msg)
        self.config = config
        self._table = _build_table(config)

    def compute(self, data: bytes) -> int:
        """Compute the CRC of `data` using the precomputed table."""
        width = self.config.width
        mask = (1 << width) - 1
        crc = self.config.init & mask
        for raw_byte in data:
            in_byte = _reflect_bits(raw_byte, 8) if self.config.refin else raw_byte
            idx = ((crc >> (width - 8)) ^ in_byte) & 0xFF
            crc = ((crc << 8) ^ self._table[idx]) & mask
        if self.config.refout:
            crc = _reflect_bits(crc, width)
        return crc ^ self.config.xorout

    @classmethod
    def crc8(cls) -> 'CRC':
        """CRC-8 (poly 0x07, init 0x00, no reflection): the basic 8-bit CRC."""
        return cls(CRCConfig(width=8, poly=0x07, init=0x00, xorout=0x00, refin=False, refout=False))

    @classmethod
    def crc16_xmodem(cls) -> 'CRC':
        """CRC-16/XMODEM (poly 0x1021, init 0x0000, no reflection)."""
        return cls(
            CRCConfig(width=16, poly=0x1021, init=0x0000, xorout=0x0000, refin=False, refout=False),
        )

    @classmethod
    def crc32(cls) -> 'CRC':
        """CRC-32 (poly 0x04C11DB7, reflected): used by Ethernet, zip, gzip, ..."""
        return cls(
            CRCConfig(
                width=32,
                poly=0x04C11DB7,
                init=0xFFFFFFFF,
                xorout=0xFFFFFFFF,
                refin=True,
                refout=True,
            ),
        )


__all__ = ['CRC', 'CRCConfig', 'compute_bitwise']
