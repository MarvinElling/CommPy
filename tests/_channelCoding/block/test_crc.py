"""Tests for commpy.CRC / CRCConfig, cross-validated against Python's stdlib CRCs."""

import binascii
import zlib

import pytest

from commpy import CRC, CRCConfig
from commpy._channelCoding.block.crc import compute_bitwise

CHECK_MESSAGE = b'123456789'  # the standard CRC catalogue check string


def test_crc32_matches_zlib():
    crc = CRC.crc32()
    assert crc.compute(CHECK_MESSAGE) == zlib.crc32(CHECK_MESSAGE)


def test_crc32_matches_zlib_for_random_messages(rng):
    crc = CRC.crc32()
    for _ in range(20):
        n = int(rng.integers(0, 200))
        data = bytes(rng.integers(0, 256, n, dtype='uint8'))
        assert crc.compute(data) == zlib.crc32(data)


def test_crc16_xmodem_matches_binascii():
    crc = CRC.crc16_xmodem()
    assert crc.compute(CHECK_MESSAGE) == binascii.crc_hqx(CHECK_MESSAGE, 0)


def test_crc16_xmodem_matches_binascii_for_random_messages(rng):
    crc = CRC.crc16_xmodem()
    for _ in range(20):
        n = int(rng.integers(0, 200))
        data = bytes(rng.integers(0, 256, n, dtype='uint8'))
        assert crc.compute(data) == binascii.crc_hqx(data, 0)


def test_table_driven_matches_bitwise_reference_crc8(rng):
    config = CRCConfig(width=8, poly=0x07, init=0x00, xorout=0x00, refin=False, refout=False)
    crc = CRC(config)
    for _ in range(50):
        n = int(rng.integers(0, 100))
        data = bytes(rng.integers(0, 256, n, dtype='uint8'))
        assert crc.compute(data) == compute_bitwise(data, config)


def test_table_driven_matches_bitwise_reference_all_presets(rng):
    for crc in (CRC.crc8(), CRC.crc16_xmodem(), CRC.crc32()):
        for _ in range(20):
            n = int(rng.integers(0, 100))
            data = bytes(rng.integers(0, 256, n, dtype='uint8'))
            assert crc.compute(data) == compute_bitwise(data, crc.config)


def test_empty_message():
    crc = CRC.crc32()
    # CRC-32 of empty input is 0 (init and xorout cancel: 0xFFFFFFFF ^ 0xFFFFFFFF).
    assert crc.compute(b'') == 0


def test_different_messages_almost_never_collide():
    crc = CRC.crc16_xmodem()
    seen = {crc.compute(bytes([i, j])) for i in range(50) for j in range(50)}
    assert len(seen) > 2400  # out of 2500 possible (i, j) pairs, essentially all should differ


def test_rejects_width_below_8():
    with pytest.raises(ValueError, match='width'):
        CRC(CRCConfig(width=4, poly=0x3, init=0, xorout=0, refin=False, refout=False))
