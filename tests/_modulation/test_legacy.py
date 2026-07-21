"""Tests for the legacy per-scheme modulator classes in commpy._modulation.legacy."""

import numpy as np

from commpy import (
    ASK_2_Modulator,
    ASK_4_Modulator,
    BPSK_Modulator,
    OOK_Modulator,
    PSK_8_Modulator,
    QPSK_Modulator,
)


def test_qpsk_modulate_matches_for_list_tuple_and_ndarray_input():
    # Regression test for the original bug: `bit_pair == [0, 0]` only matched
    # when bitstream slicing yielded a plain list, so ndarray/tuple input
    # silently produced an empty (or wrong) symbol array.
    bits_list = [1, 0, 1, 1, 0, 0, 0, 1]
    bits_tuple = tuple(bits_list)
    bits_array = np.array(bits_list)

    from_list = QPSK_Modulator.modulate(bits_list)
    from_tuple = QPSK_Modulator.modulate(bits_tuple)
    from_array = QPSK_Modulator.modulate(bits_array)

    assert len(from_list) == len(bits_list) // 2  # sanity: the original bug produced 0 symbols
    np.testing.assert_allclose(from_list, from_tuple)
    np.testing.assert_allclose(from_list, from_array)


def test_qpsk_round_trip():
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
    symbols = QPSK_Modulator.modulate(bits)
    recovered = QPSK_Modulator.demodulate(symbols)
    np.testing.assert_array_equal(recovered, bits)


def test_ook_round_trip():
    bits = [0, 1, 1, 0, 1]
    symbols = OOK_Modulator.modulate(bits)
    recovered = [1 if s.real > 0.5 else 0 for s in OOK_Modulator.demodulate(symbols)]
    assert recovered == bits


def test_bpsk_round_trip():
    bits = [0, 1, 1, 0, 1]
    symbols = BPSK_Modulator.modulate(bits)
    recovered = BPSK_Modulator.demodulate(symbols).tolist()
    assert recovered == bits


def test_ask2_round_trip():
    bits = [0, 1, 1, 0, 1]
    symbols = ASK_2_Modulator.modulate(bits)
    recovered = ASK_2_Modulator.demodulate(symbols).tolist()
    assert recovered == bits


def test_ask4_round_trip():
    bits = [0, 1, 1, 0, 1, 1, 0, 0]
    symbols = ASK_4_Modulator.modulate(bits)
    recovered = ''.join(ASK_4_Modulator.demodulate(symbols))
    assert recovered == ''.join(str(b) for b in bits)


def test_psk8_round_trip():
    bits = [1, 0, 1, 0, 1, 1, 0, 0, 1]
    symbols = PSK_8_Modulator.modulate(bits)
    recovered = ''.join(PSK_8_Modulator.demodulate(symbols))
    assert recovered == ''.join(str(b) for b in bits)
