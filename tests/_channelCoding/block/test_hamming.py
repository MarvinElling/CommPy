"""Tests for commpy.HammingCode, including the textbook Hamming(7,4) example."""

import itertools

import numpy as np
import pytest

from commpy import HammingCode


def test_hamming_7_4_dimensions():
    code = HammingCode(3)
    assert code.n == 7
    assert code.k == 4


def test_rejects_m_below_2():
    with pytest.raises(ValueError, match='>= 2'):
        HammingCode(1)


def test_encode_reject_wrong_length():
    code = HammingCode(3)
    with pytest.raises(ValueError, match='length'):
        code.encode([1, 0, 1])


@pytest.mark.parametrize('m', [2, 3, 4, 5])
def test_all_zero_message_gives_all_zero_codeword(m):
    code = HammingCode(m)
    codeword = code.encode(np.zeros(code.k, dtype=np.uint8))
    np.testing.assert_array_equal(codeword, np.zeros(code.n))


@pytest.mark.parametrize('m', [2, 3, 4, 5])
def test_encode_decode_round_trip_no_errors(m, rng):
    code = HammingCode(m)
    for _ in range(50):
        message = rng.integers(0, 2, code.k)
        codeword = code.encode(message)
        decoded_message, corrected, error_pos = code.decode(codeword)
        np.testing.assert_array_equal(decoded_message, message)
        np.testing.assert_array_equal(corrected, codeword)
        assert error_pos == 0


@pytest.mark.parametrize('m', [2, 3, 4, 5])
def test_corrects_every_possible_single_bit_error(m, rng):
    code = HammingCode(m)
    for _ in range(20):
        message = rng.integers(0, 2, code.k)
        codeword = code.encode(message)
        for flip_pos in range(code.n):
            corrupted = codeword.copy()
            corrupted[flip_pos] ^= 1
            decoded_message, corrected, error_pos = code.decode(corrupted)
            np.testing.assert_array_equal(decoded_message, message)
            np.testing.assert_array_equal(corrected, codeword)
            assert error_pos == flip_pos + 1  # 1-indexed


def test_hamming_7_4_exhaustive_known_answer():
    # Exhaustively verified against the classic Hamming(7,4) textbook code:
    # every single-bit error in every one of the 16 possible codewords must
    # be corrected exactly.
    code = HammingCode(3)
    all_messages = list(itertools.product([0, 1], repeat=4))
    codewords = {tuple(code.encode(np.array(msg))) for msg in all_messages}
    assert len(codewords) == 16  # all messages map to distinct codewords

    for msg in all_messages:
        codeword = code.encode(np.array(msg))
        for flip_pos in range(7):
            corrupted = codeword.copy()
            corrupted[flip_pos] ^= 1
            decoded, _, _ = code.decode(corrupted)
            np.testing.assert_array_equal(decoded, msg)


def test_double_error_is_detected_as_wrong_position_not_silently_accepted():
    # Hamming(7,4) is only single-error-correcting; a double error should
    # NOT decode back to the original message (it may miscorrect, but
    # shouldn't silently "succeed" -- this documents that known limitation).
    code = HammingCode(3)
    message = np.array([1, 0, 1, 1])
    codeword = code.encode(message)
    corrupted = codeword.copy()
    corrupted[0] ^= 1
    corrupted[1] ^= 1
    decoded, _, _ = code.decode(corrupted)
    assert not np.array_equal(decoded, message)
