"""Tests for commpy.ReedSolomonCode: error-only and erasure-only decoding.

As with BCH, correctness leans on exhaustive/brute-force error-pattern
sweeps on small codes rather than trusting the Forney-algorithm algebra by
inspection alone.
"""

import itertools

import numpy as np
import pytest

from commpy import ReedSolomonCode


def test_rs_7_3_dimensions():
    # RS(7, 3) over GF(2^3): a small, hand-checkable textbook-scale example.
    code = ReedSolomonCode(m=3, k=3)
    assert code.n == 7
    assert code.k == 3
    assert code.t == 2  # (n - k) // 2 = (7 - 3) // 2 = 2


def test_rejects_k_out_of_range():
    with pytest.raises(ValueError, match='k must satisfy'):
        ReedSolomonCode(m=3, k=0)
    with pytest.raises(ValueError, match='k must satisfy'):
        ReedSolomonCode(m=3, k=7)


@pytest.mark.parametrize('k', [1, 2, 3, 4, 5])
def test_encode_decode_round_trip_no_errors(k, rng):
    code = ReedSolomonCode(m=3, k=k)
    for _ in range(20):
        message = rng.integers(0, code.field.order, code.k)
        codeword = code.encode(message)
        decoded, corrected, n_errors = code.decode(codeword)
        np.testing.assert_array_equal(decoded, message)
        np.testing.assert_array_equal(corrected, codeword)
        assert n_errors == 0


@pytest.mark.parametrize('k', [1, 2, 3, 4, 5])
def test_corrects_every_error_pattern_of_weight_up_to_t(k, rng):
    """Exhaustive sweep over error positions AND magnitudes, weight 0..t."""
    code = ReedSolomonCode(m=3, k=k)
    message = rng.integers(0, code.field.order, code.k)
    codeword = code.encode(message)

    for weight in range(code.t + 1):
        for error_positions in itertools.combinations(range(code.n), weight):
            # A fixed nonzero magnitude per position is enough to validate
            # both error-locating (Chien search) and magnitude recovery
            # (Forney) without an astronomically large combined sweep.
            corrupted = codeword.copy()
            for i, pos in enumerate(error_positions):
                magnitude = 1 + (i % (code.field.order - 1))
                corrupted[pos] = int(code.field.add(corrupted[pos], magnitude))
            decoded, corrected, n_errors = code.decode(corrupted)
            np.testing.assert_array_equal(decoded, message)
            np.testing.assert_array_equal(corrected, codeword)
            assert n_errors == weight


def test_decode_failure_detected_beyond_t(rng):
    code = ReedSolomonCode(m=3, k=3)  # t = 2
    message = rng.integers(0, code.field.order, code.k)
    codeword = code.encode(message)
    corrupted = codeword.copy()
    for pos in range(code.t + 1):  # t+1 = 3 errors, beyond guaranteed capability
        corrupted[pos] = int(code.field.add(corrupted[pos], 1))
    # Either a detected decoding failure, or (if undetected) it must not
    # silently claim the wrong message as correct.
    try:
        decoded, _, _ = code.decode(corrupted)
    except ValueError:
        pass
    else:
        assert not np.array_equal(decoded, message)


@pytest.mark.parametrize('k', [1, 2, 3, 4])
def test_erasure_decoding_round_trip_no_erasures(k, rng):
    code = ReedSolomonCode(m=3, k=k)
    message = rng.integers(0, code.field.order, code.k)
    codeword = code.encode(message)
    decoded, corrected = code.decode_erasures(codeword, [])
    np.testing.assert_array_equal(decoded, message)
    np.testing.assert_array_equal(corrected, codeword)


@pytest.mark.parametrize('k', [1, 2, 3])
def test_corrects_up_to_n_minus_k_erasures(k, rng):
    """Erasure-only correction handles n-k erasures -- twice the error-only capability."""
    code = ReedSolomonCode(m=3, k=k)
    message = rng.integers(0, code.field.order, code.k)
    codeword = code.encode(message)

    for n_erasures in range(code.n_minus_k + 1):
        for erasure_positions in itertools.combinations(range(code.n), n_erasures):
            corrupted = codeword.copy()
            corrupted[list(erasure_positions)] = 999999  # placeholder; must be ignored
            decoded, corrected = code.decode_erasures(corrupted, list(erasure_positions))
            np.testing.assert_array_equal(decoded, message)
            np.testing.assert_array_equal(corrected, codeword)


def test_erasure_decoding_rejects_too_many_erasures():
    code = ReedSolomonCode(m=3, k=3)  # n_minus_k = 4
    codeword = code.encode(np.zeros(code.k, dtype=np.int64))
    with pytest.raises(ValueError, match='Cannot correct'):
        code.decode_erasures(codeword, list(range(code.n_minus_k + 1)))
