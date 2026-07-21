"""Tests for commpy.BCHCode.

Correctness is established mainly via exhaustive/brute-force error-pattern
sweeps on small codes -- the gold-standard cross-validation for algebraic
decoders, per the project's testing strategy -- rather than by trusting the
Berlekamp-Massey/Chien-search algebra by inspection alone.
"""

import itertools

import numpy as np
import pytest

from commpy import BCHCode


def test_bch_15_11_t1_dimensions():
    # BCH(15, 11, t=1) over GF(2^4) is (up to relabeling) the Hamming(15,11) code.
    code = BCHCode(m=4, t=1)
    assert code.n == 15
    assert code.k == 11


def test_bch_15_7_t2_dimensions():
    code = BCHCode(m=4, t=2)
    assert code.n == 15
    assert code.k == 7


def test_bch_15_5_t3_dimensions():
    code = BCHCode(m=4, t=3)
    assert code.n == 15
    assert code.k == 5


def test_rejects_t_too_large_for_m():
    with pytest.raises(ValueError, match='too large'):
        BCHCode(m=3, t=10)


@pytest.mark.parametrize(('m', 't'), [(4, 1), (4, 2), (4, 3)])
def test_encode_decode_round_trip_no_errors(m, t, rng):
    code = BCHCode(m, t)
    for _ in range(20):
        message = rng.integers(0, 2, code.k)
        codeword = code.encode(message)
        decoded, corrected, n_errors = code.decode(codeword)
        np.testing.assert_array_equal(decoded, message)
        np.testing.assert_array_equal(corrected, codeword)
        assert n_errors == 0


@pytest.mark.parametrize(('m', 't'), [(4, 1), (4, 2), (4, 3)])
def test_corrects_every_error_pattern_of_weight_exactly_t(m, t, rng):
    """Exhaustive sweep: every combination of exactly `t` bit flips must be corrected."""
    code = BCHCode(m, t)
    for _ in range(5):
        message = rng.integers(0, 2, code.k)
        codeword = code.encode(message)
        for error_positions in itertools.combinations(range(code.n), t):
            corrupted = codeword.copy()
            for pos in error_positions:
                corrupted[pos] ^= 1
            decoded, corrected, n_errors = code.decode(corrupted)
            np.testing.assert_array_equal(decoded, message)
            np.testing.assert_array_equal(corrected, codeword)
            assert n_errors == t


def test_corrects_every_error_weight_from_0_to_t_bch_15_7_t2(rng):
    code = BCHCode(m=4, t=2)
    message = rng.integers(0, 2, code.k)
    codeword = code.encode(message)
    for weight in range(code.t + 1):
        for error_positions in itertools.combinations(range(code.n), weight):
            corrupted = codeword.copy()
            for pos in error_positions:
                corrupted[pos] ^= 1
            decoded, _, n_errors = code.decode(corrupted)
            np.testing.assert_array_equal(decoded, message)
            assert n_errors == weight


def test_decode_failure_raises_on_excessive_errors():
    # BCH(15, 7, t=2): 3 errors exceeds the guaranteed correction capability
    # and should either raise (detected decoding failure) or, if it happens
    # to produce a result, must not silently claim success with the wrong
    # error count -- here we specifically construct a case that triggers the
    # detectable-failure path (Chien-search root count mismatch).
    code = BCHCode(m=4, t=2)
    codeword = code.encode(np.zeros(code.k, dtype=np.int64))
    corrupted = codeword.copy()
    corrupted[[0, 1, 2, 3]] ^= 1  # 4 errors, well beyond t=2
    with pytest.raises(ValueError, match='Decoding failure'):
        code.decode(corrupted)
