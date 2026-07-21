"""Tests for commpy.CyclicCode (generic cyclic code from a generator polynomial)."""

import numpy as np
import pytest

from commpy import CyclicCode, PrimeField


def test_rejects_generator_not_dividing_x_n_minus_1():
    field = PrimeField(2)
    # x^2 + 1 does not divide x^7 - 1 over GF(2) (7 is prime, the only degree-2
    # divisor factors of x^7+1 correspond to primitive cube-root-adjacent
    # structure that x^2+1 doesn't match).
    with pytest.raises(ValueError, match='divide'):
        CyclicCode(7, [1, 0, 1], field)


def test_hamming_7_4_as_a_cyclic_code(rng):
    # The (7,4) Hamming code is also realizable as a cyclic code with
    # generator g(x) = 1 + x + x^3 (degree 3 = n - k).
    field = PrimeField(2)
    code = CyclicCode(7, [1, 1, 0, 1], field)
    assert code.n == 7
    assert code.k == 4

    for _ in range(30):
        message = rng.integers(0, 2, 4)
        codeword = code.encode(message)
        assert code.is_codeword(codeword)
        np.testing.assert_array_equal(code.extract_message(codeword), message)


def test_corrupted_word_is_not_a_codeword():
    field = PrimeField(2)
    code = CyclicCode(7, [1, 1, 0, 1], field)
    codeword = code.encode(np.array([1, 0, 1, 1]))
    corrupted = codeword.copy()
    corrupted[2] ^= 1
    assert not code.is_codeword(corrupted)


def test_encode_rejects_wrong_length():
    field = PrimeField(2)
    code = CyclicCode(7, [1, 1, 0, 1], field)
    with pytest.raises(ValueError, match='length'):
        code.encode([1, 0, 1])


def test_syndrome_rejects_wrong_length():
    field = PrimeField(2)
    code = CyclicCode(7, [1, 1, 0, 1], field)
    with pytest.raises(ValueError, match='length'):
        code.syndrome([1, 0, 1])
