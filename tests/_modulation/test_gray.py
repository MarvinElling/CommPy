"""Tests for commpy._modulation.gray."""

import numpy as np

from commpy._modulation.gray import binary_to_gray, gray_code_sequence, gray_to_binary


def test_binary_to_gray_known_values():
    # Standard 3-bit Gray code sequence.
    expected = [0, 1, 3, 2, 6, 7, 5, 4]
    for n, g in enumerate(expected):
        assert binary_to_gray(n) == g


def test_gray_to_binary_is_inverse_of_binary_to_gray():
    for n in range(256):
        assert gray_to_binary(binary_to_gray(n)) == n


def test_gray_code_sequence_matches_binary_to_gray():
    seq = gray_code_sequence(4)
    expected = np.array([binary_to_gray(i) for i in range(16)])
    np.testing.assert_array_equal(seq, expected)


def test_consecutive_gray_codes_differ_by_exactly_one_bit():
    seq = gray_code_sequence(5)
    for i in range(len(seq) - 1):
        diff = int(seq[i]) ^ int(seq[i + 1])
        assert diff.bit_count() == 1


def test_vectorized_matches_scalar():
    n = np.arange(16)
    vectorized = binary_to_gray(n)
    scalar = np.array([binary_to_gray(int(x)) for x in n])
    np.testing.assert_array_equal(vectorized, scalar)
