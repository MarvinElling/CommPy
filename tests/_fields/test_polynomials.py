"""Tests for commpy._fields.polynomials (polynomial arithmetic over a FiniteField)."""

import numpy as np
import pytest

from commpy import GF2m, PrimeField
from commpy._fields.polynomials import (
    poly_add,
    poly_divmod,
    poly_eval,
    poly_mul,
    poly_sub,
    poly_trim,
)


def test_poly_trim_removes_trailing_zeros():
    np.testing.assert_array_equal(poly_trim([1, 2, 0, 0]), [1, 2])


def test_poly_trim_zero_polynomial_stays_length_one():
    np.testing.assert_array_equal(poly_trim([0, 0, 0]), [0])


def test_poly_add_over_gf2():
    field = GF2m(3)
    # (1 + x) + (1 + x^2) = x + x^2  (GF(2): 1+1=0)
    result = poly_add([1, 1, 0], [1, 0, 1], field)
    np.testing.assert_array_equal(result, [0, 1, 1])


def test_poly_sub_equals_add_over_gf2():
    field = GF2m(3)
    # Over characteristic 2, subtraction and addition coincide.
    a, b = [1, 1, 0], [1, 0, 1]
    np.testing.assert_array_equal(poly_sub(a, b, field), poly_add(a, b, field))


def test_poly_mul_matches_hand_computed_example_over_gf2():
    field = GF2m(3)
    # (1 + x) * (1 + x) = 1 + x^2   (GF(2): the x + x cross term cancels)
    result = poly_mul([1, 1], [1, 1], field)
    np.testing.assert_array_equal(result, [1, 0, 1])


def test_poly_mul_degree_is_sum_of_degrees():
    field = PrimeField(5)
    a = [1, 2, 3]  # degree 2
    b = [4, 1]  # degree 1
    result = poly_mul(a, b, field)
    assert len(result) - 1 == 3  # degree 3


def test_poly_divmod_recovers_original_via_a_eq_qb_plus_r():
    field = PrimeField(7)
    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)  # 1 + 2x + 3x^2 + 4x^3 + 5x^4
    b = np.array([1, 1], dtype=np.int64)  # 1 + x
    q, r = poly_divmod(a, b, field)
    reconstructed = poly_add(poly_mul(q, b, field), r, field)
    np.testing.assert_array_equal(poly_trim(reconstructed), poly_trim(a))


def test_poly_divmod_exact_division_has_zero_remainder():
    field = GF2m(3)
    b = [1, 1, 0]  # 1 + x
    factor = [1, 0, 1]  # 1 + x^2
    a = poly_mul(b, factor, field)
    q, r = poly_divmod(a, b, field)
    np.testing.assert_array_equal(poly_trim(r), [0])
    np.testing.assert_array_equal(poly_trim(q), poly_trim(factor))


def test_poly_divmod_by_zero_raises():
    field = PrimeField(5)
    with pytest.raises(ZeroDivisionError):
        poly_divmod([1, 2, 3], [0], field)


def test_poly_eval_matches_direct_computation_prime_field():
    field = PrimeField(7)
    coeffs = [1, 2, 3]  # 1 + 2x + 3x^2
    for x in range(7):
        expected = (1 + 2 * x + 3 * x**2) % 7
        assert poly_eval(coeffs, x, field) == expected


def test_poly_eval_vectorizes_over_array_of_points():
    field = PrimeField(7)
    coeffs = [1, 2, 3]
    xs = np.arange(7)
    expected = np.array([(1 + 2 * x + 3 * x**2) % 7 for x in xs])
    result = poly_eval(coeffs, xs, field)
    np.testing.assert_array_equal(result, expected)
