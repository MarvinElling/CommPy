"""Tests for commpy.GF2m (binary extension field GF(2**m) arithmetic)."""

import numpy as np
import pytest

from commpy import GF2m
from commpy._fields.binary_extension_field import find_primitive_polynomial, is_primitive_polynomial


def test_find_primitive_polynomial_matches_known_values():
    # Cross-checked against a from-scratch reference computation; m=8's result
    # (0x11D) also happens to match the conventional GF(256) used for QR
    # codes / CDs, though matching that convention isn't required for correctness.
    assert find_primitive_polynomial(2) == 0b111
    assert find_primitive_polynomial(3) == 0b1011
    assert find_primitive_polynomial(4) == 0b10011
    assert find_primitive_polynomial(8) == 0x11D


def test_rejects_non_primitive_modulus():
    # Find a non-primitive degree-3 polynomial via the checker itself, rather
    # than asserting a specific value is non-primitive from memory.
    non_primitive = next(
        p for p in range(1 << 3, 1 << 4) if p & 1 and not is_primitive_polynomial(3, p)
    )
    with pytest.raises(ValueError, match='primitive'):
        GF2m(3, modulus_poly=non_primitive)


def test_rejects_non_positive_degree():
    with pytest.raises(ValueError, match='positive'):
        GF2m(0)


def test_order_and_characteristic():
    field = GF2m(4)
    assert field.order == 16
    assert field.characteristic == 2
    assert len(field.elements) == 16


def test_add_and_subtract_are_xor():
    field = GF2m(4)
    assert field.add(5, 3) == 5 ^ 3
    assert field.subtract(5, 3) == 5 ^ 3  # characteristic 2: subtraction == addition


def test_multiply_by_zero_is_zero():
    field = GF2m(4)
    for a in range(field.order):
        assert field.multiply(a, 0) == 0
        assert field.multiply(0, a) == 0


def test_multiply_divide_are_inverse_for_all_nonzero_elements():
    field = GF2m(4)
    for a in range(1, field.order):
        for b in range(1, field.order):
            product = field.multiply(a, b)
            assert field.divide(product, b) == a


def test_divide_by_zero_raises():
    field = GF2m(4)
    with pytest.raises(ZeroDivisionError):
        field.divide(1, 0)


def test_every_nonzero_element_raised_to_order_minus_one_is_one():
    # Lagrange's theorem: a**(order-1) == 1 for every nonzero a.
    field = GF2m(4)
    for a in range(1, field.order):
        assert field.power(a, field.order - 1) == 1


def test_power_zero_exponent_is_one():
    field = GF2m(4)
    assert field.power(5, 0) == 1


def test_operations_vectorize_over_arrays():
    field = GF2m(4)
    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7, 8])
    expected_add = np.array([field.add(int(x), int(y)) for x, y in zip(a, b, strict=True)])
    np.testing.assert_array_equal(np.asarray(field.add(a, b)), expected_add)

    expected_mul = np.array([field.multiply(int(x), int(y)) for x, y in zip(a, b, strict=True)])
    np.testing.assert_array_equal(np.asarray(field.multiply(a, b)), expected_mul)


def test_explicit_modulus_poly_is_used():
    poly = find_primitive_polynomial(3)
    field = GF2m(3, modulus_poly=poly)
    assert field.modulus_poly == poly
