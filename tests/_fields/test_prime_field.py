"""Tests for commpy.PrimeField (GF(p) arithmetic)."""

import numpy as np
import pytest

from commpy import PrimeField


def test_rejects_non_prime():
    with pytest.raises(ValueError, match='prime'):
        PrimeField(4)


def test_rejects_non_positive():
    with pytest.raises(ValueError, match='prime'):
        PrimeField(1)


def test_arithmetic_wraps_mod_p():
    field = PrimeField(7)
    assert field.add(3, 5) == 1
    assert field.subtract(3, 5) == 5
    assert field.multiply(3, 5) == 1
    assert field.divide(6, 3) == 2


def test_arithmetic_is_vectorized():
    field = PrimeField(7)
    a = np.array([3, 3])
    b = np.array([5, 5])
    np.testing.assert_array_equal(field.add(a, b), [1, 1])


def test_divide_by_zero_raises():
    field = PrimeField(7)
    with pytest.raises(ZeroDivisionError):
        field.divide(1, 0)


def test_every_nonzero_element_has_a_multiplicative_inverse():
    field = PrimeField(11)
    for a in range(1, 11):
        b = field.divide(1, a)
        assert field.multiply(a, b) == 1


def test_primitive_roots_of_gf7():
    # GF(7)* is cyclic of order 6; known primitive roots are 3 and 5.
    field = PrimeField(7)
    assert set(field.primitive_roots()) == {3, 5}


def test_power_default_implementation_via_repeated_squaring():
    # PrimeField doesn't override FiniteField.power(); exercise the inherited default.
    field = PrimeField(7)
    for a in range(1, 7):
        for k in range(6):
            assert field.power(a, k) == pow(a, k, 7)


def test_power_rejects_negative_exponent():
    field = PrimeField(7)
    with pytest.raises(ValueError, match='non-negative'):
        field.power(3, -1)


def test_negate_default_implementation():
    # PrimeField doesn't override FiniteField.negate(); exercise the inherited default.
    field = PrimeField(7)
    for a in range(7):
        assert field.add(a, field.negate(a)) == 0


def test_divide_rejects_array_divisor():
    field = PrimeField(7)
    with pytest.raises(TypeError, match='scalar'):
        field.divide(1, np.array([1, 2]))
