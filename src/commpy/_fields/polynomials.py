"""Polynomial arithmetic over a `FiniteField`.

Polynomials are represented as 1-D numpy integer arrays of field-element
coefficients, index `i` holding the coefficient of `x**i` (low-to-high
degree order) -- the natural order for the shift-register / generator-
polynomial constructions used throughout coding theory (Phase 2's cyclic,
BCH, and Reed-Solomon codes).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from commpy._fields.base import FiniteField

Poly = NDArray[np.int64]


def poly_trim(coeffs: ArrayLike) -> Poly:
    """Remove trailing (highest-degree) zero coefficients.

    Always returns at least a length-1 array (representing the zero polynomial).
    """
    coeffs_arr = np.asarray(coeffs, dtype=np.int64)
    nonzero = np.nonzero(coeffs_arr)[0]
    if nonzero.size == 0:
        return np.zeros(1, dtype=np.int64)
    return coeffs_arr[:nonzero[-1] + 1]


def poly_add(a: ArrayLike, b: ArrayLike, field: FiniteField) -> Poly:
    """Add two polynomials over `field`."""
    a_arr = np.asarray(a, dtype=np.int64)
    b_arr = np.asarray(b, dtype=np.int64)
    n = max(len(a_arr), len(b_arr))
    a_pad = np.zeros(n, dtype=np.int64)
    b_pad = np.zeros(n, dtype=np.int64)
    a_pad[:len(a_arr)] = a_arr
    b_pad[:len(b_arr)] = b_arr
    return poly_trim(np.asarray(field.add(a_pad, b_pad)))


def poly_sub(a: ArrayLike, b: ArrayLike, field: FiniteField) -> Poly:
    """Subtract two polynomials over `field` (`a - b`)."""
    a_arr = np.asarray(a, dtype=np.int64)
    b_arr = np.asarray(b, dtype=np.int64)
    n = max(len(a_arr), len(b_arr))
    a_pad = np.zeros(n, dtype=np.int64)
    b_pad = np.zeros(n, dtype=np.int64)
    a_pad[:len(a_arr)] = a_arr
    b_pad[:len(b_arr)] = b_arr
    return poly_trim(np.asarray(field.subtract(a_pad, b_pad)))


def poly_mul(a: ArrayLike, b: ArrayLike, field: FiniteField) -> Poly:
    """Multiply two polynomials over `field` (convolution of their coefficients)."""
    a_arr = poly_trim(a)
    b_arr = poly_trim(b)
    result = np.zeros(len(a_arr) + len(b_arr) - 1, dtype=np.int64)
    for i, coeff in enumerate(a_arr):
        if coeff == 0:
            continue
        term = np.asarray(field.multiply(int(coeff), b_arr))
        result[i:i + len(b_arr)] = field.add(result[i:i + len(b_arr)], term)
    return poly_trim(result)


def poly_divmod(a: ArrayLike, b: ArrayLike, field: FiniteField) -> tuple[Poly, Poly]:
    """Polynomial long division over `field`: return `(quotient, remainder)` with `a = q*b + r`.

    Raises:
        ZeroDivisionError: If `b` is the zero polynomial.
    """
    a_arr = poly_trim(a)
    b_arr = poly_trim(b)
    if len(b_arr) == 1 and b_arr[0] == 0:
        msg = 'Cannot divide by the zero polynomial.'
        raise ZeroDivisionError(msg)

    deg_b = len(b_arr) - 1
    lead_b = int(b_arr[deg_b])
    remainder = a_arr.copy()
    quotient = np.zeros(max(len(a_arr) - deg_b, 1), dtype=np.int64)

    while True:
        remainder = poly_trim(remainder)
        deg_r = len(remainder) - 1
        is_zero = deg_r == 0 and remainder[0] == 0
        if is_zero or deg_r < deg_b:
            break
        factor = int(field.divide(int(remainder[deg_r]), lead_b))
        quotient[deg_r - deg_b] = factor
        sub_term = np.zeros(deg_r + 1, dtype=np.int64)
        sub_term[deg_r - deg_b:] = np.asarray(field.multiply(factor, b_arr))
        remainder = np.asarray(field.subtract(remainder, sub_term))

    return poly_trim(quotient), poly_trim(remainder)


def poly_eval(coeffs: ArrayLike, x: ArrayLike, field: FiniteField) -> NDArray[np.int64] | int:
    """Evaluate a polynomial at `x` via Horner's method (vectorized if `x` is an array)."""
    coeffs_arr = poly_trim(coeffs)
    x_arr = np.asarray(x, dtype=np.int64)
    result = np.zeros(x_arr.shape, dtype=np.int64)
    for coeff in coeffs_arr[::-1]:
        result = np.asarray(field.add(field.multiply(result, x_arr), int(coeff)))
    return int(result) if result.ndim == 0 else result


__all__ = ['poly_add', 'poly_divmod', 'poly_eval', 'poly_mul', 'poly_sub', 'poly_trim']
