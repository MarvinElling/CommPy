"""Binary extension field GF(2**m) arithmetic via log/antilog tables.

Needed by Reed-Solomon and BCH codes (Phase 2), which operate over GF(2**m)
rather than a prime field.
"""

import numpy as np

from commpy._fields.base import FieldValue, FiniteField, as_field_value


def _mul_by_x(state: int, m: int, poly: int) -> int:
    """Multiply `state` (a field element, as an m-bit integer) by x, mod `poly`."""
    shifted = state << 1
    if shifted & (1 << m):
        shifted ^= poly
    return shifted & ((1 << m) - 1)


def _order_of_x(m: int, poly: int) -> int:
    """Multiplicative order of x in GF(2)[x]/(poly), or -1 if never 1 within range."""
    state = 1
    limit = (1 << m) - 1
    for k in range(1, limit + 1):
        state = _mul_by_x(state, m, poly)
        if state == 1:
            return k
    return -1


def is_primitive_polynomial(m: int, poly: int) -> bool:
    """Whether `poly` (degree-`m`, bit `i` = coefficient of `x**i`) is primitive over GF(2)."""
    return _order_of_x(m, poly) == (1 << m) - 1


def find_primitive_polynomial(m: int) -> int:
    """Find the numerically smallest primitive polynomial of degree `m` over GF(2).

    The polynomial is returned as an integer whose bit `i` is the coefficient
    of `x**i` (e.g. `x**3 + x + 1` is encoded as `0b1011 == 11`). Any
    primitive polynomial of degree `m` defines an isomorphic copy of GF(2**m),
    so the specific choice returned here does not need to match the
    convention used by any particular external reference/library.

    Raises:
        RuntimeError: If no primitive polynomial of degree `m` is found
            (should not happen for any valid `m >= 1`).
    """
    for candidate in range(1 << m, 1 << (m + 1)):
        if candidate & 1 and is_primitive_polynomial(m, candidate):
            return candidate
    msg = f'No primitive polynomial of degree {m} found.'
    raise RuntimeError(msg)


class GF2m(FiniteField):
    """Binary extension field GF(2**m).

    Elements are represented as plain ints in `[0, 2**m)`: the standard
    bit-packed-polynomial representation, bit `i` holding the coefficient of
    `x**i`. Multiplication and division are O(1) per element via precomputed
    log/antilog tables and fully vectorize over numpy arrays (no Python loop,
    no JIT needed).
    """

    def __init__(self, m: int, modulus_poly: int | None = None) -> None:
        """Construct GF(2**m).

        Args:
            m: Extension degree; the field has `2**m` elements.
            modulus_poly: An explicit primitive polynomial of degree `m` (bit
                `i` = coefficient of `x**i`). If omitted, the numerically
                smallest primitive polynomial of degree `m` is used.

        Raises:
            ValueError: If `m` is not a positive integer, or `modulus_poly` is
                given but is not a primitive polynomial of degree `m`.
        """
        if not isinstance(m, int) or m < 1:
            msg = 'm must be a positive integer.'
            raise ValueError(msg)
        if modulus_poly is None:
            modulus_poly = find_primitive_polynomial(m)
        elif not is_primitive_polynomial(m, modulus_poly):
            msg = f'{modulus_poly!r} is not a primitive polynomial of degree {m}.'
            raise ValueError(msg)

        self.m = m
        self.order = 1 << m
        self.characteristic = 2
        self.modulus_poly = modulus_poly
        self.elements = np.arange(self.order)

        # exp[i] = alpha**i for i in [0, order - 2]; log[alpha**i] = i. log[0] is
        # left at 0 (undefined mathematically) but is never read: every lookup
        # against it is masked out by an explicit zero-check first.
        exp_table = np.empty(self.order - 1, dtype=np.int64)
        state = 1
        for i in range(self.order - 1):
            exp_table[i] = state
            state = _mul_by_x(state, m, modulus_poly)
        log_table = np.zeros(self.order, dtype=np.int64)
        log_table[exp_table] = np.arange(self.order - 1)
        self._exp = exp_table
        self._log = log_table

    def add(self, a: FieldValue, b: FieldValue) -> FieldValue:
        """Add two elements (XOR, since GF(2**m) has characteristic 2)."""
        return as_field_value(np.bitwise_xor(np.asarray(a), np.asarray(b)))

    def subtract(self, a: FieldValue, b: FieldValue) -> FieldValue:
        """Subtract two elements (identical to `add` at characteristic 2)."""
        return self.add(a, b)

    def multiply(self, a: FieldValue, b: FieldValue) -> FieldValue:
        """Multiply two elements via log/antilog tables (elementwise if array)."""
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
        zero_mask = (a_arr == 0) | (b_arr == 0)
        log_sum = (self._log[a_arr] + self._log[b_arr]) % (self.order - 1)
        result = np.where(zero_mask, 0, self._exp[log_sum])
        return as_field_value(result)

    def divide(self, a: FieldValue, b: FieldValue) -> FieldValue:
        """Divide two elements via log/antilog tables (elementwise if array).

        Raises:
            ZeroDivisionError: If any element of `b` is zero.
        """
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
        if np.any(b_arr == 0):
            msg = 'Division by zero is not allowed in the field.'
            raise ZeroDivisionError(msg)
        log_diff = (self._log[a_arr] - self._log[b_arr]) % (self.order - 1)
        result = np.where(a_arr == 0, 0, self._exp[log_diff])
        return as_field_value(result)

    def power(self, a: FieldValue, k: int) -> FieldValue:
        """Raise `a` to a non-negative integer power `k` via the log table (O(1))."""
        if k < 0:
            msg = 'k must be non-negative.'
            raise ValueError(msg)
        a_arr = np.asarray(a)
        if k == 0:
            return as_field_value(np.ones_like(a_arr))
        log_k = (self._log[a_arr] * k) % (self.order - 1)
        result = np.where(a_arr == 0, 0, self._exp[log_k])
        return as_field_value(result)

    def exp(self, k: FieldValue) -> FieldValue:
        """Return `alpha**k` for the field's canonical primitive element `alpha` (`== 2`).

        Vectorizes over an array of (possibly negative) exponents `k`, unlike
        `power`, which raises a fixed base to a single scalar exponent.
        """
        k_arr = np.asarray(k) % (self.order - 1)
        return as_field_value(self._exp[k_arr])


__all__ = ['GF2m', 'find_primitive_polynomial', 'is_primitive_polynomial']
