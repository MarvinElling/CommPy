"""Abstract base class for finite (Galois) field arithmetic."""

from abc import ABC, abstractmethod

import numpy as np

#: An element (or array of elements) of a finite field.
FieldValue = int | np.ndarray


def as_field_value(arr: np.ndarray) -> FieldValue:
    """Collapse a 0-d numpy array back to a plain Python int, else return it as-is.

    Field operations are implemented with numpy so they vectorize over arrays,
    but scalar-in-scalar-out ergonomics (matching `PrimeField`'s original
    behavior) are preserved by unwrapping 0-d results.
    """
    return int(arr) if arr.ndim == 0 else arr


class FiniteField(ABC):
    """Common interface for finite (Galois) field arithmetic.

    Concrete fields (`PrimeField` for GF(p), `GF2m` for GF(2**m)) implement
    the four basic operations; `power` and `negate` are provided generically
    in terms of them so subclasses need not reimplement either.
    """

    order: int
    characteristic: int

    @abstractmethod
    def add(self, a: FieldValue, b: FieldValue) -> FieldValue:
        """Add two field elements."""

    @abstractmethod
    def subtract(self, a: FieldValue, b: FieldValue) -> FieldValue:
        """Subtract two field elements (`a - b`)."""

    @abstractmethod
    def multiply(self, a: FieldValue, b: FieldValue) -> FieldValue:
        """Multiply two field elements."""

    @abstractmethod
    def divide(self, a: FieldValue, b: FieldValue) -> FieldValue:
        """Divide two field elements (`a / b`).

        Raises:
            ZeroDivisionError: If `b` is zero.
        """

    def negate(self, a: FieldValue) -> FieldValue:
        """Return the additive inverse of `a`."""
        return self.subtract(0, a)

    def power(self, a: FieldValue, k: int) -> FieldValue:
        """Raise `a` to a non-negative integer power `k` via repeated squaring.

        Raises:
            ValueError: If `k` is negative.
        """
        if k < 0:
            msg = 'k must be non-negative.'
            raise ValueError(msg)
        result: FieldValue = 1
        base = a
        while k > 0:
            if k & 1:
                result = self.multiply(result, base)
            base = self.multiply(base, base)
            k >>= 1
        return result
