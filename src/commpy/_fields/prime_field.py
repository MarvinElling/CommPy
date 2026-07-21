"""Prime (Galois) field GF(p) arithmetic."""

import numpy as np

from commpy._fields.base import FieldValue, FiniteField
from commpy._utils.maths import is_prime, modinv


class PrimeField(FiniteField):
    """Prime field GF(p), where p is a prime number."""

    def __init__(self, p: int) -> None:
        """Construct GF(p).

        Args:
            p: A prime number greater than 1.

        Raises:
            ValueError: If `p` is not a prime number greater than 1.
        """
        if not isinstance(p, int) or p <= 1 or not is_prime(p):
            msg = 'p must be a prime number greater than 1.'
            raise ValueError(msg)
        self.p = p
        self.order = p
        self.characteristic = p
        self.elements = np.arange(p)

    def add(self, a: FieldValue, b: FieldValue) -> FieldValue:
        """Add two elements in the field."""
        return (a + b) % self.p

    def subtract(self, a: FieldValue, b: FieldValue) -> FieldValue:
        """Subtract two elements in the field."""
        return (a - b) % self.p

    def multiply(self, a: FieldValue, b: FieldValue) -> FieldValue:
        """Multiply two elements in the field."""
        return (a * b) % self.p

    def divide(self, a: FieldValue, b: FieldValue) -> FieldValue:
        """Divide two elements in the field. `b` must be a scalar.

        Raises:
            TypeError: If `b` is not a scalar.
            ZeroDivisionError: If `b` is zero.
        """
        if isinstance(b, np.ndarray):
            msg = 'divide() only supports a scalar divisor b.'
            raise TypeError(msg)
        if b == 0:
            msg = 'Division by zero is not allowed in the field.'
            raise ZeroDivisionError(msg)
        b_inv = modinv(b, self.p)
        return (a * b_inv) % self.p

    def primitive_roots(self) -> list[int]:
        """Return all primitive roots of the field."""
        roots = []
        for g in self.elements:
            if g in {0, 1}:
                continue
            order = 1
            current = g
            while current != 1:
                current = (current * g) % self.p
                order += 1
            if order == self.p - 1:
                roots.append(int(g))
        return roots


if __name__ == '__main__':
    field = PrimeField(7)

    print('Field elements:', field.elements)
    print('Addition (3 + 5):', field.add(np.array([3, 3]), np.array([5, 5])))
    print('Subtraction (5 - 3):', field.subtract(5, 3))
    print('Multiplication (3 * 5):', field.multiply(3, 5))
    print('Division (6 / 3):', field.divide(6, 3))
    print('All primitive roots:', field.primitive_roots())
