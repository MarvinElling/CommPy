import numpy as np

from CommPy import is_prime


class PrimeField:
    """Class for prime fields GF(p) where p is a prime number.
    """

    def __init__(self, p):
        if not isinstance(p, int) or p <= 1 or not is_prime(p):
            raise ValueError('p must be a prime number greater than 1.')
        self.p = p
        self.elements = np.arange(p)

    def add(self, a, b):
        """Add two elements in the field."""
        return (a + b) % self.p

    def subtract(self, a, b):
        """Subtract two elements in the field."""
        return (a - b) % self.p

    def primitive_roots(self):
        """Return all primitive roots of the field."""
        roots = []
        for g in self.elements:
            if g == 0 or g == 1:
                continue
            order = 1
            current = g
            while current != 1:
                current = (current * g) % self.p
                order += 1
            if order == self.p - 1:
                roots.append(g)
        return roots


if __name__ == "__main__":
    # Example usage
    p = 7  # A prime number
    field = PrimeField(p)

    print("Field elements:", field.elements)
    print("Addition (3 + 5):", field.add(np.array([3,3]), np.array([5,5])))
    print("Subtraction (5 - 3):", field.subtract(5, 3))
    print("All primitive roots:", field.primitive_roots())
