"""Generic cyclic block code from a generator polynomial.

BCH and Reed-Solomon codes (see `.bch` / `.reed_solomon`) are cyclic codes
with a specifically constructed generator polynomial; this module provides
the shared systematic encode / syndrome machinery each builds on.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from commpy._fields.base import FiniteField
from commpy._fields.polynomials import poly_divmod, poly_trim


class CyclicCode:
    """Systematic cyclic code of length `n` over `field`, from generator polynomial `g(x)`.

    `g(x)` must divide `x**n - 1` over `field`; this is verified at
    construction. Encoding is systematic: for message polynomial `m(x)`
    (degree < k), the codeword is `c(x) = m(x)*x**deg(g) - (m(x)*x**deg(g) mod g(x))`,
    which is always divisible by `g(x)` and whose top `k` coefficients equal
    the original message unchanged.
    """

    def __init__(self, n: int, generator: ArrayLike, field: FiniteField) -> None:
        """Construct a cyclic code of length `n` from generator polynomial `generator`.

        Raises:
            ValueError: If `generator` does not divide `x**n - 1` over `field`.
        """
        self.field = field
        self.n = n
        self.generator = poly_trim(generator)
        self.deg_g = len(self.generator) - 1
        self.k = n - self.deg_g

        x_n_minus_1 = np.zeros(n + 1, dtype=np.int64)
        x_n_minus_1[0] = field.negate(1)
        x_n_minus_1[n] = 1
        _, remainder = poly_divmod(x_n_minus_1, self.generator, field)
        if not (len(remainder) == 1 and remainder[0] == 0):
            msg = 'generator does not divide x**n - 1 over the given field.'
            raise ValueError(msg)

    def encode(self, message: ArrayLike) -> NDArray[np.int64]:
        """Systematically encode a length-`k` message into a length-`n` codeword.

        Raises:
            ValueError: If `len(message) != k`.
        """
        msg = np.asarray(message, dtype=np.int64)
        if msg.size != self.k:
            err = f'message must have length {self.k}, got {msg.size}.'
            raise ValueError(err)
        shifted = np.zeros(self.n, dtype=np.int64)
        shifted[self.deg_g:] = msg
        _, remainder = poly_divmod(shifted, self.generator, self.field)

        parity = np.zeros(self.deg_g, dtype=np.int64)
        neg_rem = np.asarray(self.field.negate(remainder))
        parity[:len(neg_rem)] = neg_rem

        codeword = shifted.copy()
        codeword[:self.deg_g] = parity
        return codeword

    def syndrome(self, received: ArrayLike) -> NDArray[np.int64]:
        """Return `received(x) mod g(x)`; zero iff `received` is a valid codeword.

        Raises:
            ValueError: If `len(received) != n`.
        """
        r = np.asarray(received, dtype=np.int64)
        if r.size != self.n:
            err = f'received must have length {self.n}, got {r.size}.'
            raise ValueError(err)
        _, remainder = poly_divmod(r, self.generator, self.field)
        return remainder

    def is_codeword(self, received: ArrayLike) -> bool:
        """Whether `received` is divisible by the generator polynomial."""
        rem = self.syndrome(received)
        return bool(len(rem) == 1 and rem[0] == 0)

    def extract_message(self, codeword: ArrayLike) -> NDArray[np.int64]:
        """Extract the `k`-bit message from a (systematically encoded) codeword."""
        cw = np.asarray(codeword, dtype=np.int64)
        return cw[self.deg_g:]


__all__ = ['CyclicCode']
