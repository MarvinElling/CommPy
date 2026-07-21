"""Binary BCH (Bose-Chaudhuri-Hocquenghem) codes.

Built as a `CyclicCode` (see `.cyclic`) whose generator polynomial is the
product of the distinct minimal polynomials (over GF(2)) of
`alpha, alpha**2, ..., alpha**(2t)` for a primitive element `alpha` of
GF(2**m) -- guaranteeing the code corrects any pattern of up to `t` errors.
Decoding uses syndromes, Berlekamp-Massey, and a Chien search (`._algebraic`);
since errors are binary, correction is a simple bit flip at each located
position (no Forney magnitude step needed, unlike Reed-Solomon).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from commpy._channelCoding.block._algebraic import berlekamp_massey, chien_search
from commpy._channelCoding.block.cyclic import CyclicCode
from commpy._fields.binary_extension_field import GF2m
from commpy._fields.polynomials import poly_eval, poly_mul, poly_trim
from commpy._fields.prime_field import PrimeField


def _minimal_polynomial(beta: int, field: GF2m) -> NDArray[np.int64]:
    """Minimal polynomial of `beta` in GF(2**m) over GF(2), as a 0/1 coefficient array."""
    conjugates = []
    c = beta
    while c not in conjugates:
        conjugates.append(c)
        c = int(field.multiply(c, c))

    poly = np.array([1], dtype=np.int64)
    for c in conjugates:
        factor = np.array([c, 1], dtype=np.int64)  # (x + c) == (x - c) at characteristic 2
        poly = poly_mul(poly, factor, field)

    if not np.all((poly == 0) | (poly == 1)):
        msg = 'internal error: minimal polynomial coefficients are not in GF(2).'
        raise RuntimeError(msg)
    return poly_trim(poly)


def _bch_generator_polynomial(field: GF2m, t: int) -> NDArray[np.int64]:
    """Generator polynomial for a `t`-error-correcting binary BCH code over GF(2**m)."""
    gf2 = PrimeField(2)
    n = field.order - 1
    seen_exponents: set[int] = set()
    g = np.array([1], dtype=np.int64)

    for i in range(1, 2 * t + 1):
        if i in seen_exponents:
            continue
        beta = int(field.exp(i))
        min_poly = _minimal_polynomial(beta, field)
        g = poly_mul(g, min_poly, gf2)

        # alpha**i's whole conjugacy class {alpha**(i*2^j)} shares this same
        # minimal polynomial; mark those exponents covered too.
        exp = i
        while exp not in seen_exponents:
            seen_exponents.add(exp)
            exp = (2 * exp) % n

    return poly_trim(g)


class BCHCode:
    """Binary BCH code over GF(2**m), correcting up to `t` errors."""

    def __init__(self, m: int, t: int) -> None:
        """Construct a `t`-error-correcting binary BCH code of length `2**m - 1`.

        Args:
            m: GF(2**m) extension degree.
            t: Number of errors to guarantee correcting.

        Raises:
            ValueError: If `t` is too large for the given `m` (generator
                polynomial degree would consume the whole codeword).
        """
        self.m = m
        self.t = t
        self.field = GF2m(m)
        self.n = self.field.order - 1
        generator = _bch_generator_polynomial(self.field, t)
        if len(generator) - 1 >= self.n:
            msg = f't={t} is too large for m={m}: no data bits would remain.'
            raise ValueError(msg)
        self._cyclic = CyclicCode(self.n, generator, PrimeField(2))
        self.k = self._cyclic.k
        self.generator = generator

    def encode(self, message: ArrayLike) -> NDArray[np.int64]:
        """Systematically encode a length-`k` message into a length-`n` codeword."""
        return self._cyclic.encode(message)

    def decode(self, received: ArrayLike) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
        """Decode a length-`n` received word, correcting up to `t` errors.

        Args:
            received: Received codeword, length `n`.

        Returns:
            `(message, corrected_codeword, num_errors_corrected)`.

        Raises:
            ValueError: If `len(received) != n`, or decoding fails because
                more than `t` errors occurred (detected when the number of
                Chien-search roots doesn't match the error-locator degree).
        """
        r = np.asarray(received, dtype=np.int64) % 2
        if r.size != self.n:
            err = f'received must have length {self.n}, got {r.size}.'
            raise ValueError(err)

        exponents = np.arange(1, 2 * self.t + 1)
        alpha_js = self.field.exp(exponents)
        syndromes = np.asarray(poly_eval(r, alpha_js, self.field))
        if np.all(syndromes == 0):
            return self._cyclic.extract_message(r), r, 0

        sigma = berlekamp_massey(syndromes, self.field)
        error_positions = chien_search(sigma, self.field)
        if len(error_positions) != len(sigma) - 1:
            msg = 'Decoding failure: too many errors to correct.'
            raise ValueError(msg)

        corrected = r.copy()
        corrected[error_positions] ^= 1
        return self._cyclic.extract_message(corrected), corrected, len(error_positions)


__all__ = ['BCHCode']
