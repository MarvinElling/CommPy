"""Reed-Solomon codes over GF(2**m).

Unlike binary BCH, Reed-Solomon symbols are themselves GF(2**m)-valued, and
the generator polynomial's roots are directly consecutive powers of a
primitive element (no minimal-polynomial/LCM step is needed, since the code
is defined natively over GF(2**m) rather than being the GF(2) restriction of
a GF(2**m) construction). Decoding therefore needs an extra step beyond BCH:
Forney's algorithm to recover error *magnitudes*, not just positions, since
an error here isn't simply "flip the bit" -- it can be any nonzero symbol
offset. A separate, simpler erasures-only path is also provided, which
corrects up to `n - k` erasures at *known* positions (vs. `(n-k)//2` errors
at unknown positions) by skipping Berlekamp-Massey/Chien search entirely.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from commpy._channelCoding.block._algebraic import (
    berlekamp_massey,
    chien_search,
    error_evaluator_polynomial,
    formal_derivative,
    forney_magnitudes,
)
from commpy._channelCoding.block.cyclic import CyclicCode
from commpy._fields.binary_extension_field import GF2m
from commpy._fields.polynomials import poly_eval, poly_mul, poly_trim


def _rs_generator_polynomial(field: GF2m, n_minus_k: int) -> NDArray[np.int64]:
    """Generator polynomial `g(x) = product_{i=1}^{n-k} (x - alpha**i)`."""
    g = np.array([1], dtype=np.int64)
    for i in range(1, n_minus_k + 1):
        alpha_i = int(field.exp(i))
        factor = np.array([alpha_i, 1], dtype=np.int64)  # (x - alpha^i) == (x + alpha^i), char 2
        g = poly_mul(g, factor, field)
    return poly_trim(g)


class ReedSolomonCode:
    """Reed-Solomon code over GF(2**m): length `n = 2**m - 1`, dimension `k`.

    Corrects any pattern of up to `t = (n - k) // 2` symbol errors at unknown
    positions (`decode`), or up to `n - k` symbol erasures at known positions
    (`decode_erasures`).
    """

    def __init__(self, m: int, k: int) -> None:
        """Construct RS(`2**m - 1`, `k`) over GF(2**m).

        Args:
            m: GF(2**m) extension degree; the code has length `n = 2**m - 1`.
            k: Message (dimension) length; `0 < k < n`.

        Raises:
            ValueError: If `k` is not in `(0, n)`.
        """
        self.m = m
        self.field = GF2m(m)
        self.n = self.field.order - 1
        if not 0 < k < self.n:
            msg = f'k must satisfy 0 < k < n={self.n}, got {k}.'
            raise ValueError(msg)
        self.k = k
        self.n_minus_k = self.n - k
        self.t = self.n_minus_k // 2
        self.generator = _rs_generator_polynomial(self.field, self.n_minus_k)
        self._cyclic = CyclicCode(self.n, self.generator, self.field)

    def encode(self, message: ArrayLike) -> NDArray[np.int64]:
        """Systematically encode a length-`k` message (GF(2**m) symbols) into a codeword."""
        return self._cyclic.encode(message)

    def decode(self, received: ArrayLike) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
        """Decode a length-`n` received word, correcting up to `t` symbol errors.

        Args:
            received: Received codeword, length `n`, GF(2**m) symbols.

        Returns:
            `(message, corrected_codeword, num_errors_corrected)`.

        Raises:
            ValueError: If `len(received) != n`, or decoding fails because
                more errors occurred than `t` (detected via a Chien-search
                root-count mismatch).
        """
        r = np.asarray(received, dtype=np.int64)
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

        magnitudes = forney_magnitudes(syndromes, sigma, error_positions, self.field)
        corrected = r.copy()
        corrected[error_positions] = self.field.add(corrected[error_positions], magnitudes)
        return self._cyclic.extract_message(corrected), corrected, len(error_positions)

    def decode_erasures(
        self, received: ArrayLike, erasure_positions: ArrayLike,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Decode a length-`n` received word with known erasure positions (no unknown errors).

        Args:
            received: Received codeword, length `n`. Values at
                `erasure_positions` are ignored (treated as unknown).
            erasure_positions: 0-indexed positions of the erased symbols.

        Returns:
            `(message, corrected_codeword)`.

        Raises:
            ValueError: If `len(received) != n`, or more than `n - k`
                positions are erased (uncorrectable).
        """
        r = np.asarray(received, dtype=np.int64).copy()
        if r.size != self.n:
            err = f'received must have length {self.n}, got {r.size}.'
            raise ValueError(err)
        erasures = np.asarray(erasure_positions, dtype=np.int64)
        if erasures.size > self.n_minus_k:
            msg = f'Cannot correct {erasures.size} erasures: at most {self.n_minus_k} supported.'
            raise ValueError(msg)
        if erasures.size == 0:
            return self._cyclic.extract_message(r), r

        r[erasures] = 0  # treat erased symbols as 0 for the syndrome/Forney computation below

        exponents = np.arange(1, self.n_minus_k + 1)
        alpha_js = self.field.exp(exponents)
        syndromes = np.asarray(poly_eval(r, alpha_js, self.field))

        lam = np.array([1], dtype=np.int64)
        for pos in erasures:
            locator = int(self.field.exp(int(pos)))
            # (1 - X_l*x), X_l = alpha**pos: constant term 1 (matching the same
            # sigma[0]==1 normalization Berlekamp-Massey produces), root at
            # x = X_l**-1 = alpha**-pos -- the same convention chien_search and
            # forney_magnitudes use for the error-only path.
            factor = np.array([1, locator], dtype=np.int64)
            lam = poly_mul(lam, factor, self.field)

        omega = error_evaluator_polynomial(syndromes, lam, self.field)
        lam_deriv = formal_derivative(lam)
        x_inv = self.field.exp(-erasures)
        omega_vals = np.asarray(poly_eval(omega, x_inv, self.field))
        deriv_vals = np.asarray(poly_eval(lam_deriv, x_inv, self.field))
        magnitudes = np.asarray(self.field.divide(omega_vals, deriv_vals))

        r[erasures] = magnitudes
        return self._cyclic.extract_message(r), r


__all__ = ['ReedSolomonCode']
