"""Shared algebraic decoding primitives for BCH and Reed-Solomon codes.

Both codes decode by: (1) computing syndromes by evaluating the received
polynomial at consecutive powers of a primitive element, (2) running
Berlekamp-Massey on those syndromes to find the error-locator polynomial,
and (3) a Chien search over all nonzero field elements to find its roots
(the error positions). Reed-Solomon additionally runs Forney's algorithm to
recover error *magnitudes* (binary BCH doesn't need this: every error is
simply a bit flip).
"""

import numpy as np
from numpy.typing import NDArray

from commpy._fields.binary_extension_field import GF2m
from commpy._fields.polynomials import poly_add, poly_eval, poly_mul, poly_trim


def berlekamp_massey(syndromes: NDArray[np.int64], field: GF2m) -> NDArray[np.int64]:
    """Find the minimal-degree error-locator polynomial sigma(x) via Berlekamp-Massey.

    `syndromes` = `[S_1, S_2, ..., S_2t]` (GF(2**m) elements). Standard
    formulation (e.g. Lin & Costello); GF(2**m) has characteristic 2, so
    subtraction and addition coincide throughout.
    """
    sigma = np.array([1], dtype=np.int64)
    prev_sigma = np.array([1], dtype=np.int64)
    length = 0
    shift = 1
    prev_discrepancy = 1

    for n_iter in range(len(syndromes)):
        discrepancy = int(syndromes[n_iter])
        for i in range(1, length + 1):
            if i < len(sigma):
                term = field.multiply(sigma[i], syndromes[n_iter - i])
                discrepancy = int(field.add(discrepancy, term))

        if discrepancy == 0:
            shift += 1
            continue

        coef = field.divide(discrepancy, prev_discrepancy)
        correction = np.zeros(len(prev_sigma) + shift, dtype=np.int64)
        correction[shift:] = np.asarray(field.multiply(coef, prev_sigma))
        new_sigma = poly_add(sigma, correction, field)

        if 2 * length <= n_iter:
            prev_sigma = sigma
            length = n_iter + 1 - length
            prev_discrepancy = discrepancy
            shift = 1
        else:
            shift += 1

        sigma = new_sigma

    return poly_trim(sigma)


def chien_search(sigma: NDArray[np.int64], field: GF2m) -> NDArray[np.int64]:
    """Find the roots of `sigma(x)` among `alpha**0 .. alpha**(n-1)` via a full search.

    Returns the (0-indexed, low-degree-first) codeword positions `i` for
    which `alpha**(-i)` is a root of `sigma`, i.e. the error positions.
    Fully vectorized: evaluates `sigma` at all `n` candidate points at once.
    """
    n = field.order - 1
    positions = np.arange(n)
    candidates = field.exp(-positions)
    values = np.asarray(poly_eval(sigma, candidates, field))
    return np.asarray(positions[values == 0])


def error_evaluator_polynomial(
    syndromes: NDArray[np.int64], sigma: NDArray[np.int64], field: GF2m,
) -> NDArray[np.int64]:
    """Error evaluator `Omega(x) = [S(x) * sigma(x)] mod x**(len(syndromes))`.

    `S(x) = S_1 + S_2*x + ... + S_2t*x**(2t-1)` is the syndrome polynomial.
    Used by Reed-Solomon's Forney algorithm to recover error magnitudes.
    """
    s_poly = poly_trim(syndromes)
    product = poly_mul(s_poly, sigma, field)
    return poly_trim(product[:len(syndromes)])


def formal_derivative(poly: NDArray[np.int64]) -> NDArray[np.int64]:
    """Formal derivative of `poly` over a characteristic-2 field.

    Even-degree terms always vanish at characteristic 2 (their coefficient
    is multiplied by an even integer, i.e. by 0): only odd-degree terms of
    `poly` survive, each shifted down one degree.
    """
    deriv = np.zeros(max(len(poly) - 1, 1), dtype=np.int64)
    for j in range(1, len(poly), 2):
        deriv[j - 1] = poly[j]
    return poly_trim(deriv)


def forney_magnitudes(
    syndromes: NDArray[np.int64],
    sigma: NDArray[np.int64],
    error_positions: NDArray[np.int64],
    field: GF2m,
) -> NDArray[np.int64]:
    """Recover error magnitudes at `error_positions` via Forney's algorithm.

    `Y_l = Omega(X_l**-1) / sigma'(X_l**-1)`, where `X_l = alpha**error_positions[l]`
    (characteristic 2, so the usual formula's leading minus sign is a no-op).
    """
    omega = error_evaluator_polynomial(syndromes, sigma, field)
    sigma_deriv = formal_derivative(sigma)
    x_inv = field.exp(-error_positions)
    omega_vals = np.asarray(poly_eval(omega, x_inv, field))
    deriv_vals = np.asarray(poly_eval(sigma_deriv, x_inv, field))
    return np.asarray(field.divide(omega_vals, deriv_vals))


__all__ = [
    'berlekamp_massey',
    'chien_search',
    'error_evaluator_polynomial',
    'formal_derivative',
    'forney_magnitudes',
]
