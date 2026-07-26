"""Iterative belief-propagation decoding of an LDPC code on its Tanner graph.

Two flooding-schedule message-passing rules are provided:

- `'sum-product'`: the exact (max-log-free) rule, using the hyperbolic-tangent
  product form of the check-node update.
- `'min-sum'`: the low-complexity approximation, with an optional scaling
  (normalization) factor that recovers most of the sum-product gain.

Both consume channel log-likelihood ratios in the package-wide convention
(`L = log(P(bit=0) / P(bit=1))`, so a **positive** LLR favors bit 0), matching
`Modulator.soft_demodulate`, and both run entirely on vectorized numpy: the
per-iteration check-node update operates on the padded `(m, max_check_degree)`
adjacency grid, and the variable-node update is a single `np.bincount`.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from commpy._channelCoding.ldpc.matrices import TannerGraph

_TANH_CLIP = 1.0 - 1e-12


def _exclude_one_product(grid: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise "product of all other entries": `out[i, j] = prod_{c != j} grid[i, c]`.

    Computed from prefix and suffix cumulative products so a single zero entry
    in a row does not poison the whole row (which naive total/self division
    would).
    """
    _, w = grid.shape
    ones = np.ones((grid.shape[0], 1), dtype=np.float64)
    prefix = np.concatenate([ones, np.cumprod(grid[:, :-1], axis=1)], axis=1)
    rev = np.cumprod(grid[:, ::-1], axis=1)[:, ::-1]  # rev[:, j] = prod grid[:, j:]
    suffix = np.concatenate([rev[:, 1:], ones], axis=1)  # suffix[:, j] = prod grid[:, j+1:]
    return prefix * suffix if w > 1 else np.ones_like(grid)


def _exclude_one_min(grid: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise "min of all other entries": `out[i, j] = min_{c != j} grid[i, c]`."""
    m, w = grid.shape
    if w < 2:
        return np.full_like(grid, np.inf)
    order = np.argsort(grid, axis=1)
    rows = np.arange(m)
    smallest = grid[rows, order[:, 0]]
    second = grid[rows, order[:, 1]]
    out = np.repeat(smallest[:, None], w, axis=1)
    out[rows, order[:, 0]] = second
    return out


def _check_update_sum_product(
    graph: TannerGraph, var_to_check: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Check-node update (sum-product / tanh rule), returning per-edge messages."""
    mask = graph.check_mask
    tanh_half = np.tanh(0.5 * var_to_check)
    grid = np.ones(graph.check_edges.shape, dtype=np.float64)
    grid[mask] = tanh_half[graph.check_edges[mask]]

    excl = np.clip(_exclude_one_product(grid), -_TANH_CLIP, _TANH_CLIP)
    out = np.zeros(var_to_check.shape, dtype=np.float64)
    out[graph.check_edges[mask]] = 2.0 * np.arctanh(excl[mask])
    return out


def _check_update_min_sum(
    graph: TannerGraph, var_to_check: NDArray[np.float64], normalization: float,
) -> NDArray[np.float64]:
    """Check-node update (normalized min-sum), returning per-edge messages."""
    mask = graph.check_mask
    magnitudes = np.full(graph.check_edges.shape, np.inf, dtype=np.float64)
    signs = np.ones(graph.check_edges.shape, dtype=np.float64)
    magnitudes[mask] = np.abs(var_to_check[graph.check_edges[mask]])
    signs[mask] = np.where(var_to_check[graph.check_edges[mask]] >= 0, 1.0, -1.0)

    excl_min = _exclude_one_min(magnitudes)
    excl_sign = np.prod(signs, axis=1, keepdims=True) * signs  # divide out self sign (+/-1)
    grid = normalization * excl_sign * excl_min

    out = np.zeros(var_to_check.shape, dtype=np.float64)
    out[graph.check_edges[mask]] = grid[mask]
    return out


def bp_decode(
    graph: TannerGraph,
    channel_llr: ArrayLike,
    *,
    max_iter: int = 50,
    method: str = 'sum-product',
    normalization: float = 0.75,
) -> tuple[NDArray[np.uint8], NDArray[np.float64], int]:
    """Belief-propagation decode a length-`n` LLR vector on an LDPC Tanner graph.

    Args:
        graph: The code's `TannerGraph` (from `build_tanner`).
        channel_llr: Channel LLRs, length `graph.n`, `L = log(P(0)/P(1))`
            (positive favors bit 0), as produced by `Modulator.soft_demodulate`.
        max_iter: Maximum number of flooding iterations.
        method: `'sum-product'` (exact) or `'min-sum'` (approximate).
        normalization: Scaling factor for `'min-sum'` (ignored otherwise);
            `1.0` is plain min-sum, `~0.75` a common normalized value.

    Returns:
        `(codeword, posterior_llr, iterations)`: the hard-decision codeword
        (`uint8`, length `n`), the final per-bit posterior LLRs, and the number
        of iterations actually run (stops early once `H c^T = 0`).

    Raises:
        ValueError: If `len(channel_llr) != graph.n`, `max_iter < 1`, or
            `method` is not a recognized rule.
    """
    if method not in ('sum-product', 'min-sum'):
        msg = f"method must be 'sum-product' or 'min-sum', got {method!r}."
        raise ValueError(msg)
    if max_iter < 1:
        msg = f'max_iter must be >= 1, got {max_iter}.'
        raise ValueError(msg)
    llr = np.asarray(channel_llr, dtype=np.float64)
    if llr.size != graph.n:
        msg = f'channel_llr must have length {graph.n}, got {llr.size}.'
        raise ValueError(msg)

    edge_var = graph.edge_var
    var_to_check = llr[edge_var].copy()
    posterior = llr.copy()
    codeword = (posterior < 0).astype(np.uint8)
    iterations = 0

    for iteration in range(1, max_iter + 1):
        iterations = iteration
        check_to_var = (
            _check_update_sum_product(graph, var_to_check)
            if method == 'sum-product'
            else _check_update_min_sum(graph, var_to_check, normalization)
        )
        incoming = np.bincount(edge_var, weights=check_to_var, minlength=graph.n)
        posterior = llr + incoming
        codeword = (posterior < 0).astype(np.uint8)
        if not np.any((graph.H @ codeword) % 2):
            break
        var_to_check = posterior[edge_var] - check_to_var

    return codeword, posterior, iterations


__all__ = ['bp_decode']
