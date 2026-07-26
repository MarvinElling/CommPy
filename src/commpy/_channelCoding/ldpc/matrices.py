"""Construction of low-density parity-check (LDPC) matrices and derived structures.

An LDPC code is the null space over GF(2) of a *sparse* parity-check matrix
`H` of shape `(m, n)`: the length-`n` binary vectors `c` with `H c^T = 0`
(mod 2). This module provides the pieces the `LDPCCode` class ties together:

- `make_gallager` builds a regular `(w_c, w_r)` parity-check matrix from
  Gallager's classic ensemble (constant column weight `w_c`, row weight `w_r`).
- `qc_lift` expands a small protograph "base matrix" into a quasi-cyclic `H`
  by replacing each entry with a circulant-shifted identity (or a zero block)
  -- the construction style used by 5G-NR / 802.11n LDPC codes.
- `parity_check_to_generator` reduces `H` to a systematic generator matrix `G`
  via Gauss-Jordan elimination over GF(2), so encoding is a single mat-vec.
- `build_tanner` precomputes the check-node adjacency the belief-propagation
  decoder walks, padded to a dense `(m, max_check_degree)` layout so the whole
  check-node update vectorizes.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def make_gallager(
    n: int, w_c: int, w_r: int, *, rng: np.random.Generator | None = None,
) -> NDArray[np.uint8]:
    """Build a regular `(w_c, w_r)` LDPC parity-check matrix (Gallager ensemble).

    The matrix is a vertical stack of `w_c` sub-bands. The first band tiles the
    `n` columns into `n // w_r` rows of `w_r` consecutive ones; each subsequent
    band is a random column permutation of the first. Every column therefore
    has exactly `w_c` ones and every row exactly `w_r`, giving code length `n`,
    `m = n * w_c // w_r` parity checks, and design rate `1 - w_c / w_r`.

    Args:
        n: Codeword length; must be a multiple of `w_r`.
        w_c: Column weight (checks each variable participates in); `>= 1`.
        w_r: Row weight (variables each check connects); `>= 1`.
        rng: Optional `np.random.Generator` for reproducibility.

    Returns:
        The `(m, n)` parity-check matrix as a `uint8` array of 0/1.

    Raises:
        ValueError: If `w_c < 1`, `w_r < 1`, or `n` is not a multiple of `w_r`.
    """
    if w_c < 1 or w_r < 1:
        msg = 'w_c and w_r must both be >= 1.'
        raise ValueError(msg)
    if n % w_r != 0:
        msg = f'n must be a multiple of w_r={w_r}, got n={n}.'
        raise ValueError(msg)
    if rng is None:
        rng = np.random.default_rng()

    sub_m = n // w_r
    base = np.zeros((sub_m, n), dtype=np.uint8)
    for r in range(sub_m):
        base[r, r * w_r:(r + 1) * w_r] = 1

    bands = [base]
    bands.extend(base[:, rng.permutation(n)] for _ in range(w_c - 1))
    return np.vstack(bands)


def qc_lift(base_matrix: ArrayLike, z: int) -> NDArray[np.uint8]:
    """Expand a protograph base matrix into a quasi-cyclic parity-check matrix.

    Each entry of the `(mb, nb)` base matrix is replaced by a `z x z` block:
    a value `s >= 0` becomes the identity cyclically shifted right by `s`
    columns; the sentinel `-1` becomes an all-zero block. The result has shape
    `(mb * z, nb * z)`.

    Args:
        base_matrix: Integer base matrix; entries are circulant shifts in
            `[0, z)` or `-1` for a zero block.
        z: Lifting (expansion) factor; `>= 1`.

    Returns:
        The lifted `(mb * z, nb * z)` parity-check matrix as `uint8`.

    Raises:
        ValueError: If `z < 1` or any entry is `< -1` or `>= z`.
    """
    if z < 1:
        msg = f'z must be >= 1, got {z}.'
        raise ValueError(msg)
    base = np.asarray(base_matrix, dtype=np.int64)
    if np.any(base < -1) or np.any(base >= z):
        msg = f'base_matrix entries must be -1 (zero block) or in [0, z={z}).'
        raise ValueError(msg)

    mb, nb = base.shape
    identity = np.eye(z, dtype=np.uint8)
    H = np.zeros((mb * z, nb * z), dtype=np.uint8)
    for i in range(mb):
        for j in range(nb):
            shift = int(base[i, j])
            if shift >= 0:
                H[i * z:(i + 1) * z, j * z:(j + 1) * z] = np.roll(identity, shift, axis=1)
    return H


def parity_check_to_generator(
    H: ArrayLike,
) -> tuple[NDArray[np.uint8], NDArray[np.int64], NDArray[np.int64], int]:
    """Reduce a parity-check matrix to a systematic generator matrix over GF(2).

    Runs Gauss-Jordan elimination on `H` to reduced row echelon form. The pivot
    columns become parity positions; the remaining `k = n - rank(H)` columns
    become information positions. Each returned generator row is a valid
    codeword (`H c^T = 0`) that carries a single information bit, so
    `message @ G % 2` is a systematic encoder.

    Args:
        H: Parity-check matrix, shape `(m, n)`, 0/1 valued.

    Returns:
        `(G, info_positions, parity_positions, rank)`: the `(k, n)` generator
        matrix, the (sorted) information-column indices, the pivot/parity-column
        indices, and `rank(H)` (redundant rows are handled, so `k = n - rank`).
    """
    M = (np.asarray(H, dtype=np.uint8) % 2).copy()
    m, n = M.shape

    pivot_cols: list[int] = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        pivots = np.nonzero(M[row:, col])[0]
        if pivots.size == 0:
            continue
        pivot = row + int(pivots[0])
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
        mask = M[:, col].astype(bool).copy()
        mask[row] = False
        M[mask] ^= M[row]
        pivot_cols.append(col)
        row += 1

    rank = row
    parity_positions = np.array(pivot_cols, dtype=np.int64)
    is_pivot = np.zeros(n, dtype=bool)
    is_pivot[parity_positions] = True
    info_positions = np.nonzero(~is_pivot)[0].astype(np.int64)
    k = int(info_positions.size)

    G = np.zeros((k, n), dtype=np.uint8)
    G[np.arange(k), info_positions] = 1
    # For pivot row i (pivot column parity_positions[i]) the RREF encodes
    # c[parity_positions[i]] = sum_j M[i, info_positions[j]] * c[info_positions[j]].
    G[:, parity_positions] = M[:rank][:, info_positions].T
    return G, info_positions, parity_positions, rank


@dataclass(frozen=True)
class TannerGraph:
    """Precomputed check-node adjacency of an LDPC Tanner graph.

    Edges (the ones of `H`) are numbered in row-major order. `check_edges` maps
    each check node to the ids of its incident edges, padded with `-1` to the
    maximum check degree; `check_mask` marks the valid (non-padding) entries.
    `edge_var` gives the variable node each edge touches, so the variable-node
    update reduces to a single `np.bincount`.
    """

    n: int
    m: int
    edge_var: NDArray[np.int64]
    check_edges: NDArray[np.int64]
    check_mask: NDArray[np.bool_]
    H: NDArray[np.uint8]


def build_tanner(H: ArrayLike) -> TannerGraph:
    """Precompute the check-node adjacency used by belief propagation.

    Args:
        H: Parity-check matrix, shape `(m, n)`, 0/1 valued.

    Returns:
        A `TannerGraph` with edges numbered in row-major order and check-node
        incidence padded to a dense `(m, max_check_degree)` layout.
    """
    Hm = np.asarray(H, dtype=np.uint8) % 2
    m, n = Hm.shape
    check_idx, var_idx = np.nonzero(Hm)  # row-major: edges sorted by check node
    n_edges = int(check_idx.size)
    edge_ids = np.arange(n_edges, dtype=np.int64)

    check_degree = Hm.sum(axis=1, dtype=np.int64)
    max_dc = int(check_degree.max()) if m > 0 else 0
    group_start = np.zeros(m, dtype=np.int64)
    if m > 0:
        group_start[1:] = np.cumsum(check_degree)[:-1]
    pos_in_check = edge_ids - group_start[check_idx]

    check_edges = np.full((m, max_dc), -1, dtype=np.int64)
    check_mask = np.zeros((m, max_dc), dtype=bool)
    check_edges[check_idx, pos_in_check] = edge_ids
    check_mask[check_idx, pos_in_check] = True

    return TannerGraph(
        n=n, m=m, edge_var=var_idx.astype(np.int64),
        check_edges=check_edges, check_mask=check_mask, H=Hm,
    )


__all__ = [
    'TannerGraph',
    'build_tanner',
    'make_gallager',
    'parity_check_to_generator',
    'qc_lift',
]
