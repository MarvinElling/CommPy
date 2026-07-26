"""Tests for LDPC matrix construction, systematic reduction, and Tanner graph.

Correctness of `parity_check_to_generator` is pinned down structurally: every
generator row must lie in the null space of `H` (a valid codeword), the
information/parity column partition must cover all columns exactly once, and
the reported rank must match an independent GF(2) rank. This is the
gold-standard check for a linear-code generator -- stronger than inspecting the
elimination by hand.
"""

import numpy as np
import pytest

from commpy import HammingCode
from commpy._channelCoding.ldpc.matrices import (
    build_tanner,
    make_gallager,
    parity_check_to_generator,
    qc_lift,
)


def _gf2_rank(matrix):
    M = (np.asarray(matrix, dtype=np.uint8) % 2).copy()
    rows, cols = M.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if M[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for r in range(rows):
            if r != rank and M[r, col]:
                M[r] ^= M[rank]
        rank += 1
    return rank


@pytest.mark.parametrize(('n', 'w_c', 'w_r'), [(12, 2, 3), (48, 3, 6), (20, 2, 4)])
def test_gallager_has_exact_column_and_row_weights(n, w_c, w_r, rng):
    H = make_gallager(n, w_c, w_r, rng=rng)
    assert H.shape == (n * w_c // w_r, n)
    assert H.dtype == np.uint8
    np.testing.assert_array_equal(H.sum(axis=0), np.full(n, w_c))  # every column weight w_c
    np.testing.assert_array_equal(H.sum(axis=1), np.full(H.shape[0], w_r))  # every row weight w_r


def test_gallager_rejects_bad_dimensions():
    with pytest.raises(ValueError, match='multiple of w_r'):
        make_gallager(10, 2, 3)
    with pytest.raises(ValueError, match='>= 1'):
        make_gallager(12, 0, 3)


def test_qc_lift_shape_and_blocks():
    base = np.array([[0, -1], [1, 2]])
    z = 4
    H = qc_lift(base, z)
    assert H.shape == (2 * z, 2 * z)
    # -1 entry -> all-zero block
    np.testing.assert_array_equal(H[0:z, z:2 * z], np.zeros((z, z), dtype=np.uint8))
    # shift 0 -> identity
    np.testing.assert_array_equal(H[0:z, 0:z], np.eye(z, dtype=np.uint8))
    # shift 1 -> identity rolled right by one column
    np.testing.assert_array_equal(H[z:2 * z, 0:z], np.roll(np.eye(z, dtype=np.uint8), 1, axis=1))


def test_qc_lift_rejects_bad_input():
    with pytest.raises(ValueError, match='z must be'):
        qc_lift([[0]], 0)
    with pytest.raises(ValueError, match='zero block'):
        qc_lift([[5]], 4)  # shift 5 out of range for z=4


@pytest.mark.parametrize(('n', 'w_c', 'w_r'), [(12, 2, 3), (48, 3, 6)])
def test_generator_rows_are_codewords(n, w_c, w_r, rng):
    H = make_gallager(n, w_c, w_r, rng=rng)
    G, info_positions, parity_positions, rank = parity_check_to_generator(H)

    # Every generator row is a valid codeword: H G^T = 0 over GF(2).
    assert np.all((H @ G.T) % 2 == 0)
    # Rank matches an independent computation, and k = n - rank.
    assert rank == _gf2_rank(H)
    assert G.shape == (n - rank, n)
    # Information / parity columns partition all n columns exactly once.
    assert info_positions.size + parity_positions.size == n
    assert set(info_positions.tolist()) | set(parity_positions.tolist()) == set(range(n))
    assert set(info_positions.tolist()) & set(parity_positions.tolist()) == set()
    # Systematic: the info columns of G form the identity.
    np.testing.assert_array_equal(G[:, info_positions], np.eye(n - rank, dtype=np.uint8))


def test_generator_matches_hamming_reference():
    # A Hamming code is a known linear code; its H must reduce to the same
    # dimension (k = n - m) and yield a valid generator.
    ham = HammingCode(3)  # (7, 4)
    G, info_positions, _, rank = parity_check_to_generator(ham.H)
    assert rank == 3
    assert G.shape == (4, 7)
    assert info_positions.size == 4
    assert np.all((ham.H @ G.T) % 2 == 0)


def test_build_tanner_matches_parity_matrix(rng):
    H = make_gallager(12, 2, 3, rng=rng)
    graph = build_tanner(H)

    assert graph.n == H.shape[1]
    assert graph.m == H.shape[0]
    assert graph.edge_var.size == int(H.sum())  # one edge per 1 in H
    # Each check's valid-edge count equals its row weight.
    np.testing.assert_array_equal(graph.check_mask.sum(axis=1), H.sum(axis=1))
    # The edges of each check node point back to exactly that row's ones.
    for c in range(graph.m):
        edge_ids = graph.check_edges[c][graph.check_mask[c]]
        variables = np.sort(graph.edge_var[edge_ids])
        np.testing.assert_array_equal(variables, np.nonzero(H[c])[0])
