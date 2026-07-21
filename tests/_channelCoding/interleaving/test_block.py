"""Tests for commpy.BlockInterleaver."""

import numpy as np
import pytest

from commpy import BlockInterleaver


def test_round_trip(rng):
    il = BlockInterleaver(rows=4, cols=6)
    data = rng.integers(0, 2, 24)
    interleaved = il.interleave(data)
    recovered = il.deinterleave(interleaved)
    np.testing.assert_array_equal(recovered, data)


def test_interleave_actually_reorders():
    il = BlockInterleaver(rows=3, cols=3)
    data = np.arange(9)
    interleaved = il.interleave(data)
    assert not np.array_equal(interleaved, data)
    assert sorted(interleaved.tolist()) == sorted(data.tolist())


def test_known_small_example():
    # Write rows: [[0,1,2],[3,4,5]]; read columns: 0,3,1,4,2,5.
    il = BlockInterleaver(rows=2, cols=3)
    data = np.array([0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(il.interleave(data), [0, 3, 1, 4, 2, 5])


def test_rejects_wrong_length():
    il = BlockInterleaver(rows=4, cols=6)
    with pytest.raises(ValueError, match='length'):
        il.interleave(np.zeros(10))
    with pytest.raises(ValueError, match='length'):
        il.deinterleave(np.zeros(10))


def test_burst_error_is_spread_out():
    # A contiguous burst error in the interleaved stream should map back to
    # scattered (non-contiguous) positions in the original data order --
    # the whole point of interleaving against burst-error channels.
    il = BlockInterleaver(rows=5, cols=5)
    data = np.arange(25)
    interleaved = il.interleave(data)
    corrupted = interleaved.copy()
    corrupted[10:15] = -1  # a contiguous burst of 5 erasures
    deinterleaved = il.deinterleave(corrupted)
    erased_original_positions = np.where(deinterleaved == -1)[0]
    # Erased positions should be spread out, not clustered together.
    assert len(erased_original_positions) == 5
    assert np.max(np.diff(sorted(erased_original_positions))) > 1
