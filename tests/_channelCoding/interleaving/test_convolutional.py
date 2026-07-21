"""Tests for commpy.ConvolutionalInterleaver.

Unlike the block interleaver, this is a streaming transform with a fixed
end-to-end latency of `n_lanes * (n_lanes - 1) * delay_increment` symbols
(see the module docstring for the derivation): the first `total_delay`
outputs of interleave+deinterleave include lane zero-fill, so round-trip
checks compare the steady-state region only.
"""

import numpy as np
import pytest

from commpy import ConvolutionalInterleaver


def test_round_trip_steady_state():
    n_lanes, delay = 4, 3
    il = ConvolutionalInterleaver(n_lanes, delay, is_deinterleaver=False)
    dil = ConvolutionalInterleaver(n_lanes, delay, is_deinterleaver=True)
    assert il.total_delay == n_lanes * (n_lanes - 1) * delay

    rng = np.random.default_rng(0)
    n = 200
    data = 1 + rng.integers(0, 255, n)  # never 0, so any 0 in output is unambiguous padding

    interleaved = il.process(data)
    recovered = dil.process(interleaved)

    total_delay = il.total_delay
    np.testing.assert_array_equal(recovered[total_delay:], data[:n - total_delay])


def test_interleaver_actually_reorders():
    il = ConvolutionalInterleaver(4, 3)
    data = np.arange(50)
    interleaved = il.process(data)
    assert not np.array_equal(interleaved[:20], data[:20])


def test_single_lane_is_identity():
    # With only 1 lane there's nothing to interleave against.
    il = ConvolutionalInterleaver(1, 5)
    data = np.arange(20)
    np.testing.assert_array_equal(il.process(data), data)


def test_zero_delay_increment_is_identity():
    il = ConvolutionalInterleaver(4, 0)
    data = np.arange(20)
    np.testing.assert_array_equal(il.process(data), data)


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError, match='n_lanes'):
        ConvolutionalInterleaver(0, 3)
    with pytest.raises(ValueError, match='delay_increment'):
        ConvolutionalInterleaver(4, -1)


def test_burst_error_is_spread_out():
    n_lanes, delay = 4, 4
    il = ConvolutionalInterleaver(n_lanes, delay, is_deinterleaver=False)
    dil = ConvolutionalInterleaver(n_lanes, delay, is_deinterleaver=True)
    total_delay = il.total_delay

    n = 200
    data = np.arange(1, n + 1)  # never 0
    interleaved = il.process(data)
    corrupted = interleaved.copy()
    corrupted[100:104] = 0  # contiguous burst of erasures, marked with the unused value 0
    recovered = dil.process(corrupted)

    steady = recovered[total_delay:]
    erased_positions = np.where(steady == 0)[0]
    assert len(erased_positions) == 4
    assert np.max(np.diff(sorted(erased_positions))) > 1
