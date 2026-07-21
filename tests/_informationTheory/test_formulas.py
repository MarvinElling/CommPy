"""Tests for commpy information-theory formulas: entropy, mutual information, channel capacity."""

import math
from itertools import pairwise

import numpy as np
import pytest

from commpy import (
    binary_entropy,
    channel_capacity_awgn,
    channel_capacity_bsc,
    channel_capacity_dmc,
    mutual_information,
    shannon_entropy,
)


def test_fair_coin_has_one_bit_of_entropy():
    assert shannon_entropy([0.5, 0.5]) == pytest.approx(1.0)


def test_certain_outcome_has_zero_entropy():
    assert shannon_entropy([1.0, 0.0]) == pytest.approx(0.0)


def test_uniform_distribution_over_n_outcomes_is_log2_n():
    n = 8
    probs = [1.0 / n] * n
    assert shannon_entropy(probs) == pytest.approx(math.log2(n))


def test_zero_probabilities_are_ignored_not_nan():
    # 0 * log2(0) is conventionally treated as 0, not NaN/-inf.
    assert shannon_entropy([0.5, 0.5, 0.0]) == pytest.approx(1.0)


def test_binary_entropy_matches_shannon_entropy():
    for p in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        assert binary_entropy(p) == pytest.approx(shannon_entropy([p, 1 - p]))


def test_binary_entropy_boundary_is_zero():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0


def test_mutual_information_independent_variables_is_zero():
    # p(x,y) = p(x)*p(y) for independent X, Y -> I(X;Y) = 0.
    p_x = np.array([0.3, 0.7])
    p_y = np.array([0.6, 0.4])
    joint = np.outer(p_x, p_y)
    assert mutual_information(joint) == pytest.approx(0.0, abs=1e-9)


def test_mutual_information_of_perfectly_correlated_variables_is_entropy():
    # X == Y deterministically -> I(X;Y) = H(X).
    joint = np.diag([0.5, 0.3, 0.2])
    assert mutual_information(joint) == pytest.approx(shannon_entropy([0.5, 0.3, 0.2]))


def test_channel_capacity_bsc_perfect_and_useless_channel():
    assert channel_capacity_bsc(0.0) == pytest.approx(1.0)
    # An always-flip channel is still deterministic (hence perfect), just relabeled.
    assert channel_capacity_bsc(1.0) == pytest.approx(1.0)
    assert channel_capacity_bsc(0.5) == pytest.approx(0.0)  # pure noise


def test_channel_capacity_bsc_matches_one_minus_binary_entropy():
    for p in [0.05, 0.1, 0.2, 0.3, 0.4]:
        assert channel_capacity_bsc(p) == pytest.approx(1 - binary_entropy(p))


def test_channel_capacity_awgn_zero_snr_is_zero():
    assert channel_capacity_awgn(0.0) == pytest.approx(0.0)


def test_channel_capacity_awgn_increases_with_snr():
    caps = [channel_capacity_awgn(snr) for snr in [0.1, 1.0, 10.0, 100.0]]
    assert all(c2 > c1 for c1, c2 in pairwise(caps))


def test_channel_capacity_dmc_matches_closed_form_bsc():
    for p in [0.0, 0.05, 0.1, 0.3, 0.5]:
        q = np.array([[1 - p, p], [p, 1 - p]])
        capacity, p_opt = channel_capacity_dmc(q)
        assert capacity == pytest.approx(channel_capacity_bsc(p), abs=1e-6)
        np.testing.assert_allclose(p_opt, [0.5, 0.5], atol=1e-4)


def test_channel_capacity_dmc_matches_brute_force_on_asymmetric_channel():
    # Z-channel: input 0 is noiseless; input 1 flips to 0 with probability p.
    p = 0.3
    q = np.array([[1.0, 0.0], [p, 1 - p]])
    capacity, p_opt = channel_capacity_dmc(q)

    grid = np.linspace(0, 1, 2001)
    best = 0.0
    for p0 in grid:
        joint = np.array([p0, 1 - p0])[:, None] * q
        mi = mutual_information(joint)
        best = max(best, mi)
    assert capacity == pytest.approx(best, abs=1e-3)
    assert not np.allclose(p_opt, [0.5, 0.5], atol=1e-2)  # optimal input is NOT uniform here


def test_channel_capacity_dmc_noiseless_identity_channel_is_log2_n():
    n = 4
    q = np.eye(n)
    capacity, _ = channel_capacity_dmc(q)
    assert capacity == pytest.approx(math.log2(n))
