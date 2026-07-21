"""Tests for commpy.rate_distortion_binary."""

from itertools import pairwise

import pytest

from commpy import binary_entropy, rate_distortion_binary


def test_zero_distortion_costs_full_entropy():
    for p in [0.1, 0.3, 0.5, 0.7]:
        assert rate_distortion_binary(p, 0.0) == pytest.approx(binary_entropy(p))


def test_rate_is_zero_beyond_max_distortion():
    p = 0.3
    d_max = min(p, 1 - p)
    assert rate_distortion_binary(p, d_max) == pytest.approx(0.0, abs=1e-9)
    assert rate_distortion_binary(p, d_max + 0.1) == 0.0
    assert rate_distortion_binary(p, 1.0) == 0.0


def test_rate_is_monotonically_decreasing_in_distortion():
    p = 0.3
    d_max = min(p, 1 - p)
    ds = [0.0, 0.05, 0.1, 0.15, 0.2, d_max]
    rates = [rate_distortion_binary(p, d) for d in ds]
    assert all(r1 >= r2 for r1, r2 in pairwise(rates))


def test_fair_source_rate_distortion():
    # p=0.5: R(D) = 1 - H_b(D) for 0 <= D <= 0.5, the textbook result.
    assert rate_distortion_binary(0.5, 0.0) == pytest.approx(1.0)
    assert rate_distortion_binary(0.5, 0.11) == pytest.approx(1 - binary_entropy(0.11))
    assert rate_distortion_binary(0.5, 0.5) == pytest.approx(0.0, abs=1e-9)


def test_rejects_invalid_p():
    with pytest.raises(ValueError, match='p must be'):
        rate_distortion_binary(1.5, 0.1)


def test_rejects_negative_distortion():
    with pytest.raises(ValueError, match='distortion'):
        rate_distortion_binary(0.5, -0.1)
