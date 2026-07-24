"""Tests for commpy's Monte-Carlo link-level simulation framework."""

import matplotlib as mpl

mpl.use('Agg')  # headless backend so plot_waterfall doesn't require a display

import numpy as np
from scipy.special import erfc

from commpy import Channels, MPSKModulator, plot_waterfall, simulate_ber, simulate_error_rate


def test_simulate_ber_bpsk_awgn_matches_theory(rng):
    mod = MPSKModulator(2)
    result = simulate_ber(
        mod, Channels.awgn, snr_db_range=[4.0, 8.0],
        bits_per_batch=20_000, target_errors=200, max_trials=5_000_000, rng=rng,
    )
    for i, snr_db in enumerate(result.snr_db):
        snr_lin = 10**(snr_db / 10)
        theoretical = 0.5 * erfc(np.sqrt(snr_lin))
        # The true BER should fall within the estimator's own confidence
        # interval (with a small numerical-edge margin), and the point
        # estimate itself should be in the right ballpark.
        assert result.ci_lower[i] - 0.01 <= theoretical <= result.ci_upper[i] + 0.01
        assert abs(result.error_rate[i] - theoretical) < 0.4 * theoretical + 0.005


def test_simulate_ber_result_shapes_match_snr_range(rng):
    mod = MPSKModulator(2)
    snr_range = [0.0, 5.0, 10.0]
    result = simulate_ber(
        mod, Channels.awgn, snr_db_range=snr_range, bits_per_batch=2_000,
        target_errors=20, max_trials=200_000, rng=rng,
    )
    assert result.snr_db.shape == (3,)
    result_arrays = (
        result.error_rate, result.ci_lower, result.ci_upper, result.n_trials, result.n_errors,
    )
    for arr in result_arrays:
        assert arr.shape == (3,)
    assert np.all(result.n_trials > 0)


def test_simulate_error_rate_stops_at_target_errors(rng):
    def always_10_percent(snr_db, trial_rng, n_trials):  # noqa: ARG001 -- signature fixed by simulate_error_rate
        return n_trials // 10, n_trials

    result = simulate_error_rate(
        always_10_percent, snr_db_range=[0.0], target_errors=50,
        max_trials=10_000_000, trials_per_batch=1_000, rng=rng,
    )
    assert result.n_errors[0] >= 50
    # Early stopping should avoid running anywhere near max_trials.
    assert result.n_trials[0] < 100_000


def test_simulate_error_rate_hits_max_trials_when_target_unreachable(rng):
    def zero_errors(snr_db, trial_rng, n_trials):  # noqa: ARG001 -- signature fixed by simulate_error_rate
        return 0, n_trials

    result = simulate_error_rate(
        zero_errors, snr_db_range=[20.0], target_errors=10,
        max_trials=5_000, trials_per_batch=1_000, rng=rng,
    )
    assert result.n_errors[0] == 0
    assert result.n_trials[0] == 5_000
    assert result.error_rate[0] == 0.0


def test_plot_waterfall_runs_without_error(rng):
    mod = MPSKModulator(2)
    result = simulate_ber(
        mod, Channels.awgn, snr_db_range=[2.0, 6.0], bits_per_batch=2_000,
        target_errors=20, max_trials=200_000, rng=rng,
    )
    ax = plot_waterfall(result, theoretical=lambda snr_db: 0.5 * erfc(np.sqrt(10**(snr_db / 10))))
    assert ax is not None
