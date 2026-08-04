"""Tests for commpy's system- and channel-level plots."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from commpy import (
    Channels,
    MQAMModulator,
    OFDMModulator,
    mmse_equalizer,
    plot_capacity_curves,
    plot_channel_response,
    plot_equalizer_response,
    plot_error_rate_comparison,
    plot_mimo_capacity_cdf,
    plot_ofdm_grid,
    plot_papr_ccdf,
    simulate_ber,
    zf_equalizer,
)

TAPS = np.array([1.0, 0.35, -0.18, 0.07])


def legend_labels(ax):
    legend = ax.get_legend()
    assert legend is not None
    return [text.get_text() for text in legend.get_texts()]


def image_array(ax, index=0):
    array = ax.images[index].get_array()
    assert array is not None
    return np.asarray(array)


@pytest.fixture
def results(rng):
    def run(order):
        modulator = MQAMModulator(order)
        return simulate_ber(
            modulator, lambda x, snr, trial_rng: Channels.awgn(x, snr, rng=trial_rng),
            np.arange(0.0, 7.0, 3.0), bits_per_batch=2000, target_errors=20, rng=rng,
        )

    return {'QPSK': run(4), '16-QAM': run(16)}


class TestErrorRateComparison:
    def test_draws_one_labeled_curve_per_result(self, results):
        ax = plot_error_rate_comparison(results)
        assert legend_labels(ax) == ['QPSK', '16-QAM']

    def test_keeps_the_log_scale_and_confidence_intervals(self, results):
        ax = plot_error_rate_comparison(results)
        assert ax.get_yscale() == 'log'
        # One ErrorbarContainer per curve, each carrying its interval bars.
        assert len(ax.containers) == 2

    def test_color_follows_the_label_not_the_position(self, results):
        full = plot_error_rate_comparison(results)
        subset = plot_error_rate_comparison({'QPSK': results['QPSK']})
        assert full.containers[0][0].get_color() == subset.containers[0][0].get_color()

    def test_overlays_theoretical_references(self, results):
        theoretical = {'QPSK': np.full(results['QPSK'].snr_db.size, 0.01)}
        ax = plot_error_rate_comparison(results, theoretical=theoretical)
        assert any(line.get_linestyle() == '--' for line in ax.lines)

    def test_rejects_an_empty_comparison(self):
        with pytest.raises(ValueError, match='at least one curve'):
            plot_error_rate_comparison({})


class TestChannelAndEqualizer:
    def test_channel_response_uses_two_panels(self):
        ax = plot_channel_response(TAPS)
        assert len(ax.figure.axes) == 2
        assert ax.get_title() == 'Channel impulse response'

    def test_channel_response_into_a_supplied_axes_draws_the_spectrum(self):
        _, ax = plt.subplots()
        assert plot_channel_response(TAPS, ax=ax) is ax
        assert len(ax.figure.axes) == 1

    def test_equalizer_plot_shows_channel_equalizer_and_combined(self):
        ax = plot_equalizer_response(TAPS, zf_equalizer(TAPS, 15))
        assert legend_labels(ax) == ['channel', 'equalizer', 'combined']

    def test_a_longer_equalizer_leaves_less_residual_isi(self):
        def residual(n_taps):
            ax = plot_equalizer_response(TAPS, mmse_equalizer(TAPS, n_taps, 0.001))
            return float(ax.get_title().split('residual ISI ')[1].rstrip('%)'))

        assert residual(21) < residual(3)


class TestOFDM:
    def test_grid_masks_inactive_subcarriers(self):
        modulator = OFDMModulator(64, 16, active_subcarriers=np.arange(1, 27))
        grid = np.ones((10, 64), dtype=np.complex128)
        drawn = image_array(plot_ofdm_grid(grid, active_subcarriers=modulator.active_subcarriers))
        assert np.all(np.isnan(drawn[0]))  # subcarrier 0 is a guard
        assert not np.any(np.isnan(drawn[1:27]))

    def test_grid_is_oriented_with_subcarriers_on_the_y_axis(self):
        assert image_array(plot_ofdm_grid(np.ones((10, 64)))).shape == (64, 10)

    def test_papr_ccdf_is_monotonically_decreasing(self, rng):
        symbols = np.fft.ifft(
            rng.standard_normal((200, 64)) + 1j * rng.standard_normal((200, 64)), axis=1,
        )
        ax = plot_papr_ccdf(symbols)
        ccdf = np.asarray(ax.lines[0].get_ydata())
        assert np.all(np.diff(ccdf) <= 0)
        assert ax.get_yscale() == 'log'


class TestCapacity:
    def test_mimo_cdf_rises_monotonically_from_zero_to_one(self, rng):
        ax = plot_mimo_capacity_cdf(2, 2, 10.0, n_realizations=200, rng=rng)
        probabilities = np.asarray(ax.lines[0].get_ydata())
        capacities = np.asarray(ax.lines[0].get_xdata())
        assert np.all(np.diff(probabilities) > 0)
        assert np.all(np.diff(capacities) >= 0)

    def test_more_snr_shifts_the_whole_capacity_distribution_right(self, rng):
        ax = plot_mimo_capacity_cdf(2, 2, [0.0, 15.0], n_realizations=200, rng=rng)
        low, high = (np.asarray(line.get_xdata()) for line in ax.lines)
        assert high.min() > low.max()

    def test_single_snr_needs_no_legend(self, rng):
        assert plot_mimo_capacity_cdf(2, 2, 5.0, n_realizations=50, rng=rng).get_legend() is None

    def test_soft_decision_capacity_dominates_hard_decision(self):
        ax = plot_capacity_curves()
        awgn = np.asarray(ax.lines[0].get_ydata())
        bsc = np.asarray(ax.lines[1].get_ydata())
        # Hard decisions throw information away, so the BSC curve can never win.
        assert np.all(awgn >= bsc - 1e-9)

    def test_capacity_curves_share_one_x_axis(self):
        ax = plot_capacity_curves()
        assert np.array_equal(
            np.asarray(ax.lines[0].get_xdata()), np.asarray(ax.lines[1].get_xdata()),
        )
