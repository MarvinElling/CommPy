"""Tests for commpy's signal- and modulation-domain plots."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from commpy import (
    Channels,
    MQAMModulator,
    plot_constellation,
    plot_eye_diagram,
    plot_filter_response,
    plot_iq_time,
    plot_psd,
    plot_spectrogram,
    root_raised_cosine_filter,
)

SPS = 8


# matplotlib's getters are typed as returning ArrayLike; these unwrap them so
# the assertions below can index and inspect shapes.
def offsets(collection):
    return np.asarray(collection.get_offsets())


def xdata(line):
    return np.asarray(line.get_xdata())


def ydata(line):
    return np.asarray(line.get_ydata())


def segments(collection):
    assert isinstance(collection, LineCollection)
    return collection.get_segments()


def legend_labels(ax):
    legend = ax.get_legend()
    assert legend is not None
    return [text.get_text() for text in legend.get_texts()]


@pytest.fixture
def modulator():
    return MQAMModulator(16)


@pytest.fixture
def symbols(modulator, rng):
    return modulator.modulate(rng.integers(0, 2, 800))


@pytest.fixture
def shaped(symbols):
    taps = root_raised_cosine_filter(1.0, 0.35, 6)(np.arange(0.0, 6.0, 1.0 / SPS))
    upsampled = np.zeros(symbols.size * SPS, dtype=np.complex128)
    upsampled[::SPS] = symbols
    return np.convolve(upsampled, taps, mode='same')


class TestConstellation:
    def test_draws_reference_points_from_a_modulator(self, modulator):
        ax = plot_constellation(modulator)
        # One PathCollection holding all M points.
        assert len(ax.collections) == 1
        assert offsets(ax.collections[0]).shape == (16, 2)

    def test_received_cloud_adds_a_second_series_and_a_legend(self, modulator, symbols, rng):
        received = Channels.awgn(symbols, 15.0, rng=rng)
        ax = plot_constellation(modulator, received=received)
        assert len(ax.collections) == 2
        assert legend_labels(ax) == ['received', 'constellation']

    def test_no_legend_for_a_single_series(self, modulator):
        assert plot_constellation(modulator).get_legend() is None

    def test_labels_annotate_every_point_with_its_bits(self, modulator):
        ax = plot_constellation(modulator, labels=True)
        texts = {t.get_text() for t in ax.texts}
        assert len(texts) == 16
        assert all(len(t) == modulator.bits_per_symbol for t in texts)

    def test_regions_draw_the_decision_boundaries(self, modulator):
        ax = plot_constellation(modulator, regions=True)
        assert len(ax.images) == 1

    def test_accepts_raw_reference_points(self):
        ax = plot_constellation(np.array([1 + 0j, -1 + 0j]))
        assert offsets(ax.collections[0]).shape == (2, 2)

    def test_labels_need_a_modulator(self):
        with pytest.raises(TypeError, match='bit labels are unknown'):
            plot_constellation(np.array([1 + 0j, -1 + 0j]), labels=True)

    def test_axes_are_square_so_the_geometry_is_not_distorted(self, modulator):
        assert plot_constellation(modulator).get_aspect() == 1.0


class TestEyeDiagram:
    def test_traces_span_exactly_two_symbol_periods(self, shaped):
        ax = plot_eye_diagram(shaped, SPS)
        assert ax.get_xlim() == (0.0, 2.0)
        traces = segments(ax.collections[0])
        assert traces[0][-1][0] == pytest.approx(2.0)

    def test_complex_input_draws_i_and_q_separately(self, shaped):
        ax = plot_eye_diagram(shaped, SPS)
        collections = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(collections) == 2
        assert legend_labels(ax) == ['I', 'Q']

    def test_real_input_draws_one_series_without_a_legend(self, shaped):
        ax = plot_eye_diagram(shaped.real, SPS)
        assert len([c for c in ax.collections if isinstance(c, LineCollection)]) == 1
        assert ax.get_legend() is None

    def test_n_traces_caps_the_number_of_overlaid_windows(self, shaped):
        ax = plot_eye_diagram(shaped, SPS, n_traces=17)
        assert len(segments(ax.collections[0])) == 17

    def test_rejects_sps_below_two(self, shaped):
        with pytest.raises(ValueError, match='at least 2'):
            plot_eye_diagram(shaped, 1)

    def test_rejects_a_signal_too_short_for_one_window(self):
        with pytest.raises(ValueError, match='too short'):
            plot_eye_diagram(np.zeros(4), SPS)


class TestSpectra:
    def test_psd_is_two_sided_and_centered_on_dc(self, shaped):
        ax = plot_psd(shaped, fs=float(SPS))
        freqs = xdata(ax.lines[0])
        assert freqs.min() < 0 < freqs.max()
        assert np.all(np.diff(freqs) > 0)

    def test_psd_peaks_at_the_occupied_band(self, shaped):
        ax = plot_psd(shaped, fs=float(SPS))
        freqs, power = xdata(ax.lines[0]), ydata(ax.lines[0])
        # An RRC-shaped baseband signal concentrates its power around DC.
        assert abs(freqs[np.argmax(power)]) < 1.0

    def test_spectrogram_carries_a_colorbar_for_its_magnitude_scale(self, shaped):
        ax = plot_spectrogram(shaped, fs=float(SPS))
        assert len(ax.images) == 1
        assert len(ax.figure.axes) == 2  # data panel + colorbar


class TestIQTime:
    def test_complex_input_draws_i_and_q(self, shaped):
        ax = plot_iq_time(shaped, fs=float(SPS))
        assert [line.get_label() for line in ax.lines] == ['I', 'Q']

    def test_time_axis_is_scaled_by_the_sampling_rate(self, shaped):
        ax = plot_iq_time(shaped, fs=float(SPS))
        assert xdata(ax.lines[0])[-1] == pytest.approx((shaped.size - 1) / SPS)

    def test_real_input_draws_one_unlabeled_series(self, shaped):
        assert plot_iq_time(shaped.real).get_legend() is None


class TestFilterResponse:
    def test_both_domains_use_two_stacked_panels_not_a_second_y_axis(self):
        ax = plot_filter_response(np.array([1.0, 0.5, 0.2]), fs=2.0)
        assert len(ax.figure.axes) == 2
        assert ax.get_title() == 'Impulse response'

    def test_single_domain_draws_into_a_supplied_axes(self):
        _, ax = plt.subplots()
        returned = plot_filter_response(np.array([1.0, 0.5]), fs=2.0, domain='freq', ax=ax)
        assert returned is ax
        assert len(ax.figure.axes) == 1

    def test_magnitude_response_is_normalized_to_its_peak(self):
        ax = plot_filter_response(np.array([1.0, 0.5, 0.2]), fs=2.0, domain='freq')
        assert ydata(ax.lines[0]).max() == pytest.approx(0.0)

    def test_accepts_a_pulse_shape_callable(self):
        pulse = root_raised_cosine_filter(1.0, 0.35, 6)
        ax = plot_filter_response(pulse, fs=8.0, symbol_period=1.0, span=6, domain='time')
        assert xdata(ax.lines[0]).size == 48

    def test_callable_without_its_sampling_parameters_is_rejected(self):
        pulse = root_raised_cosine_filter(1.0, 0.35, 6)
        with pytest.raises(ValueError, match='symbol_period and span'):
            plot_filter_response(pulse, fs=8.0, domain='time')

    def test_both_domains_cannot_reuse_one_supplied_axes(self):
        _, ax = plt.subplots()
        with pytest.raises(ValueError, match='cannot reuse'):
            plot_filter_response(np.array([1.0]), ax=ax)

    def test_rejects_an_unknown_domain(self):
        with pytest.raises(ValueError, match='domain must be'):
            plot_filter_response(np.array([1.0]), domain='phase')


@pytest.mark.parametrize(
    ('plot_fn', 'args'),
    [
        (plot_psd, ()),
        (plot_spectrogram, ()),
        (plot_iq_time, ()),
    ],
)
def test_supplied_axes_is_reused_rather_than_replaced(plot_fn, args, shaped):
    _, ax = plt.subplots()
    n_figures = len(plt.get_fignums())
    assert plot_fn(shaped, float(SPS), *args, ax=ax) is ax
    assert len(plt.get_fignums()) == n_figures


def test_every_plot_returns_an_axes(modulator, shaped):
    results = [
        plot_constellation(modulator),
        plot_eye_diagram(shaped, SPS),
        plot_psd(shaped, float(SPS)),
        plot_spectrogram(shaped, float(SPS)),
        plot_iq_time(shaped, float(SPS)),
        plot_filter_response(np.array([1.0, 0.5]), fs=2.0),
    ]
    assert all(isinstance(result, Axes) for result in results)
