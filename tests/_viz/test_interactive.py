"""Tests for commpy's optional Plotly plotting backend."""

import numpy as np
import pytest

from commpy import (
    Channels,
    LDPCCode,
    MQAMModulator,
    plotly_constellation,
    plotly_eye_diagram,
    plotly_psd,
    plotly_tanner_graph,
    plotly_waterfall,
    root_raised_cosine_filter,
    simulate_ber,
)

pytest.importorskip('plotly', reason='the interactive backend needs the [viz] extra')

SPS = 8


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


def trace_names(figure):
    return [trace.name for trace in figure.data]


class TestConstellation:
    def test_reference_and_received_are_separate_traces(self, modulator, symbols, rng):
        received = Channels.awgn(symbols, 15.0, rng=rng)
        figure = plotly_constellation(modulator, received=received)
        assert trace_names(figure) == ['received', 'constellation']

    def test_reference_only_needs_one_trace(self, modulator):
        assert trace_names(plotly_constellation(modulator)) == ['constellation']

    def test_axes_are_locked_square(self, modulator):
        figure = plotly_constellation(modulator)
        assert figure.layout.yaxis.scaleanchor == 'x'
        assert figure.layout.yaxis.scaleratio == 1

    def test_accepts_raw_reference_points(self):
        figure = plotly_constellation(np.array([1 + 0j, -1 + 0j]))
        assert len(figure.data[0].x) == 2


class TestEyeDiagram:
    def test_windows_are_joined_by_nan_separators(self, shaped):
        figure = plotly_eye_diagram(shaped, SPS, n_traces=10)
        # 10 windows of 2*sps+1 points, each followed by a NaN break.
        assert len(figure.data[0].x) == 10 * (2 * SPS + 2)
        assert np.count_nonzero(np.isnan(figure.data[0].y)) == 10

    def test_rejects_sps_below_two(self, shaped):
        with pytest.raises(ValueError, match='at least 2'):
            plotly_eye_diagram(shaped, 1)

    def test_rejects_a_signal_too_short_for_one_window(self):
        with pytest.raises(ValueError, match='too short'):
            plotly_eye_diagram(np.zeros(4), SPS)


class TestSpectrumAndWaterfall:
    def test_psd_is_two_sided_and_sorted(self, shaped):
        figure = plotly_psd(shaped, fs=float(SPS))
        freqs = np.asarray(figure.data[0].x)
        assert freqs.min() < 0 < freqs.max()
        assert np.all(np.diff(freqs) > 0)

    def test_waterfall_is_logarithmic_and_carries_intervals(self, rng):
        result = simulate_ber(
            MQAMModulator(4), lambda x, snr, gen: Channels.awgn(x, snr, rng=gen),
            np.arange(0.0, 7.0, 3.0), bits_per_batch=2000, target_errors=20, rng=rng,
        )
        figure = plotly_waterfall({'QPSK': result})
        assert figure.layout.yaxis.type == 'log'
        assert figure.data[0].error_y.visible
        # Interval arms are clamped at zero, as in the matplotlib waterfall.
        assert np.all(np.asarray(figure.data[0].error_y.array) >= 0)

    def test_waterfall_rejects_an_empty_comparison(self):
        with pytest.raises(ValueError, match='at least one curve'):
            plotly_waterfall({})


class TestTannerGraph:
    def test_nodes_and_edges_are_separate_traces(self, rng):
        figure = plotly_tanner_graph(LDPCCode.from_gallager(24, 3, 6, rng=rng))
        assert trace_names(figure) == ['edges', 'variable nodes', 'check nodes']

    def test_hover_text_reports_node_degrees(self, rng):
        code = LDPCCode.from_gallager(24, 3, 6, rng=rng)
        figure = plotly_tanner_graph(code)
        assert figure.data[1].text[0] == 'v0, degree 3'
        assert figure.data[2].text[0] == f'c0, degree {int(code.H[0].sum())}'

    def test_edge_trace_has_one_break_per_edge(self, rng):
        code = LDPCCode.from_gallager(24, 3, 6, rng=rng)
        figure = plotly_tanner_graph(code)
        assert len(figure.data[0].x) == 3 * int(code.H.sum())

    def test_accepts_a_bare_matrix(self):
        figure = plotly_tanner_graph(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
        assert len(figure.data[1].x) == 3
