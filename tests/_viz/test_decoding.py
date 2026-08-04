"""Tests for commpy's decoder-diagnostic plots."""

from itertools import pairwise

import numpy as np
import pytest
from matplotlib.collections import LineCollection

from commpy import (
    Channels,
    ConvolutionalEncoder,
    LDPCCode,
    MPSKModulator,
    PolarCode,
    Trellis,
    plot_decoder_convergence,
    plot_exit_chart,
    plot_llr_histogram,
    plot_scl_paths,
    plot_viterbi_paths,
)
from commpy._channelCoding.turbo.rsc import RSCTrellis


@pytest.fixture
def bpsk():
    return MPSKModulator(2)


@pytest.fixture
def ldpc(rng):
    return LDPCCode.from_gallager(96, 3, 6, rng=rng)


def legend_labels(ax):
    legend = ax.get_legend()
    assert legend is not None
    return [text.get_text() for text in legend.get_texts()]


def received_llr(code, modulator, rng, snr_db=0.0):
    message = rng.integers(0, 2, code.k).astype(np.uint8)
    codeword = code.encode(message)
    noisy = Channels.awgn(modulator.modulate(codeword), snr_db, rng=rng)
    return message, codeword, modulator.soft_demodulate(noisy, 10 ** (-snr_db / 10))


class TestLLRHistogram:
    def test_splitting_by_bit_puts_bit_zero_on_the_positive_side(self, ldpc, bpsk, rng):
        _, codeword, llr = received_llr(ldpc, bpsk, rng, snr_db=3.0)
        ax = plot_llr_histogram(llr, bits=codeword)
        assert np.mean(llr[codeword == 0]) > 0 > np.mean(llr[codeword == 1])
        assert legend_labels(ax) == ['sent bit 0', 'sent bit 1']

    def test_pooled_histogram_without_bits_has_no_legend(self, ldpc, bpsk, rng):
        *_, llr = received_llr(ldpc, bpsk, rng)
        assert plot_llr_histogram(llr).get_legend() is None

    def test_marks_the_decision_threshold_at_zero(self, ldpc, bpsk, rng):
        *_, llr = received_llr(ldpc, bpsk, rng)
        ax = plot_llr_histogram(llr)
        assert any(np.allclose(np.asarray(line.get_xdata()), 0.0) for line in ax.lines)

    def test_rejects_a_bit_vector_of_the_wrong_length(self, ldpc, bpsk, rng):
        *_, llr = received_llr(ldpc, bpsk, rng)
        with pytest.raises(ValueError, match='same length'):
            plot_llr_histogram(llr, bits=np.zeros(3))


class TestDecoderConvergence:
    def test_syndrome_weight_decreases_and_reaches_zero(self, ldpc, bpsk, rng):
        *_, llr = received_llr(ldpc, bpsk, rng, snr_db=0.0)
        counts = np.asarray(plot_decoder_convergence(ldpc, llr, max_iter=12).lines[0].get_ydata())
        assert counts[0] > 0
        assert counts[-1] == 0
        assert np.all(np.diff(counts) <= 0)

    def test_reference_switches_the_curve_to_true_bit_errors(self, ldpc, bpsk, rng):
        message, _, llr = received_llr(ldpc, bpsk, rng, snr_db=0.0)
        ax = plot_decoder_convergence(ldpc, llr, max_iter=8, reference=message)
        assert ax.get_ylabel() == 'message bit errors'

    def test_traces_exactly_one_point_per_iteration(self, ldpc, bpsk, rng):
        *_, llr = received_llr(ldpc, bpsk, rng)
        assert np.asarray(
            plot_decoder_convergence(ldpc, llr, max_iter=7).lines[0].get_xdata(),
        ).tolist() == list(range(1, 8))

    def test_rejects_a_non_iterative_decoder(self):
        code = PolarCode(64, 32)
        with pytest.raises(TypeError, match='neither max_iter nor iterations'):
            plot_decoder_convergence(code, np.zeros(64), max_iter=3)


class TestExitChart:
    @pytest.fixture
    def chart(self, rng):
        return plot_exit_chart(
            RSCTrellis(3, 0o7, 0o5), snr_db=-4.0, n_points=6, n_bits=2000, rng=rng,
        )

    def test_information_measures_stay_within_one_bit(self, chart):
        for line in chart.lines:
            assert np.all((np.asarray(line.get_xdata()) >= 0) & (
                np.asarray(line.get_xdata()) <= 1))
            assert np.all((np.asarray(line.get_ydata()) >= 0) & (
                np.asarray(line.get_ydata()) <= 1))

    def test_transfer_characteristic_is_non_decreasing(self, chart):
        i_e = np.asarray(chart.lines[0].get_ydata())
        # More a-priori information can never make the extrinsic output worse.
        assert np.all(np.diff(i_e) > -0.05)

    def test_extrinsic_information_beats_a_priori_so_the_tunnel_is_open(self, chart):
        i_a = np.asarray(chart.lines[0].get_xdata())
        i_e = np.asarray(chart.lines[0].get_ydata())
        assert np.all(i_e > i_a)

    def test_second_curve_is_the_mirror_of_the_first(self, chart):
        first, second = chart.lines
        assert np.array_equal(np.asarray(first.get_xdata()),
                              np.asarray(second.get_ydata()))
        assert np.array_equal(np.asarray(first.get_ydata()),
                              np.asarray(second.get_xdata()))


class TestViterbiPaths:
    @pytest.fixture
    def trellis(self):
        return Trellis(3, [0o7, 0o5])

    @pytest.fixture
    def received(self, trellis, rng):
        coded, _ = ConvolutionalEncoder(trellis).encode(rng.integers(0, 2, 20))
        corrupted = coded.copy()
        corrupted[3] ^= 1
        return corrupted

    def test_draws_one_survivor_per_state_and_stage(self, trellis, received):
        ax = plot_viterbi_paths(trellis, received, max_stages=6)
        survivors = next(c for c in ax.collections if isinstance(c, LineCollection))
        assert len(survivors.get_segments()) == 6 * trellis.n_states

    def test_highlighted_path_follows_real_trellis_transitions(self, trellis, received):
        ax = plot_viterbi_paths(trellis, received, max_stages=8)
        states = np.asarray(ax.lines[0].get_ydata()).astype(int)
        for current, following in pairwise(states):
            assert following in set(trellis.next_state[current].tolist())

    def test_path_starts_in_the_zero_state(self, trellis, received):
        ax = plot_viterbi_paths(trellis, received, max_stages=5)
        assert np.asarray(ax.lines[0].get_ydata())[0] == 0

    def test_max_stages_caps_the_drawing_not_the_decoding(self, trellis, received):
        ax = plot_viterbi_paths(trellis, received, max_stages=3)
        assert ax.get_xlim()[1] == pytest.approx(3.3)

    def test_rejects_an_unknown_mode(self, trellis, received):
        with pytest.raises(ValueError, match='hard'):
            plot_viterbi_paths(trellis, received, mode='fuzzy')

    def test_rejects_a_length_inconsistent_with_the_trellis(self, trellis):
        with pytest.raises(ValueError, match='multiple of'):
            plot_viterbi_paths(trellis, np.zeros(5))


class TestSCLPaths:
    def test_metrics_are_sorted_best_first(self, bpsk, rng):
        code = PolarCode(64, 32)
        _, _, llr = received_llr(code, bpsk, rng, snr_db=0.0)
        ax = plot_scl_paths(llr, code.frozen, list_size=8)
        metrics = np.array([bar.get_height() for bar in ax.containers[0]])
        assert np.all(np.diff(metrics) >= 0)

    def test_one_bar_per_survivor(self, bpsk, rng):
        code = PolarCode(64, 32)
        _, _, llr = received_llr(code, bpsk, rng)
        assert len(plot_scl_paths(llr, code.frozen, list_size=4).containers[0]) == 4

    def test_winner_is_highlighted_against_the_rest(self, bpsk, rng):
        code = PolarCode(64, 32)
        _, _, llr = received_llr(code, bpsk, rng)
        bars = plot_scl_paths(llr, code.frozen, list_size=4).containers[0]
        colors = {tuple(bar.get_facecolor()) for bar in bars}
        assert len(colors) == 2
