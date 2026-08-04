"""Tests for commpy's animated visualizations."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.animation import FuncAnimation

from commpy import (
    Channels,
    ConvolutionalEncoder,
    LDPCCode,
    MPSKModulator,
    MQAMModulator,
    PolarCode,
    Trellis,
    animate_constellation,
    animate_decoding,
    animate_viterbi,
)

# Tests that only inspect an animation's structure never render it, and
# matplotlib warns about exactly that when the object is collected.
pytestmark = pytest.mark.filterwarnings('ignore:Animation was deleted:UserWarning')


def frame_count(animation):
    return len(list(animation.new_frame_seq()))


def render(animation, tmp_path, name):
    """Write the animation out, which forces every frame to be drawn."""
    path = tmp_path / f'{name}.gif'
    animation.save(str(path), writer='pillow', fps=4)
    return path


@pytest.fixture
def ldpc(rng):
    return LDPCCode.from_gallager(48, 3, 6, rng=rng)


@pytest.fixture
def ldpc_llr(ldpc, rng):
    modulator = MPSKModulator(2)
    codeword = ldpc.encode(rng.integers(0, 2, ldpc.k).astype(np.uint8))
    noisy = Channels.awgn(modulator.modulate(codeword), 0.0, rng=rng)
    return modulator.soft_demodulate(noisy, 1.0)


@pytest.fixture
def trellis():
    return Trellis(3, [0o7, 0o5])


@pytest.fixture
def received(trellis, rng):
    coded, _ = ConvolutionalEncoder(trellis).encode(rng.integers(0, 2, 12))
    corrupted = coded.copy()
    corrupted[3] ^= 1
    return corrupted


class TestConstellationAnimation:
    def test_one_frame_per_snr_point(self, rng):
        animation = animate_constellation(
            MQAMModulator(4), [15.0, 10.0, 5.0], n_symbols=100, rng=rng,
        )
        assert isinstance(animation, FuncAnimation)
        assert frame_count(animation) == 3

    def test_every_frame_renders(self, rng, tmp_path):
        animation = animate_constellation(
            MQAMModulator(4), [12.0, 6.0], n_symbols=100, rng=rng,
        )
        assert render(animation, tmp_path, 'constellation').stat().st_size > 0

    def test_reference_constellation_is_drawn_behind_the_cloud(self, rng):
        modulator = MQAMModulator(16)
        animate_constellation(modulator, [10.0], n_symbols=50, rng=rng)
        # The animation builds its figure with plt.subplots, so it is current.
        reference = np.asarray(plt.gcf().axes[0].collections[1].get_offsets())
        assert reference.shape == (modulator.M, 2)

    def test_accepts_a_custom_channel(self, rng):
        def noiseless(symbols, snr_db, generator):  # noqa: ARG001 -- signature fixed by the callback contract
            return symbols

        animation = animate_constellation(
            MQAMModulator(4), [10.0], n_symbols=40, channel_fn=noiseless, rng=rng,
        )
        assert frame_count(animation) == 1


class TestDecodingAnimation:
    def test_one_frame_per_iteration(self, ldpc, ldpc_llr):
        assert frame_count(animate_decoding(ldpc, ldpc_llr, max_iter=5)) == 5

    def test_every_frame_renders(self, ldpc, ldpc_llr, tmp_path):
        animation = animate_decoding(ldpc, ldpc_llr, max_iter=4)
        assert render(animation, tmp_path, 'decoding').stat().st_size > 0

    def test_draws_one_bar_per_codeword_bit(self, ldpc, ldpc_llr):
        animate_decoding(ldpc, ldpc_llr, max_iter=3)
        assert len(plt.gcf().axes[0].containers[0]) == ldpc.n

    def test_rejects_a_decoder_without_an_iteration_cap(self):
        with pytest.raises(TypeError, match='neither max_iter nor iterations'):
            animate_decoding(PolarCode(64, 32), np.zeros(64))

    def test_rejects_an_object_that_cannot_decode_at_all(self):
        with pytest.raises(TypeError, match='no decode\\(\\) method'):
            animate_decoding(object(), np.zeros(8))


class TestViterbiAnimation:
    def test_frames_cover_the_forward_pass_and_the_traceback(self, trellis, received):
        animation = animate_viterbi(trellis, received, max_stages=5)
        # Five forward stages, then the traceback grows over six path nodes.
        assert frame_count(animation) == 5 + 6

    def test_every_frame_renders(self, trellis, received, tmp_path):
        animation = animate_viterbi(trellis, received, max_stages=4)
        assert render(animation, tmp_path, 'viterbi').stat().st_size > 0

    def test_rejects_an_unknown_mode(self, trellis, received):
        with pytest.raises(ValueError, match='hard'):
            animate_viterbi(trellis, received, mode='fuzzy')

    def test_rejects_a_length_inconsistent_with_the_trellis(self, trellis):
        with pytest.raises(ValueError, match='multiple of'):
            animate_viterbi(trellis, np.zeros(5))
