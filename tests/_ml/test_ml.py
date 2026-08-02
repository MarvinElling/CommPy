"""Tests for the optional PyTorch AI-for-wireless layer (`commpy.ml`).

Skipped entirely when torch is not installed. The learning tests assert
*improvement* rather than exact values (training an autoencoder below the
uncoded error rate; a neural demapper agreeing with the analytic one at high
SNR; a neural min-sum decoder reproducing the classical decoder with unit
weights), which is the robust way to test stochastic optimization.
"""

import numpy as np
import pytest

torch = pytest.importorskip('torch')

import commpy  # noqa: E402
from commpy import (  # noqa: E402
    Channels,
    LDPCCode,
    MQAMModulator,
    ml,
)


@pytest.fixture(autouse=True)
def _seed_torch():
    torch.manual_seed(0)


def test_ml_layer_is_not_flat_re_exported():
    # The torch-dependent API must live under commpy.ml, never on `commpy` itself.
    for name in ('Autoencoder', 'NeuralDemapper', 'NeuralMinSumDecoder'):
        assert name not in commpy.__all__
        assert name in ml.__all__


def test_awgn_channel_is_differentiable_and_shapes_match():
    x = torch.zeros(8, 4, 2, requires_grad=True)
    y = ml.awgn(x + 1.0, snr_db=5.0)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    with pytest.raises(ValueError, match='trailing dimension'):
        ml.awgn(torch.zeros(4, 3), snr_db=5.0)


def test_normalize_power_gives_unit_symbol_energy():
    x = 3.0 * torch.randn(1000, 2)
    normalized = ml.normalize_power(x)
    mean_energy = normalized.pow(2).sum(dim=-1).mean().item()
    assert abs(mean_energy - 1.0) < 1e-4


def test_autoencoder_learns_below_uncoded():
    model = ml.Autoencoder(num_messages=4, num_channel_uses=2, hidden=32)
    before = ml.block_error_rate(model, 6.0, num_messages=20_000, seed=1)
    ml.train_autoencoder(model, snr_db=6.0, steps=600, batch_size=256, learning_rate=1e-2, seed=0)
    after = ml.block_error_rate(model, 6.0, num_messages=20_000, seed=1)
    assert after < before
    assert after < 0.05  # a (2, 4) autoencoder trains to a low block-error rate


def test_neural_demapper_matches_analytic_at_high_snr():
    mod = MQAMModulator(4)
    demapper = ml.NeuralDemapper(bits_per_symbol=mod.bits_per_symbol, hidden=32)
    ml.train_demapper(demapper, mod, snr_db=10.0, steps=400, symbols_per_step=1000, seed=0)

    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, mod.bits_per_symbol * 4000)
    received = Channels.awgn(mod.modulate(bits), 12.0, rng=rng)
    llr_nn = demapper.soft_demodulate(received)
    llr_true = mod.soft_demodulate(received, noise_var=10.0 ** -1.2)
    assert llr_nn.shape == llr_true.shape
    agreement = np.mean((llr_nn < 0) == (llr_true < 0))
    assert agreement > 0.98


def test_neural_min_sum_reproduces_classical_decoding():
    code = LDPCCode.from_gallager(n=48, w_c=3, w_r=6, rng=np.random.default_rng(0))
    decoder = ml.NeuralMinSumDecoder(code, num_iterations=8)  # unit weights == plain min-sum

    rng = np.random.default_rng(3)
    for _ in range(5):
        message = rng.integers(0, 2, code.k).astype(np.uint8)
        codeword = code.encode(message)
        llr = np.where(codeword == 0, 8.0, -8.0)
        np.testing.assert_array_equal(decoder.decode(llr), codeword)


def test_neural_min_sum_batched_decode_shape_and_training():
    code = LDPCCode.from_gallager(n=48, w_c=3, w_r=6, rng=np.random.default_rng(0))
    decoder = ml.NeuralMinSumDecoder(code, num_iterations=5)
    batch = np.tile(np.full(code.n, 8.0), (4, 1))
    assert decoder.decode(batch).shape == (4, code.n)

    history = ml.train_neural_min_sum(decoder, code, snr_db=1.0, steps=30, batch_size=32, seed=0)
    assert len(history) == 30
    assert all(np.isfinite(history))
    assert history[-1] <= history[0]  # training does not diverge
