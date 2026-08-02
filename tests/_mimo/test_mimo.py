"""Tests for the MIMO suite: channel, Alamouti STBC, detectors, and capacity.

The key correctness check is a cross-validation of the K-best sphere detector
against exhaustive maximum-likelihood detection: for two transmit antennas a
list size of at least `M` keeps every survivor at the top layer, so K-best is
*exactly* ML -- an equality check, in the brute-force style used across the
suite. The rest are structural, recovery-at-high-SNR, diversity-gain, and
closed-form-capacity checks.
"""

import numpy as np
import pytest

from commpy import (
    MQAMModulator,
    alamouti_decode,
    alamouti_encode,
    ergodic_mimo_capacity,
    kbest_detector,
    mimo_awgn,
    mimo_capacity,
    mimo_noise_variance,
    ml_detector,
    mmse_detector,
    rayleigh_channel_matrix,
    zf_detector,
)


def _random_symbols(mod, n_tx, n_uses, rng):
    bits = rng.integers(0, 2, n_tx * n_uses * mod.bits_per_symbol)
    return bits, mod.modulate(bits).reshape(n_tx, n_uses)


def test_channel_matrix_shape_and_unit_variance(rng):
    H = rayleigh_channel_matrix(200, 100, rng=rng)
    assert H.shape == (200, 100)
    assert abs(float(np.mean(np.abs(H) ** 2)) - 1.0) < 0.02  # unit average power


def test_mimo_awgn_shapes_and_validation(rng):
    H = rayleigh_channel_matrix(3, 2, rng=rng)
    assert mimo_awgn(np.ones(2, dtype=complex), H, 10.0, rng=rng).shape == (3,)
    assert mimo_awgn(np.ones((2, 5), dtype=complex), H, 10.0, rng=rng).shape == (3, 5)
    with pytest.raises(ValueError, match='transmit dimension'):
        mimo_awgn(np.ones(4, dtype=complex), H, 10.0, rng=rng)


@pytest.mark.parametrize('detector', ['zf', 'mmse'])
def test_linear_detectors_recover_symbols_at_high_snr(detector, rng):
    mod = MQAMModulator(4)
    H = rayleigh_channel_matrix(4, 2, rng=rng)  # 2x4, well-conditioned on average
    bits, x = _random_symbols(mod, 2, 2000, rng)
    snr_db = 20.0
    y = mimo_awgn(x, H, snr_db, rng=rng)
    x_hat = (
        zf_detector(y, H) if detector == 'zf'
        else mmse_detector(y, H, mimo_noise_variance(snr_db))
    )
    ber = float(np.mean(mod.demodulate(x_hat.reshape(-1)) != bits))
    assert ber < 1e-3


def test_zero_forcing_is_exact_without_noise(rng):
    H = rayleigh_channel_matrix(3, 2, rng=rng)
    x = rng.standard_normal((2, 10)) + 1j * rng.standard_normal((2, 10))
    np.testing.assert_allclose(zf_detector(H @ x, H), x, atol=1e-9)


def test_kbest_equals_ml_for_two_transmit_antennas(rng):
    mod = MQAMModulator(4)
    constellation = mod.constellation
    for _ in range(20):
        H = rayleigh_channel_matrix(2, 2, rng=rng)
        _, x = _random_symbols(mod, 2, 50, rng)
        y = mimo_awgn(x, H, 8.0, rng=rng)
        ml = ml_detector(y, H, constellation)
        kbest = kbest_detector(y, H, constellation, list_size=constellation.size)
        np.testing.assert_allclose(kbest, ml)  # list_size >= M -> exact ML for n_tx=2


def test_kbest_validation(rng):
    H = rayleigh_channel_matrix(2, 2, rng=rng)
    mod = MQAMModulator(4)
    with pytest.raises(ValueError, match='list_size'):
        kbest_detector(np.ones(2, dtype=complex), H, mod.constellation, list_size=0)
    tall = rayleigh_channel_matrix(2, 3, rng=rng)  # n_rx < n_tx
    with pytest.raises(ValueError, match='n_rx >= n_tx'):
        kbest_detector(np.ones(2, dtype=complex), tall, mod.constellation)


def test_alamouti_round_trip_and_validation(rng):
    mod = MQAMModulator(4)
    for n_rx in (1, 2):
        H = rayleigh_channel_matrix(n_rx, 2, rng=rng)
        bits = rng.integers(0, 2, 2000 * mod.bits_per_symbol)
        symbols = mod.modulate(bits)
        transmitted = alamouti_encode(symbols)
        assert transmitted.shape == (2, symbols.size)
        received = mimo_awgn(transmitted, H, 20.0, rng=rng)
        recovered = mod.demodulate(alamouti_decode(received, H))
        assert float(np.mean(recovered != bits)) < 1e-3

    with pytest.raises(ValueError, match='even number'):
        alamouti_encode(np.ones(3, dtype=complex))


def test_alamouti_beats_siso_diversity(rng):
    mod = MQAMModulator(4)
    snr_db, n_sym = 6.0, 4000
    bits = rng.integers(0, 2, n_sym * mod.bits_per_symbol)
    symbols = mod.modulate(bits)

    h_siso = rayleigh_channel_matrix(1, 1, rng=rng)
    siso_rx = mimo_awgn(symbols[None, :], h_siso, snr_db, rng=rng)
    siso_ber = float(np.mean(mod.demodulate((siso_rx / h_siso[0, 0]).reshape(-1)) != bits))

    h_alamouti = rayleigh_channel_matrix(1, 2, rng=rng)
    alamouti_rx = mimo_awgn(alamouti_encode(symbols), h_alamouti, snr_db, rng=rng)
    alamouti_ber = float(np.mean(mod.demodulate(alamouti_decode(alamouti_rx, h_alamouti)) != bits))

    assert alamouti_ber < siso_ber  # transmit diversity lowers the error rate


def test_mimo_capacity_matches_closed_form():
    n, snr_db = 3, 10.0
    identity_capacity = mimo_capacity(np.eye(n), snr_db)
    expected = n * np.log2(1 + 10 ** (snr_db / 10) / n)  # H = I: n parallel channels
    assert identity_capacity == pytest.approx(expected)


def test_ergodic_capacity_grows_with_antennas(rng):
    c1 = ergodic_mimo_capacity(1, 1, 10.0, n_trials=400, rng=rng)
    c2 = ergodic_mimo_capacity(2, 2, 10.0, n_trials=400, rng=rng)
    c4 = ergodic_mimo_capacity(4, 4, 10.0, n_trials=400, rng=rng)
    assert c1 < c2 < c4  # spatial-multiplexing gain
