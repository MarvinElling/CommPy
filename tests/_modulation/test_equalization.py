"""Tests for commpy.zf_equalizer / mmse_equalizer."""

import numpy as np

from commpy import mmse_equalizer, zf_equalizer


def test_zf_equalizer_inverts_trivial_channel():
    # An ideal (delta) channel needs no equalization: the ZF solution should
    # itself be (approximately) a delta at the target delay.
    channel = [1.0]
    n_taps = 5
    w = zf_equalizer(channel, n_taps)
    peak = np.argmax(np.abs(w))
    assert np.abs(w[peak]) > 0.9
    other_energy = np.sum(w**2) - w[peak] ** 2
    assert other_energy < 1e-6


def test_zf_equalizer_reduces_isi():
    channel = [0.2, 1.0, 0.3, -0.1]
    n_taps = 15
    w = zf_equalizer(channel, n_taps)
    combined = np.convolve(channel, w)
    peak_idx = np.argmax(np.abs(combined))
    peak = combined[peak_idx]
    residual = combined.copy()
    residual[peak_idx] = 0
    # The combined channel+equalizer response should be strongly peaked
    # (most energy at one tap) -- i.e. ISI is substantially suppressed.
    assert np.max(np.abs(residual)) < 0.15 * np.abs(peak)


def test_mmse_matches_zf_at_zero_noise():
    channel = [0.2, 1.0, -0.3, 0.15]
    n_taps = 11
    w_zf = zf_equalizer(channel, n_taps)
    w_mmse = mmse_equalizer(channel, n_taps, noise_var=0.0)
    np.testing.assert_allclose(w_mmse, w_zf, atol=1e-6)


def test_mmse_shrinks_with_higher_noise_variance():
    channel = [0.2, 1.0, -0.3, 0.15]
    n_taps = 11
    w_low_noise = mmse_equalizer(channel, n_taps, noise_var=1e-6)
    w_high_noise = mmse_equalizer(channel, n_taps, noise_var=10.0)
    assert np.linalg.norm(w_high_noise) < np.linalg.norm(w_low_noise)


def test_mmse_outperforms_zf_under_noise():
    # Classic linear-equalizer tradeoff: MMSE trades a bit of residual ISI
    # for much better noise suppression, so at high noise its equalized
    # output should have lower error energy than ZF's noise-amplified one.
    rng = np.random.default_rng(0)
    channel = np.array([0.3, 1.0, -0.4, 0.2, -0.1])
    n_taps = 21
    noise_var = 0.5

    w_zf = zf_equalizer(channel, n_taps)
    w_mmse = mmse_equalizer(channel, n_taps, noise_var=noise_var)

    n_trials = 200
    zf_errors = []
    mmse_errors = []
    delay = (len(channel) + n_taps - 1) // 2
    for _ in range(n_trials):
        bits = rng.choice([-1.0, 1.0], size=200)
        tx = np.convolve(bits, channel)
        noise = rng.normal(scale=np.sqrt(noise_var), size=tx.shape)
        rx = tx + noise

        eq_zf = np.convolve(rx, w_zf)
        eq_mmse = np.convolve(rx, w_mmse)

        # Compare against the delayed original bit sequence at valid indices.
        n_valid = len(bits) - 5
        target = bits[:n_valid]
        zf_errors.append(np.mean((eq_zf[delay:delay + n_valid] - target) ** 2))
        mmse_errors.append(np.mean((eq_mmse[delay:delay + n_valid] - target) ** 2))

    assert np.mean(mmse_errors) < np.mean(zf_errors)
