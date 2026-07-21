"""Tests for commpy.Channels (discrete and analog channel impairment models)."""

import numpy as np

from commpy import Channels


def test_bsc(rng):
    bits = np.zeros(10000, dtype=int)
    out = Channels.bsc(bits, p=0.1, rng=rng)
    flips = np.sum(out != bits)
    frac = flips / len(bits)
    assert abs(frac - 0.1) < 0.01


def test_bec(rng):
    bits = np.ones(10000, dtype=int)
    out = Channels.bec(bits, p=0.2, erasure_value=-1, rng=rng)
    erasures = np.sum(out == -1)
    frac = erasures / len(bits)
    assert abs(frac - 0.2) < 0.02


def test_awgn(rng):
    x = np.ones(100000)
    snr_db = 10.0
    y = Channels.awgn(x, snr_db, rng=rng)
    noise = y - x
    signal_power = np.mean(x**2)
    noise_power = np.mean(noise**2)
    snr_est = 10 * np.log10(signal_power / noise_power)
    assert abs(snr_est - snr_db) < 0.25


def test_rayleigh_reduces_snr(rng):
    x = np.ones(50000, dtype=complex)
    y = Channels.rayleigh(x, snr_db=20.0, rng=rng)
    assert y.shape == x.shape
    # Fading should introduce variance in received magnitude that a pure
    # AWGN channel at the same nominal SNR would not have.
    assert np.std(np.abs(y)) > 0.05


def test_rician_high_k_approaches_awgn(rng):
    x = np.ones(50000, dtype=complex)
    y = Channels.rician(x, snr_db=20.0, k_factor=1000.0, rng=rng)
    # Very high K-factor => dominated by the LOS component => mean magnitude near 1.
    assert abs(np.mean(np.abs(y)) - 1.0) < 0.05


def test_z_channel_only_flips_ones(rng):
    bits = np.array([0] * 5000 + [1] * 5000)
    out = Channels.z_channel(bits, p=0.3, rng=rng)
    assert np.all(out[bits == 0] == 0)
    flipped_ones = np.sum(out[bits == 1] == 0)
    frac = flipped_ones / 5000
    assert abs(frac - 0.3) < 0.03


def test_gilbert_elliott_error_bounds(rng):
    bits = np.zeros(20000, dtype=int)
    out = Channels.gilbert_elliott(
        bits, p_gb=0.05, p_bg=0.2, p_good=0.0, p_bad=0.5, rng=rng,
    )
    err_frac = np.mean(out != bits)
    # Overall error rate must lie strictly between the two state error rates.
    assert 0.0 < err_frac < 0.5


def test_quantize_range_and_levels():
    x = np.linspace(-1.0, 1.0, 1000)
    xq = Channels.quantize(x, bits=2, vmin=-1.0, vmax=1.0)
    unique_levels = np.unique(np.round(xq, 10))
    assert len(unique_levels) <= 4
    assert np.max(xq) <= 1.0 + 1e-9
    assert np.min(xq) >= -1.0 - 1e-9
