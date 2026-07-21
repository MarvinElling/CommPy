"""Tests for commpy.gardner_timing_error / estimate_cfo_mth_power / costas_loop_bpsk."""

import numpy as np
import pytest

from commpy import (
    costas_loop_bpsk,
    estimate_cfo_mth_power,
    gardner_timing_error,
    raised_cosine_filter,
)


def _pulse_shaped_bpsk(rng, n_sym, oversample, rolloff=0.35, span=8):
    """Build a pulse-shaped BPSK signal.

    Uses explicit `'full'`-mode convolution with a tracked peak index --
    `'same'`-mode convolution's implicit centering is fragile across
    different array-length combinations and was a source of test bugs here.

    Returns `(bits, fine, base)`, `base` being the fine-grid index of symbol
    0's peak.
    """
    bits = (rng.integers(0, 2, n_sym) * 2 - 1).astype(float)
    g = raised_cosine_filter(1.0, rolloff, span)
    tau = np.arange(0, span * oversample) / oversample
    pulse = g(tau)
    peak_idx = int(np.argmax(pulse))

    upsampled = np.zeros(n_sym * oversample)
    upsampled[::oversample] = bits
    fine = np.convolve(upsampled, pulse, mode='full')
    base = peak_idx  # fine-grid index where symbol 0's peak lands
    return bits, fine, base


def _sample_grid(fine, base, step, offset, n_symbols):
    """Sample `fine` on a `step`-spaced grid starting at `base + offset`."""
    idx = base + offset + np.arange(n_symbols) * step
    idx = idx[(idx >= 0) & (idx < len(fine))]
    return fine[idx]


def test_gardner_error_near_zero_at_correct_timing(rng):
    oversample = 40
    n_sym = 200
    _, fine, base = _pulse_shaped_bpsk(rng, n_sym, oversample)
    # Sample at 2 samples/symbol, aligned to the (known, by construction) optimal instant.
    samples = _sample_grid(fine, base, oversample // 2, 0, 2 * n_sym)
    err = gardner_timing_error(samples[40:-40])
    assert abs(np.mean(err)) < 0.05


def test_gardner_error_sign_indicates_early_vs_late(rng):
    oversample = 40
    n_sym = 200
    _, fine, base = _pulse_shaped_bpsk(rng, n_sym, oversample)
    step = oversample // 2

    early = _sample_grid(fine, base, step, -4, 2 * n_sym)
    late = _sample_grid(fine, base, step, 4, 2 * n_sym)
    err_early = np.mean(gardner_timing_error(early[40:-40]))
    err_late = np.mean(gardner_timing_error(late[40:-40]))
    assert err_early < 0
    assert err_late > 0


def test_gardner_rejects_too_few_samples():
    with pytest.raises(ValueError, match='at least 2 symbols'):
        gardner_timing_error([1 + 0j, 2 + 0j])


def test_cfo_estimation_matches_true_offset(rng):
    fs = 1000.0
    n = 4000
    t = np.arange(n) / fs
    true_cfo = 15.3
    bits = rng.integers(0, 4, n)
    symbols = np.exp(1j * (np.pi / 2 * bits))  # QPSK
    signal = symbols * np.exp(1j * 2 * np.pi * true_cfo * t)

    est = estimate_cfo_mth_power(signal, fs, m_order=4)
    resolution = fs / n / 4
    assert abs(est - true_cfo) < 2 * resolution


def test_cfo_estimation_zero_offset():
    fs = 1000.0
    n = 2000
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, n)
    symbols = np.where(bits == 0, 1.0 + 0j, -1.0 + 0j)  # BPSK, no offset
    est = estimate_cfo_mth_power(symbols, fs, m_order=2)
    assert abs(est) < fs / n


def test_costas_loop_converges_and_recovers_bpsk(rng):
    n = 2000
    bits = rng.integers(0, 2, n) * 2 - 1
    phase_offset = 1.2
    signal = bits.astype(complex) * np.exp(1j * phase_offset)

    corrected, phase_history = costas_loop_bpsk(signal, loop_gain=0.05)
    tail = corrected[-200:]
    assert np.mean(np.abs(tail.imag)) < 0.01

    recovered_bits = np.sign(tail.real)
    true_tail = bits[-200:]
    # BPSK Costas loops have an inherent 180-degree phase ambiguity.
    match_rate = max(
        np.mean(recovered_bits == true_tail), np.mean(recovered_bits == -true_tail),
    )
    assert match_rate > 0.99
    assert phase_history.shape == signal.shape
