"""Tests for commpy.IQWaveform (pulse-shaped IQ waveform synthesis)."""

import matplotlib as mpl

mpl.use('Agg')  # headless backend so plot_* methods don't require a display

import numpy as np

from commpy import IQWaveform


def test_baseband_rect_pulse_matches_symbols_at_sample_centers():
    i_symbols = np.array([1.0, -1.0, 1.0, -1.0])
    q_symbols = np.zeros(4)
    T = 1e-3
    fs = 10_000.0  # 10 samples per symbol
    wf = IQWaveform(i_symbols, q_symbols, T, fs, f0=0.0)

    samples_per_symbol = int(T * fs)
    for n, expected in enumerate(i_symbols):
        idx = n * samples_per_symbol + samples_per_symbol // 2
        assert wf.s_I[idx] == expected

    # f0=0 => carrier_cos is constant sqrt(2), carrier_sin is 0, so s = sqrt(2) * s_I.
    np.testing.assert_allclose(wf.s, np.sqrt(2) * wf.s_I)


def test_output_length_matches_symbol_count_times_samples_per_symbol():
    N = 8
    T = 1e-3
    fs = 5_000.0
    wf = IQWaveform(np.ones(N), np.zeros(N), T, fs)
    assert wf.t.shape[0] == int(N * T * fs)
    assert wf.s.shape == wf.t.shape


def test_custom_pulse_shape_is_used():
    calls = []

    def tracking_pulse(tau: np.ndarray) -> np.ndarray:
        calls.append(tau)
        return np.ones_like(tau)

    IQWaveform(np.ones(2), np.zeros(2), T=1e-3, fs=1_000.0, pulse_shape=tracking_pulse)
    assert len(calls) == 2  # invoked once per symbol


def test_plot_methods_run_without_error():
    wf = IQWaveform(np.array([1.0, -1.0, 1.0, -1.0]), np.zeros(4), T=1e-3, fs=4_000.0)
    wf.plot_waveform()
    wf.plot_IQ_baseband()
    wf.plot_eye()
