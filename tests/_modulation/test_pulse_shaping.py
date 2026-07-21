"""Tests for commpy.raised_cosine_filter / root_raised_cosine_filter.

Both return callables compatible with `IQWaveform`'s `pulse_shape`
parameter: causal, defined on `tau in [0, span*T)`, peaking at `span*T/2`.
"""

import numpy as np
import pytest

from commpy import IQWaveform, raised_cosine_filter, root_raised_cosine_filter


@pytest.mark.parametrize('rolloff', [0.0, 0.25, 0.5, 1.0])
def test_rc_zero_isi_at_symbol_spaced_instants(rolloff):
    T = 1.0
    span = 8
    g = raised_cosine_filter(T, rolloff, span)
    center = span * T / 2
    # Sample at the peak and at every other symbol-spaced instant within the span.
    offsets = np.arange(-span // 2 + 1, span // 2)
    tau = np.asarray(center + offsets * T, dtype=np.float64)
    values = g(tau)
    expected = (offsets == 0).astype(float)
    np.testing.assert_allclose(values, expected, atol=1e-9)


def test_rc_peaks_at_span_half():
    T = 2.0
    span = 6
    g = raised_cosine_filter(T, 0.35, span)
    tau = np.linspace(0, span * T, 5001)
    values = g(tau)
    peak_tau = tau[np.argmax(values)]
    assert abs(peak_tau - span * T / 2) < 1e-2


def test_rc_rejects_invalid_rolloff():
    with pytest.raises(ValueError, match='rolloff'):
        raised_cosine_filter(1.0, 1.5, 8)
    with pytest.raises(ValueError, match='rolloff'):
        raised_cosine_filter(1.0, -0.1, 8)


def test_rrc_rejects_invalid_rolloff():
    with pytest.raises(ValueError, match='rolloff'):
        root_raised_cosine_filter(1.0, -0.1, 8)


def test_rrc_self_convolution_approximates_rc():
    # A matched RRC/RRC pair's combined response is the RC pulse (up to a
    # scale factor); verify this numerically via discrete convolution.
    T = 1.0
    rolloff = 0.35
    span = 8
    oversample = 64
    g_rrc = root_raised_cosine_filter(T, rolloff, span)
    g_rc = raised_cosine_filter(T, rolloff, span)

    dt = T / oversample
    tau = np.arange(0, span * T, dt)
    h = g_rrc(tau)
    conv = np.convolve(h, h, mode='same') * dt
    conv /= conv[len(conv) // 2]  # normalize peak to 1, matching RC's peak of 1

    center = span * T / 2
    offsets = np.arange(-3, 4)
    sample_idx = len(conv) // 2 + (offsets * oversample).astype(int)
    rc_expected = g_rc(np.asarray(center + offsets * T, dtype=np.float64))
    np.testing.assert_allclose(conv[sample_idx], rc_expected, atol=0.02)


def test_pulse_shape_integrates_with_iqwaveform():
    T = 1e-3
    fs = 20_000.0
    span = 6
    pulse = raised_cosine_filter(T, 0.3, span)
    symbols = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    wf = IQWaveform(symbols, np.zeros(5), T, fs, f0=0.0, pulse_shape=pulse, span=span)
    assert np.all(np.isfinite(wf.s))
    assert wf.s.shape == wf.t.shape
