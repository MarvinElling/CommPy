"""Minimal receiver synchronization primitives: symbol timing, carrier frequency, carrier phase.

Deliberately narrow in scope (per the project's design notes): each function
is a single well-defined textbook algorithm, not a configurable PLL/loop-
filter framework. `gardner_timing_error` is a timing-error *discriminator*
(callers wanting closed-loop resampling can feed it into their own loop
filter); `costas_loop_bpsk` bundles a fixed proportional-gain feedback loop
since a single scalar gain is simple enough not to need general
configurability.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gardner_timing_error(samples: ArrayLike) -> NDArray[np.float64]:
    """Gardner (1986) non-data-aided timing-error detector.

    Requires `samples` at exactly 2 samples/symbol, alternating on-time and
    mid-symbol instants: `[y_0, y_0.5, y_1, y_1.5, y_2, ...]`. For symbol
    index `k`, `error[k] = Re{ conj(y[k+0.5]) * (y[k+1] - y[k]) }`: zero at
    the correct sampling instant, negative when sampling early, positive
    when sampling late (sign convention verified against the point of
    maximum eye-opening in the test suite).

    Args:
        samples: Complex baseband samples at 2 samples/symbol, length
            `2*n_symbols`.

    Returns:
        Per-symbol timing error, length `n_symbols - 1`.

    Raises:
        ValueError: If `len(samples) < 4` (need at least 2 full symbols).
    """
    arr = np.asarray(samples, dtype=np.complex128)
    if arr.size < 4:
        msg = 'samples must contain at least 2 symbols (4 samples at 2 sps).'
        raise ValueError(msg)
    on_time = arr[0::2]
    mid = arr[1::2]
    n_errors = min(len(on_time) - 1, len(mid))
    return np.asarray(
        (np.conj(mid[:n_errors]) * (on_time[1:n_errors + 1] - on_time[:n_errors])).real,
    )


def estimate_cfo_mth_power(signal: ArrayLike, fs: float, m_order: int) -> float:
    """Estimate carrier frequency offset (CFO) via the M-th power method.

    Raising an M-PSK signal to the M-th power strips the (M-ary) modulation,
    leaving a pure tone at `M * cfo`; the CFO is recovered from that tone's
    FFT peak. Resolution is `fs / len(signal) / m_order`.

    Args:
        signal: Complex baseband M-PSK samples (one or more samples/symbol).
        fs: Sample rate (Hz).
        m_order: The PSK order (2 for BPSK, 4 for QPSK, 8 for 8-PSK, ...).

    Returns:
        Estimated carrier frequency offset (Hz).
    """
    arr = np.asarray(signal, dtype=np.complex128)
    powered = arr**m_order
    spectrum = np.fft.fftshift(np.fft.fft(powered))
    freqs = np.fft.fftshift(np.fft.fftfreq(arr.size, d=1 / fs))
    peak_idx = int(np.argmax(np.abs(spectrum)))
    return float(freqs[peak_idx] / m_order)


def costas_loop_bpsk(
    signal: ArrayLike, loop_gain: float = 0.05,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Minimal first-order Costas loop for BPSK carrier phase recovery.

    Per-sample proportional feedback: `phase += loop_gain * sign(Re(c)) * Im(c)`,
    `c` being the phase-corrected sample. Like any BPSK Costas loop, the
    recovered phase is only known up to a 180-degree ambiguity (resolvable
    at a higher layer, e.g. via a known preamble or differential encoding).

    Args:
        signal: Complex baseband BPSK samples with a carrier phase/frequency
            offset to track out.
        loop_gain: Proportional feedback gain (larger converges faster but
            tracks noise more).

    Returns:
        `(corrected_signal, phase_estimate)`: the phase-corrected samples
        and the loop's phase estimate at each sample.
    """
    arr = np.asarray(signal, dtype=np.complex128)
    n = arr.size
    corrected = np.empty(n, dtype=np.complex128)
    phase_history = np.empty(n, dtype=np.float64)
    phase = 0.0
    for i in range(n):
        c = arr[i] * np.exp(-1j * phase)
        corrected[i] = c
        error = np.sign(c.real) * c.imag
        phase += loop_gain * error
        phase_history[i] = phase
    return corrected, phase_history


__all__ = ['costas_loop_bpsk', 'estimate_cfo_mth_power', 'gardner_timing_error']
