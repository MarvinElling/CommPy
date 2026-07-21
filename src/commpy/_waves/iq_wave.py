"""IQWaveform class for generating and plotting IQ modulated waveforms."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray


class IQWaveform:
    """Generates and plots a pulse-shaped, optionally up-converted IQ waveform."""

    def __init__(  # noqa: PLR0913 -- each parameter is a distinct, physically meaningful waveform quantity
        self,
        I: ArrayLike,  # noqa: E741 -- "I" (in-phase) is the standard DSP/RF term of art
        Q: ArrayLike,
        T: float,
        fs: float,
        f0: float = 0.0,
        pulse_shape: Callable[[np.ndarray], np.ndarray] | None = None,
        span: int = 4,
    ) -> None:
        """Construct an IQ waveform from a symbol sequence.

        Args:
            I: In-phase symbol values (real numbers), e.g. `I = syms.real`.
            Q: Quadrature symbol values (real numbers), e.g. `Q = syms.imag`.
            T: Symbol period (seconds).
            fs: Sample rate (Hz).
            f0: Carrier frequency (Hz). If 0, output is baseband only.
            pulse_shape: Callable `tau -> g(tau)` giving the pulse shape. Defaults
                to a rectangular pulse.
            span: Pulse truncation window in multiples of `T`, for computational
                efficiency.
        """
        self.I = np.asarray(I)
        self.Q = np.asarray(Q)
        self.T = T
        self.fs = fs
        self.f0 = f0
        self.span = span
        self.N = len(self.I)
        self.pulse_shape = pulse_shape if pulse_shape is not None else self._rect_pulse
        self.t, self.s, self.s_I, self.s_Q = self._make_waveform()

    def _rect_pulse(self, tau: np.ndarray) -> np.ndarray:
        return ((tau >= 0) & (tau < self.T)).astype(float)

    def _make_waveform(
        self,
    ) -> tuple[
        NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
    ]:
        n_symbols = self.N
        symbol_period = self.T
        fs = self.fs
        span_window = self.span

        n_samples = int(n_symbols * symbol_period * fs)
        t = np.arange(n_samples) / fs

        s_i = np.zeros_like(t)
        s_q = np.zeros_like(t)

        # Synthesize s_I(t) and s_Q(t) as shaped pulse trains.
        for n, (i_n, q_n) in enumerate(zip(self.I, self.Q, strict=True)):
            tn = n * symbol_period
            # Only non-zero within [tn, tn + span*T].
            mask = (t >= tn) & (t < tn + span_window * symbol_period)
            tau = t[mask] - tn
            gs = self.pulse_shape(tau)
            s_i[mask] += i_n * gs
            s_q[mask] += q_n * gs

        # Bandpass: s(t) = s_I(t) * sqrt(2)*cos(2*pi*f0*t) - s_Q(t) * sqrt(2)*sin(2*pi*f0*t).
        carrier_cos = np.sqrt(2) * np.cos(2 * np.pi * self.f0 * t)
        carrier_sin = np.sqrt(2) * np.sin(2 * np.pi * self.f0 * t)
        s = s_i * carrier_cos - s_q * carrier_sin
        return t, s, s_i, s_q

    def plot_waveform(self) -> None:
        r"""Plot the analog transmit signal \tilde{S}(t)."""
        plt.figure(figsize=(10, 3))
        plt.plot(self.t, self.s, lw=1.2)
        plt.xlabel('t [s]')
        plt.ylabel('$\\tilde{S}(t)$')
        plt.title('Bandpass IQ Modulated Signal')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_IQ_baseband(self) -> None:  # noqa: N802 -- public API name kept for backward compatibility
        """Plot baseband I/Q components as a function of time."""
        plt.figure(figsize=(10, 4))
        plt.plot(self.t, self.s_I, label='I(t)', lw=1.2)
        plt.plot(self.t, self.s_Q, label='Q(t)', lw=1.2)
        plt.xlabel('t [s]')
        plt.ylabel('Amplitude')
        plt.title('Baseband I/Q Components (Pulse Shaped)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_eye(self, n_traces: int = 2) -> None:
        """Plot an eye diagram for I(t).

        Args:
            n_traces: Number of trailing symbol periods to omit from the plot
                (kept for consistency with earlier trace windows).
        """
        period = int(self.T * self.fs)
        plt.figure()
        for k in range(len(self.s_I) // period - n_traces):
            plt.plot(np.arange(period) / self.fs, self.s_I[k * period:(k + 1) * period], 'b')
        plt.xlabel('t [symbol period]')
        plt.title('Eye diagram (I)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# %%
# Example usage:
if __name__ == '__main__':
    from commpy._modulation.legacy import OOK_Modulator

    _rng = np.random.default_rng()
    bits = _rng.integers(0, 2, 16)
    syms = OOK_Modulator.modulate(bits)
    I = syms.real  # noqa: E741 -- "I" (in-phase) is the standard DSP/RF term of art
    Q = syms.imag
    T = 1e-3
    fs = 300000
    f0 = 20000

    tx = IQWaveform(I, Q, T, fs, f0)  # Default: rectangular pulse
    tx.plot_waveform()
    tx.plot_IQ_baseband()
    tx.plot_eye()
