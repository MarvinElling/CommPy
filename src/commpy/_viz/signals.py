"""Signal- and modulation-domain plots: constellations, eyes, spectra, filters.

These are the plots you reach for while looking at a *waveform*: what the
symbols look like after the channel, how much inter-symbol interference the
pulse shaping left, where the signal sits in frequency, and how a filter shapes
it. Everything here takes raw arrays (or a `Modulator`), so it composes with any
part of the library that produces samples.
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from numpy.typing import ArrayLike, NDArray
from scipy.signal import welch

from commpy._modulation.base import Modulator
from commpy._viz.style import (
    AXIS_COLOR,
    SEQUENTIAL_CMAP,
    SERIES_COLORS,
    TEXT_MUTED,
    TEXT_SECONDARY,
    _apply_chrome,
    _axes,
    _finalize,
)

# Grid resolution per axis for decision-region boundaries. 320 keeps the
# boundary within a pixel of exact at typical figure sizes while staying well
# under a tenth of a second even for 256-ary constellations.
_REGION_GRID = 320


def _decision_boundaries(
    constellation: NDArray[np.complex128], limit: float,
) -> tuple[NDArray[np.float64], tuple[float, float, float, float]]:
    """Return a boundary mask of the nearest-neighbor decision regions.

    Args:
        constellation: Reference constellation points.
        limit: Half-width of the square region of the complex plane to cover.

    Returns:
        `(mask, extent)` where `mask` is 1.0 on region boundaries and NaN
        elsewhere (so it renders as lines over a transparent background), and
        `extent` is the `imshow` extent matching it.
    """
    axis = np.linspace(-limit, limit, _REGION_GRID)
    re, im = np.meshgrid(axis, axis)
    points = re + 1j * im
    nearest = np.argmin(np.abs(points[..., None] - constellation[None, None, :]), axis=-1)

    edge = np.zeros(nearest.shape, dtype=bool)
    edge[:, :-1] |= nearest[:, :-1] != nearest[:, 1:]
    edge[:-1, :] |= nearest[:-1, :] != nearest[1:, :]
    mask = np.where(edge, 1.0, np.nan)
    return mask, (-limit, limit, -limit, limit)


def plot_constellation(
    source: Modulator | ArrayLike,
    *,
    received: ArrayLike | None = None,
    labels: bool = False,
    regions: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot a constellation diagram, optionally over a cloud of received symbols.

    Args:
        source: A `Modulator` (its `constellation` and `bit_labels` are used) or
            an array of complex reference points. Note that the legacy
            per-scheme modulators (`BPSK_Modulator` and friends) carry no
            constellation attribute and are not accepted.
        received: Received/noisy symbols to scatter behind the reference points.
        labels: Annotate each reference point with its bit label. Requires a
            `Modulator` source. Readable up to about 16 points.
        regions: Draw the nearest-neighbor decision-region boundaries.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the constellation was drawn on.

    Raises:
        TypeError: If `labels` is requested for a non-`Modulator` source.
    """
    if isinstance(source, Modulator):
        points = source.constellation
        bit_labels = source.bit_labels
    else:
        points = np.asarray(source, dtype=np.complex128)
        bit_labels = None
        if labels:
            msg = 'labels=True needs a Modulator source (bit labels are unknown for raw points).'
            raise TypeError(msg)

    ax = _axes(ax, figsize=(5.5, 5.5))
    ax.set_aspect('equal')

    extent_source = points if received is None else np.concatenate(
        [points, np.asarray(received, dtype=np.complex128)],
    )
    limit = float(np.max(np.abs(np.concatenate([extent_source.real, extent_source.imag])))) * 1.25

    # The axis cross sits below the region boundaries: where the two coincide
    # (they do, for any symmetric constellation) the boundary must win, or the
    # centre divider reads as lighter than its siblings.
    ax.axhline(0.0, color=AXIS_COLOR, linewidth=0.8, zorder=0)
    ax.axvline(0.0, color=AXIS_COLOR, linewidth=0.8, zorder=0)

    if regions:
        mask, extent = _decision_boundaries(points, limit)
        ax.imshow(
            mask, extent=extent, origin='lower', cmap=ListedColormap([TEXT_MUTED]),
            interpolation='nearest', zorder=1,
        )

    if received is not None:
        rx = np.asarray(received, dtype=np.complex128)
        ax.scatter(
            rx.real, rx.imag, s=6, alpha=0.35, color=SERIES_COLORS[0],
            edgecolors='none', label='received', zorder=2,
        )
    ax.scatter(
        points.real, points.imag, s=70, color=SERIES_COLORS[1], marker='o',
        edgecolors=AXIS_COLOR, linewidths=1.5, label='constellation', zorder=3,
    )

    if labels and bit_labels is not None:
        offset = 0.045 * limit
        for point, bits in zip(points, bit_labels, strict=True):
            ax.annotate(
                ''.join(str(int(b)) for b in bits),
                (float(point.real), float(point.imag) + offset),
                ha='center', va='bottom', fontsize=8, color=TEXT_MUTED, zorder=4,
            )

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    _finalize(ax, title='Constellation', xlabel='In-phase', ylabel='Quadrature')
    if received is not None:
        # Below the axes: a square constellation fills its frame, so any in-frame
        # legend box would sit on top of symbols.
        ax.legend(
            loc='upper center', bbox_to_anchor=(0.5, -0.12), ncols=2,
            frameon=False, labelcolor=TEXT_SECONDARY,
        )
    return ax


def plot_eye_diagram(
    signal: ArrayLike,
    sps: int,
    *,
    n_traces: int = 200,
    offset: int = 0,
    ax: Axes | None = None,
) -> Axes:
    """Plot an eye diagram by overlaying two-symbol windows of a signal.

    The width of the open "eye" shows the timing margin, its height the noise
    margin; a closed eye means inter-symbol interference dominates.

    Args:
        signal: Oversampled baseband samples. Complex input draws I and Q
            separately.
        sps: Samples per symbol.
        n_traces: Maximum number of overlaid windows.
        offset: Sample offset of the first window, used to align the eye.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the eye diagram was drawn on.

    Raises:
        ValueError: If `sps < 2`, or the signal is too short for one window.
    """
    if sps < 2:
        msg = f'sps must be at least 2 to form an eye, got {sps}.'
        raise ValueError(msg)
    samples = np.asarray(signal)
    # 2*sps + 1 samples, not 2*sps: the extra endpoint makes each trace close at
    # exactly two symbol periods instead of stopping one sample short.
    window = 2 * sps + 1
    n_windows = (samples.size - offset - window) // sps + 1
    if n_windows < 1:
        msg = f'Signal too short: need at least {window + offset} samples for one window.'
        raise ValueError(msg)
    n_windows = min(n_windows, n_traces)

    starts = offset + sps * np.arange(n_windows)
    traces = samples[starts[:, None] + np.arange(window)[None, :]]
    x = np.arange(window) / sps

    ax = _axes(ax, figsize=(7.0, 4.5))
    is_complex = bool(np.iscomplexobj(samples))
    components: list[tuple[NDArray[np.float64], str]] = (
        [(traces.real, 'I'), (traces.imag, 'Q')]
        if is_complex
        else [(traces.astype(np.float64), '')]
    )
    for (values, name), color in zip(components, SERIES_COLORS, strict=False):
        segments = np.stack([np.broadcast_to(x, values.shape), values], axis=-1)
        ax.add_collection(
            LineCollection(list(segments), colors=color, linewidths=0.8, alpha=0.3),
        )
        if name:
            ax.plot([], [], color=color, label=name)

    # LineCollection does not drive autoscaling, so the limits are set by hand.
    drawn = np.concatenate([values.ravel() for values, _ in components])
    span_y = float(np.ptp(drawn))
    margin = 0.1 * span_y if span_y > 0 else 0.1
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(float(np.min(drawn)) - margin, float(np.max(drawn)) + margin)
    return _finalize(
        ax, title='Eye diagram', xlabel='Time (symbol periods)', ylabel='Amplitude',
        legend=is_complex,
    )


def plot_psd(
    signal: ArrayLike,
    fs: float,
    *,
    nperseg: int = 256,
    ax: Axes | None = None,
) -> Axes:
    """Plot the two-sided power spectral density of a baseband signal.

    Uses Welch's method and always returns a two-sided, zero-centered spectrum:
    complex baseband is not symmetric about DC, so folding it would discard
    information.

    Args:
        signal: Baseband samples.
        fs: Sampling rate in Hz.
        nperseg: Welch segment length; clamped to the signal length.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the spectrum was drawn on.
    """
    samples = np.asarray(signal)
    freqs, psd = welch(
        samples, fs=fs, nperseg=min(nperseg, samples.size), return_onesided=False,
    )
    order = np.argsort(freqs)
    freqs, psd = freqs[order], psd[order]

    ax = _axes(ax, figsize=(7.0, 4.5))
    ax.plot(freqs, 10 * np.log10(np.maximum(psd, np.finfo(float).tiny)), color=SERIES_COLORS[0])
    return _finalize(ax, title='Power spectral density', xlabel='Frequency (Hz)',
                     ylabel='PSD (dB/Hz)')


def plot_spectrogram(
    signal: ArrayLike,
    fs: float,
    *,
    nperseg: int = 256,
    ax: Axes | None = None,
) -> Axes:
    """Plot a spectrogram showing how the signal's spectrum evolves over time.

    Args:
        signal: Baseband samples.
        fs: Sampling rate in Hz.
        nperseg: FFT length per time slice; clamped to the signal length.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the spectrogram was drawn on.
    """
    samples = np.asarray(signal)
    nfft = min(nperseg, samples.size)

    ax = _axes(ax, figsize=(7.0, 4.5))
    _, _, _, image = ax.specgram(
        samples, NFFT=nfft, Fs=fs, noverlap=nfft // 2, cmap=SEQUENTIAL_CMAP,
        sides='twosided' if np.iscomplexobj(samples) else 'onesided',
    )
    bar = ax.figure.colorbar(image, ax=ax)
    bar.set_label('Power (dB)', color=TEXT_MUTED)
    return _finalize(ax, title='Spectrogram', xlabel='Time (s)', ylabel='Frequency (Hz)',
                     grid=False)


def plot_iq_time(
    signal: ArrayLike,
    fs: float = 1.0,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot the in-phase and quadrature components of a signal over time.

    Args:
        signal: Baseband samples. Real input is drawn as a single trace.
        fs: Sampling rate in Hz; the time axis is samples/`fs`.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the waveform was drawn on.
    """
    samples = np.asarray(signal)
    t = np.arange(samples.size) / fs

    ax = _axes(ax, figsize=(8.0, 3.5))
    is_complex = np.iscomplexobj(samples)
    if is_complex:
        ax.plot(t, samples.real, color=SERIES_COLORS[0], label='I')
        ax.plot(t, samples.imag, color=SERIES_COLORS[1], label='Q')
    else:
        ax.plot(t, samples, color=SERIES_COLORS[0])
    return _finalize(ax, title='Baseband I/Q', xlabel='Time (s)', ylabel='Amplitude',
                     legend=is_complex)


def _impulse_response(
    filt: Callable[[NDArray[np.float64]], NDArray[np.float64]] | ArrayLike,
    fs: float,
    symbol_period: float | None,
    span: int | None,
) -> NDArray[np.float64]:
    """Sample a filter given either as taps or as a causal pulse-shape callable."""
    if not callable(filt):
        return np.asarray(filt, dtype=np.float64)
    if symbol_period is None or span is None:
        msg = 'symbol_period and span are required when filt is a pulse-shape callable.'
        raise ValueError(msg)
    tau = np.arange(0.0, span * symbol_period, 1.0 / fs)
    return np.asarray(filt(tau), dtype=np.float64)


def _draw_impulse(ax: Axes, h: NDArray[np.float64], fs: float) -> None:
    """Draw a filter's impulse response into `ax`."""
    ax.plot(np.arange(h.size) / fs, h, color=SERIES_COLORS[0])
    _finalize(ax, title='Impulse response', xlabel='Time (s)', ylabel='Amplitude')


def _draw_magnitude(ax: Axes, h: NDArray[np.float64], fs: float) -> None:
    """Draw a filter's normalized magnitude response into `ax`."""
    n_fft = max(1024, 4 * h.size)
    spectrum = np.fft.fftshift(np.fft.fft(h, n=n_fft))
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / fs))
    magnitude = np.abs(spectrum)
    magnitude_db = 20 * np.log10(
        np.maximum(magnitude / np.max(magnitude), np.finfo(float).tiny),
    )
    ax.plot(freqs, magnitude_db, color=SERIES_COLORS[1])
    ax.set_ylim(-80.0, 5.0)
    _finalize(ax, title='Magnitude response', xlabel='Frequency (Hz)', ylabel='Magnitude (dB)')


def plot_filter_response(  # noqa: PLR0913 -- taps/callable input needs its own sampling parameters
    filt: Callable[[NDArray[np.float64]], NDArray[np.float64]] | ArrayLike,
    *,
    fs: float = 1.0,
    symbol_period: float | None = None,
    span: int | None = None,
    domain: str = 'both',
    ax: Axes | None = None,
) -> Axes:
    """Plot a filter's impulse response, its magnitude response, or both.

    Accepts either a tap array or one of the causal pulse-shape callables from
    `raised_cosine_filter`/`root_raised_cosine_filter` -- for a callable, pass
    the same `symbol_period` and `span` used to build it.

    The two domains are drawn as separate stacked panels rather than sharing one
    axes with two scales; a second y-axis on the same frame invites reading a
    crossing point that carries no meaning.

    Args:
        filt: Filter taps, or a callable `g(tau)` defined on `[0, span*T)`.
        fs: Sampling rate in Hz.
        symbol_period: Symbol period `T`; required for a callable.
        span: Truncation window in multiples of `T`; required for a callable.
        domain: `'time'`, `'freq'`, or `'both'` (stacked panels).
        ax: Axes to draw into; only valid for a single domain.

    Returns:
        The axes drawn on. For `domain='both'` this is the impulse-response
        panel; the magnitude panel is `ax.figure.axes[1]`.

    Raises:
        ValueError: If `domain` is unknown, if an explicit `ax` is combined with
            `domain='both'`, or if a callable is passed without
            `symbol_period`/`span`.
    """
    if domain not in {'time', 'freq', 'both'}:
        msg = f"domain must be 'time', 'freq', or 'both', got {domain!r}."
        raise ValueError(msg)
    if domain == 'both' and ax is not None:
        msg = "domain='both' draws two panels and cannot reuse a single supplied ax."
        raise ValueError(msg)

    h = _impulse_response(filt, fs, symbol_period, span)

    if domain == 'both':
        fig, panels = plt.subplots(2, 1, figsize=(7.0, 6.0))
        time_ax: Axes = panels[0]
        freq_ax: Axes = panels[1]
        _apply_chrome(time_ax)
        _apply_chrome(freq_ax)
        _draw_impulse(time_ax, h, fs)
        _draw_magnitude(freq_ax, h, fs)
        fig.tight_layout()
        return time_ax

    target = _axes(ax, figsize=(7.0, 4.0))
    if domain == 'time':
        _draw_impulse(target, h, fs)
    else:
        _draw_magnitude(target, h, fs)
    return target


__all__ = [
    'plot_constellation',
    'plot_eye_diagram',
    'plot_filter_response',
    'plot_iq_time',
    'plot_psd',
    'plot_spectrogram',
]
