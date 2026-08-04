"""System- and channel-level plots: link performance, channels, OFDM, MIMO.

These sit one level above the waveform: they characterize a *link* rather than
a signal. The error-rate comparison builds on `plot_waterfall`, so a curve
drawn here and one drawn there are the same object with the same confidence
intervals.
"""

from collections.abc import Mapping

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray

from commpy._informationTheory.formulas import channel_capacity_awgn, channel_capacity_bsc
from commpy._mimo.capacity import mimo_capacity
from commpy._mimo.channel import rayleigh_channel_matrix
from commpy._ofdm.ofdm import papr_ccdf
from commpy._simulation.link_simulation import SimulationResult, plot_waterfall
from commpy._viz.signals import _draw_magnitude
from commpy._viz.style import (
    GRID_COLOR,
    SEQUENTIAL_CMAP,
    SERIES_COLORS,
    TEXT_MUTED,
    _apply_chrome,
    _axes,
    _finalize,
    series_colors,
)


def plot_error_rate_comparison(
    results: Mapping[str, SimulationResult],
    *,
    theoretical: Mapping[str, ArrayLike] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot several error-rate curves on one waterfall, with confidence intervals.

    Each curve is drawn by `plot_waterfall`, so the error bars are the same
    Wilson-score intervals a single-curve plot would show -- which matters when
    comparing codes, because two curves whose intervals overlap have not
    actually been separated by the simulation.

    Args:
        results: Mapping of label to `SimulationResult`, e.g.
            `{'uncoded': ..., 'LDPC': ...}`. Colors are assigned in mapping
            order and stay attached to a label.
        theoretical: Optional mapping of label to a closed-form reference curve
            evaluated at that result's SNR points.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the curves were drawn on.

    Raises:
        ValueError: If `results` is empty.
    """
    if not results:
        msg = 'results must contain at least one curve.'
        raise ValueError(msg)

    ax = _axes(ax, figsize=(7.5, 5.5))
    colors = series_colors(len(results))
    for (label, result), color in zip(results.items(), colors, strict=True):
        plot_waterfall(result, ax=ax, label=label, color=color)
        if theoretical is not None and label in theoretical:
            ax.plot(
                result.snr_db, np.asarray(theoretical[label], dtype=np.float64),
                '--', color=color, alpha=0.7, linewidth=1.2,
            )

    _finalize(
        ax, title='Error-rate comparison', xlabel='SNR (dB)', ylabel='Error rate',
        legend=True,
    )
    # plot_waterfall turns on decade-minor gridlines; mute them to match, or they
    # read as data on a log axis.
    ax.grid(True, which='minor', color=GRID_COLOR, linewidth=0.4)
    return ax


def plot_channel_response(
    taps: ArrayLike,
    *,
    fs: float = 1.0,
    ax: Axes | None = None,
) -> Axes:
    """Plot a channel's impulse response and the magnitude response it implies.

    Deep notches in the magnitude response are what an equalizer has to undo,
    and are where a zero-forcing design will amplify noise the most.

    Args:
        taps: Channel impulse response, e.g. `[1.0, 0.3, -0.15]`. CommPy has no
            tapped-delay-line channel model, so these come from you -- the
            fading models in `Channels` are flat and have no delay profile.
        fs: Sampling rate in Hz.
        ax: Axes to draw into; only the magnitude panel is drawn when supplied,
            since one axes cannot carry both domains honestly.

    Returns:
        The axes drawn on. With a new figure this is the impulse-response
        panel and `ax.figure.axes[1]` is the magnitude panel.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 -- only needed for the two-panel layout

    h = np.asarray(taps, dtype=np.float64)

    if ax is not None:
        _draw_magnitude(ax, h, fs)
        return ax

    fig, panels = plt.subplots(2, 1, figsize=(7.0, 6.0))
    time_ax: Axes = panels[0]
    freq_ax: Axes = panels[1]
    _apply_chrome(time_ax)
    _apply_chrome(freq_ax)
    time_ax.stem(np.arange(h.size) / fs, h, linefmt=SERIES_COLORS[0],
                 markerfmt='o', basefmt=TEXT_MUTED)
    _finalize(time_ax, title='Channel impulse response', xlabel='Delay (s)',
              ylabel='Amplitude')
    _draw_magnitude(freq_ax, h, fs)
    fig.tight_layout()
    return time_ax


def plot_equalizer_response(
    channel_taps: ArrayLike,
    eq_taps: ArrayLike,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot channel, equalizer, and combined magnitude responses on one axes.

    A perfect equalizer would make the combined response flat; how far it
    deviates is the residual inter-symbol interference, reported in the title as
    the peak-to-total energy ratio of the combined impulse response.

    Note this is a *static* design plot. The equalizers in CommPy
    (`zf_equalizer`, `mmse_equalizer`) solve for taps against a known channel;
    there is no adaptive LMS/RLS equalizer, so there is no convergence
    trajectory to draw.

    Args:
        channel_taps: Channel impulse response.
        eq_taps: Equalizer taps, e.g. from `zf_equalizer` or `mmse_equalizer`.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the responses were drawn on.
    """
    channel = np.asarray(channel_taps, dtype=np.float64)
    equalizer = np.asarray(eq_taps, dtype=np.float64)
    combined = np.convolve(channel, equalizer)

    n_fft = 1024
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft))

    ax = _axes(ax, figsize=(7.5, 4.5))
    responses = [
        ('channel', channel), ('equalizer', equalizer), ('combined', combined),
    ]
    for (label, taps), color in zip(responses, series_colors(3), strict=True):
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(taps, n=n_fft)))
        ax.plot(
            freqs, 20 * np.log10(np.maximum(spectrum, np.finfo(float).tiny)),
            color=color, label=label,
        )

    peak = float(np.max(combined**2))
    residual_isi = 1.0 - peak / float(np.sum(combined**2))
    return _finalize(
        ax, title=f'Equalizer response (residual ISI {residual_isi:.1%})',
        xlabel='Normalized frequency', ylabel='Magnitude (dB)', legend=True,
    )


def plot_ofdm_grid(
    grid: ArrayLike,
    *,
    active_subcarriers: ArrayLike | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot the magnitude of an OFDM resource grid over symbol and subcarrier.

    Args:
        grid: Shape `(n_symbols, n_subcarriers)` of frequency-domain values,
            as `OFDMDemodulator.demodulate` produces per symbol.
        active_subcarriers: Optional indices of the occupied subcarriers (an
            `OFDMModulator.active_subcarriers`); everything else is masked out
            so guard bands read as empty rather than as zero-power signal.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the grid was drawn on.
    """
    values = np.abs(np.asarray(grid, dtype=np.complex128))
    if active_subcarriers is not None:
        mask = np.ones(values.shape[1], dtype=bool)
        mask[np.asarray(active_subcarriers, dtype=np.int64)] = False
        values = np.where(mask[None, :], np.nan, values)

    ax = _axes(ax, figsize=(7.5, 4.5))
    image = ax.imshow(
        values.T, cmap=SEQUENTIAL_CMAP, origin='lower', aspect='auto',
        interpolation='nearest',
    )
    bar = ax.figure.colorbar(image, ax=ax)
    bar.set_label('Magnitude', color=TEXT_MUTED)
    return _finalize(
        ax, title='OFDM resource grid', xlabel='OFDM symbol', ylabel='Subcarrier',
        grid=False,
    )


def plot_papr_ccdf(
    ofdm_symbols: ArrayLike,
    *,
    thresholds_db: ArrayLike | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot the complementary CDF of an OFDM signal's peak-to-average power ratio.

    Read it as "the fraction of symbols whose PAPR exceeds x dB" -- the tail is
    what sets how much back-off a power amplifier needs.

    Args:
        ofdm_symbols: Shape `(n_symbols, n_fft)`, one time-domain OFDM symbol
            per row (no cyclic prefix), as `papr_ccdf` expects.
        thresholds_db: PAPR thresholds in dB; defaults to 0..14 dB.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the CCDF was drawn on.
    """
    thresholds = (
        np.linspace(0.0, 14.0, 57) if thresholds_db is None
        else np.asarray(thresholds_db, dtype=np.float64)
    )
    ccdf = papr_ccdf(ofdm_symbols, thresholds)

    ax = _axes(ax, figsize=(7.0, 4.5))
    ax.semilogy(thresholds, np.maximum(ccdf, np.finfo(float).tiny), color=SERIES_COLORS[0])
    ax.set_ylim(bottom=max(1.0 / np.asarray(ofdm_symbols).shape[0] / 2, 1e-6), top=1.5)
    return _finalize(
        ax, title='PAPR complementary CDF', xlabel='PAPR threshold (dB)',
        ylabel='P(PAPR > threshold)',
    )


def _capacity_cdf(
    n_rx: int, n_tx: int, snr_db: float, n_realizations: int, rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Return sorted capacities over independent Rayleigh channel draws."""
    capacities = [
        mimo_capacity(rayleigh_channel_matrix(n_rx, n_tx, rng=rng), snr_db)
        for _ in range(n_realizations)
    ]
    return np.sort(np.asarray(capacities, dtype=np.float64))


def plot_mimo_capacity_cdf(  # noqa: PLR0913 -- antenna counts, SNR, sample size, and seed are all independent
    n_tx: int,
    n_rx: int,
    snr_db: float | ArrayLike,
    *,
    n_realizations: int = 1000,
    rng: np.random.Generator | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot the CDF of MIMO capacity over random Rayleigh channel realizations.

    A fading MIMO channel has no single capacity, it has a distribution. The
    steepness of the CDF is the spatial diversity: more antennas make the curve
    stand up, so the *outage* capacity at the low percentiles rises even when
    the mean barely moves.

    Args:
        n_tx: Number of transmit antennas.
        n_rx: Number of receive antennas.
        snr_db: Total SNR in dB; pass a sequence to overlay several SNRs.
        n_realizations: Independent channel draws per SNR.
        rng: Optional `np.random.Generator` for reproducibility.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the CDFs were drawn on.
    """
    if rng is None:
        rng = np.random.default_rng()
    snr_values = np.atleast_1d(np.asarray(snr_db, dtype=np.float64))
    probabilities = (np.arange(n_realizations) + 0.5) / n_realizations

    ax = _axes(ax, figsize=(7.0, 4.5))
    for snr, color in zip(snr_values, series_colors(snr_values.size), strict=True):
        capacities = _capacity_cdf(n_rx, n_tx, float(snr), n_realizations, rng)
        ax.plot(capacities, probabilities, color=color, label=f'{snr:g} dB')

    ax.set_ylim(0.0, 1.0)
    return _finalize(
        ax, title=f'MIMO capacity CDF ({n_tx}x{n_rx}, Rayleigh)',
        xlabel='Capacity (bits/channel use)', ylabel='P(capacity <= x)',
        legend=snr_values.size > 1,
    )


def plot_capacity_curves(
    *,
    snr_db_range: ArrayLike | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot the AWGN and binary-symmetric-channel capacity limits.

    Overlay a measured throughput on this and the vertical gap is how far the
    scheme sits from the Shannon limit.

    The BSC curve is drawn against the crossover probability a hard-decision
    BPSK receiver would suffer at each SNR, so both curves share one x-axis
    honestly rather than needing a second scale -- and the gap between them is
    exactly the cost of throwing away soft information.

    Args:
        snr_db_range: SNR points in dB; defaults to -10..20 dB.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the capacity curves were drawn on.
    """
    from scipy.special import erfc  # noqa: PLC0415 -- only this function needs it

    snr_db = (
        np.linspace(-10.0, 20.0, 121) if snr_db_range is None
        else np.asarray(snr_db_range, dtype=np.float64)
    )
    snr_linear = 10.0 ** (snr_db / 10.0)

    awgn = np.array([channel_capacity_awgn(float(s)) for s in snr_linear])
    crossover = 0.5 * erfc(np.sqrt(snr_linear))
    bsc = np.array([channel_capacity_bsc(float(p)) for p in crossover])

    ax = _axes(ax, figsize=(7.0, 4.5))
    for (label, curve), color in zip(
        [('AWGN (soft decisions)', awgn), ('BSC (hard decisions)', bsc)],
        series_colors(2), strict=True,
    ):
        ax.plot(snr_db, curve, color=color, label=label)

    return _finalize(
        ax, title='Channel capacity limits', xlabel='SNR (dB)',
        ylabel='Capacity (bits/channel use)', legend=True,
    )


__all__ = [
    'plot_capacity_curves',
    'plot_channel_response',
    'plot_equalizer_response',
    'plot_error_rate_comparison',
    'plot_mimo_capacity_cdf',
    'plot_ofdm_grid',
    'plot_papr_ccdf',
]
