"""Monte-Carlo link-level error-rate simulation (BER/FER/SER) with confidence intervals.

`simulate_error_rate` is the generic core: it drives an arbitrary
"run N trials at this SNR, count errors" function with early stopping, so it
is equally usable for bit-, frame-, or symbol-error rates. `simulate_ber` is
a convenience wrapper for the common modulate -> channel -> demodulate case.
`plot_waterfall` renders the resulting curve.
"""

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from commpy._modulation.base import Modulator


@dataclass(frozen=True)
class SimulationResult:
    """Per-SNR-point results of a Monte-Carlo error-rate simulation."""

    snr_db: NDArray[np.float64]
    error_rate: NDArray[np.float64]
    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]
    n_trials: NDArray[np.int64]
    n_errors: NDArray[np.int64]


def _wilson_interval(n_errors: int, n_trials: int, confidence: float) -> tuple[float, float]:
    """Wilson score confidence interval for a Bernoulli error-rate estimate.

    Preferred over the normal approximation here because it stays within
    [0, 1] and remains well-behaved when `n_errors` is small (e.g. right at
    the `target_errors` threshold), which the normal approximation is not.
    """
    if n_trials == 0:
        return 0.0, 1.0
    z = float(norm.ppf(0.5 + confidence / 2.0))
    p_hat = n_errors / n_trials
    denom = 1.0 + z**2 / n_trials
    center = (p_hat + z**2 / (2 * n_trials)) / denom
    half_width = (
        (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n_trials + z**2 / (4 * n_trials**2))
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def simulate_error_rate(  # noqa: PLR0913 -- each parameter controls a distinct stopping/estimation criterion
    trial_fn: Callable[[float, np.random.Generator, int], tuple[int, int]],
    snr_db_range: ArrayLike,
    *,
    target_errors: int = 100,
    max_trials: int = 10_000_000,
    trials_per_batch: int = 10_000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Run a Monte-Carlo error-rate sweep over a range of SNR points.

    For each SNR point, `trial_fn` is called in batches until either
    `target_errors` errors have been observed or `max_trials` trials have
    run, whichever comes first. This is the standard early-stopping strategy
    for Monte-Carlo BER/FER simulation: it bounds the estimator's relative
    variance without spending a fixed, large trial budget at every SNR
    point when high-SNR points would otherwise converge almost immediately.

    Args:
        trial_fn: `(snr_db, rng, n_trials) -> (n_errors, n_trials_run)`. Runs
            up to `n_trials` independent trials (bits, frames, symbols, ...)
            at the given SNR and returns how many were in error and how many
            were actually run. Returning `n_trials_run < n_trials` (e.g. to
            respect an internal batch granularity) is supported; returning 0
            ends the sweep for that SNR point early.
        snr_db_range: SNR points (dB) to evaluate.
        target_errors: Stop a given SNR point once this many errors have
            been observed.
        max_trials: Hard cap on trials per SNR point, in case `target_errors`
            is never reached (e.g. at very high SNR).
        trials_per_batch: Number of trials requested per call to `trial_fn`.
        confidence: Confidence level for the Wilson score interval on the
            estimated error rate, e.g. 0.95 for a 95% CI.
        rng: Optional `np.random.Generator` for reproducibility.

    Returns:
        A `SimulationResult` with one entry per SNR point.
    """
    if rng is None:
        rng = np.random.default_rng()
    snr_points = np.asarray(snr_db_range, dtype=np.float64)

    error_rate = np.empty(snr_points.shape, dtype=np.float64)
    ci_lower = np.empty(snr_points.shape, dtype=np.float64)
    ci_upper = np.empty(snr_points.shape, dtype=np.float64)
    n_trials_arr = np.empty(snr_points.shape, dtype=np.int64)
    n_errors_arr = np.empty(snr_points.shape, dtype=np.int64)

    for i, snr_db in enumerate(snr_points):
        total_errors = 0
        total_trials = 0
        while total_errors < target_errors and total_trials < max_trials:
            batch = min(trials_per_batch, max_trials - total_trials)
            errors, run = trial_fn(float(snr_db), rng, batch)
            total_errors += errors
            total_trials += run
            if run == 0:
                break  # trial_fn made no progress; avoid spinning forever.

        error_rate[i] = total_errors / total_trials if total_trials > 0 else 0.0
        ci_lower[i], ci_upper[i] = _wilson_interval(total_errors, total_trials, confidence)
        n_trials_arr[i] = total_trials
        n_errors_arr[i] = total_errors

    return SimulationResult(
        snr_db=snr_points,
        error_rate=error_rate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_trials=n_trials_arr,
        n_errors=n_errors_arr,
    )


_ChannelFn = Callable[[NDArray[np.complex128], float, np.random.Generator], NDArray[np.complex128]]


def simulate_ber(  # noqa: PLR0913 -- each parameter controls a distinct stopping/estimation criterion
    modulator: Modulator,
    channel_fn: _ChannelFn,
    snr_db_range: ArrayLike,
    *,
    bits_per_batch: int = 10_000,
    target_errors: int = 100,
    max_trials: int = 10_000_000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Monte-Carlo bit-error-rate sweep for a `Modulator` over a channel.

    Convenience wrapper around `simulate_error_rate` for the common
    modulate -> channel -> demodulate -> count-bit-errors case.

    Args:
        modulator: Any `Modulator` (e.g. `MQAMModulator(16)`).
        channel_fn: `(symbols, snr_db, rng) -> received_symbols`, called
            positionally so `Channels.awgn`/`Channels.rayleigh` work
            directly; channels with extra parameters (e.g. `Channels.rician`)
            need a small wrapper, e.g. `lambda x, s, r: Channels.rician(x, s, rng=r)`.
        snr_db_range: SNR points (dB) to evaluate.
        bits_per_batch: Number of bits simulated per trial batch (rounded
            down to a multiple of `modulator.bits_per_symbol`).
        target_errors: See `simulate_error_rate`.
        max_trials: See `simulate_error_rate` (counted in bits).
        confidence: See `simulate_error_rate`.
        rng: Optional `np.random.Generator` for reproducibility.

    Returns:
        A `SimulationResult` in bits (`n_trials`/`n_errors` are bit counts).
    """
    bits_per_symbol = modulator.bits_per_symbol
    n_bits = max(bits_per_symbol, (bits_per_batch // bits_per_symbol) * bits_per_symbol)

    def trial_fn(snr_db: float, trial_rng: np.random.Generator, n_trials: int) -> tuple[int, int]:
        batch_bits = (n_trials // bits_per_symbol) * bits_per_symbol
        if batch_bits == 0:
            return 0, 0
        bits = trial_rng.integers(0, 2, batch_bits)
        symbols = modulator.modulate(bits)
        received = channel_fn(symbols, snr_db, trial_rng)
        recovered = modulator.demodulate(received)
        errors = int(np.sum(recovered != bits))
        return errors, batch_bits

    return simulate_error_rate(
        trial_fn,
        snr_db_range,
        target_errors=target_errors,
        max_trials=max_trials,
        trials_per_batch=n_bits,
        confidence=confidence,
        rng=rng,
    )


def plot_waterfall(
    result: SimulationResult,
    theoretical: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot a BER/FER-vs-SNR waterfall curve with confidence-interval error bars.

    Args:
        result: A `SimulationResult` from `simulate_error_rate`/`simulate_ber`.
        theoretical: Optional `snr_db -> error_rate` closed-form reference
            curve, overlaid for comparison.
        ax: Optional axes to plot into; a new figure/axes is created if omitted.

    Returns:
        The axes the curve was plotted on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    lower_err = result.error_rate - result.ci_lower
    upper_err = result.ci_upper - result.error_rate
    ax.errorbar(
        result.snr_db, result.error_rate, yerr=np.vstack([lower_err, upper_err]),
        fmt='o-', capsize=3, label='measured',
    )
    if theoretical is not None:
        ax.plot(result.snr_db, theoretical(result.snr_db), '--', label='theoretical')
    ax.set_yscale('log')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Error rate')
    ax.set_title('Waterfall curve')
    ax.grid(True, which='both')
    ax.legend()
    return ax


__all__ = ['SimulationResult', 'plot_waterfall', 'simulate_ber', 'simulate_error_rate']
