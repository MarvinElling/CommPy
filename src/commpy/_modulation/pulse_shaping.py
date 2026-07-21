"""Raised-cosine (RC) and root-raised-cosine (RRC) pulse-shaping filters.

Both factories return a *causal* callable `g(tau)` for `tau` in
`[0, span*T)`, peaking at `tau = span*T/2` -- exactly the convention
`IQWaveform`'s `pulse_shape` parameter expects (it evaluates the callable on
`tau = t - symbol_start_time`, always `>= 0`). Internally this is just the
textbook (symmetric, zero-centered) RC/RRC impulse response shifted by
`span*T/2`.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def _raised_cosine(
    t: NDArray[np.float64], symbol_period: float, rolloff: float,
) -> NDArray[np.float64]:
    """Symmetric (zero-centered) raised-cosine impulse response."""
    g = np.sinc(t / symbol_period) * np.cos(np.pi * rolloff * t / symbol_period)
    if rolloff > 0:
        denom = 1 - (2 * rolloff * t / symbol_period) ** 2
        singular = np.isclose(denom, 0.0)
        safe_denom = np.where(singular, 1.0, denom)
        g = g / safe_denom
        limit_val = (np.pi / 4) * np.sinc(1 / (2 * rolloff))
        g = np.where(singular, limit_val, g)
    return g


def _root_raised_cosine(
    t: NDArray[np.float64], symbol_period: float, rolloff: float,
) -> NDArray[np.float64]:
    """Symmetric (zero-centered) root-raised-cosine impulse response (unit energy)."""
    if rolloff == 0:
        return np.asarray(np.sinc(t / symbol_period) / np.sqrt(symbol_period))

    x = t / symbol_period
    num = np.sin(np.pi * x * (1 - rolloff)) + 4 * rolloff * x * np.cos(np.pi * x * (1 + rolloff))
    denom = np.pi * x * (1 - (4 * rolloff * x) ** 2)

    zero_t = np.isclose(t, 0.0)
    singular = np.isclose(np.abs(4 * rolloff * x), 1.0) & ~zero_t
    safe_denom = np.where(zero_t | singular, 1.0, denom)
    h = num / safe_denom

    h_at_zero = 1 - rolloff + 4 * rolloff / np.pi
    h_at_singularity = (rolloff / np.sqrt(2)) * (
        (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
    )
    h = np.where(zero_t, h_at_zero, h)
    h = np.where(singular, h_at_singularity, h)
    return np.asarray(h / np.sqrt(symbol_period))


def raised_cosine_filter(
    symbol_period: float, rolloff: float, span: int,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Build a causal raised-cosine pulse-shape callable for `IQWaveform(pulse_shape=...)`.

    The raised-cosine pulse satisfies the Nyquist zero-ISI criterion:
    `g(k*T) == 0` for every nonzero integer `k`, so symbol-spaced sampling at
    the peak sees no inter-symbol interference from neighboring symbols.

    Args:
        symbol_period: Symbol period `T`.
        rolloff: Roll-off factor `beta` in `[0, 1]`.
        span: Truncation window in multiples of `T` (matches `IQWaveform`'s
            own `span` parameter -- pass the same value to both).

    Raises:
        ValueError: If `rolloff` is not in `[0, 1]`.
    """
    if not 0 <= rolloff <= 1:
        msg = f'rolloff must be in [0, 1], got {rolloff}.'
        raise ValueError(msg)
    center = span * symbol_period / 2

    def pulse(tau: NDArray[np.float64]) -> NDArray[np.float64]:
        return _raised_cosine(np.asarray(tau, dtype=np.float64) - center, symbol_period, rolloff)

    return pulse


def root_raised_cosine_filter(
    symbol_period: float, rolloff: float, span: int,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Build a causal root-raised-cosine pulse-shape callable for `IQWaveform(pulse_shape=...)`.

    A matched pair of RRC filters (transmit + receive matched filter) has a
    combined response equal to the raised-cosine pulse, achieving zero-ISI
    while splitting the filtering gain evenly and matched-filtering for
    optimal noise performance.

    Args:
        symbol_period: Symbol period `T`.
        rolloff: Roll-off factor `beta` in `[0, 1]`.
        span: Truncation window in multiples of `T`.

    Raises:
        ValueError: If `rolloff` is not in `[0, 1]`.
    """
    if not 0 <= rolloff <= 1:
        msg = f'rolloff must be in [0, 1], got {rolloff}.'
        raise ValueError(msg)
    center = span * symbol_period / 2

    def pulse(tau: NDArray[np.float64]) -> NDArray[np.float64]:
        shifted = np.asarray(tau, dtype=np.float64) - center
        return _root_raised_cosine(shifted, symbol_period, rolloff)

    return pulse


__all__ = ['raised_cosine_filter', 'root_raised_cosine_filter']
