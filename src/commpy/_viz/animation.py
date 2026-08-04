"""Animated views of processes that a single frame cannot show.

Some things in a link are *processes*: a constellation collapsing as SNR falls,
a belief-propagation decoder pulling its LLRs apart iteration by iteration, a
Viterbi traceback committing to a path. Each function here returns a
`matplotlib.animation.FuncAnimation`, which a notebook renders with
`HTML(anim.to_jshtml())` and a script writes out with
`anim.save('name.gif', writer='pillow')`.

Keep a reference to the returned object: matplotlib animations stop the moment
they are garbage-collected.
"""

from collections.abc import Callable

import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.collections import LineCollection
from numpy.typing import ArrayLike, NDArray

from commpy._channelCoding.convolutional.trellis import Trellis
from commpy._channelCoding.convolutional.viterbi import (
    _acs_forward,
    _branch_metrics_hard,
    _branch_metrics_soft,
    _traceback,
)
from commpy._channels.channels import Channels
from commpy._modulation.base import Modulator
from commpy._viz.decoding import _iterative_decoder
from commpy._viz.style import (
    AXIS_COLOR,
    SERIES_COLORS,
    TEXT_MUTED,
    TEXT_SECONDARY,
    _figure_axes,
    _finalize,
)

# Frame interval in milliseconds -- slow enough to read a changing constellation,
# fast enough that a 20-frame animation does not feel like a slideshow.
_INTERVAL_MS = 400


def animate_constellation(
    modulator: Modulator,
    snr_db_range: ArrayLike,
    *,
    n_symbols: int = 2000,
    channel_fn: Callable[[NDArray[np.complex128], float, np.random.Generator],
                         NDArray[np.complex128]] | None = None,
    rng: np.random.Generator | None = None,
) -> FuncAnimation:
    """Animate a received constellation collapsing as the SNR sweeps.

    Watching the clouds grow into each other shows *where* a modulation order
    starts to fail: the decision regions closest together fill in first.

    Args:
        modulator: The modulator whose constellation is transmitted.
        snr_db_range: SNR values in dB, one animation frame each.
        n_symbols: Symbols drawn per frame.
        channel_fn: Optional `(symbols, snr_db, rng) -> symbols` channel;
            defaults to AWGN.
        rng: Optional `np.random.Generator` for reproducibility.

    Returns:
        The animation. Keep a reference to it or it stops immediately.
    """
    if rng is None:
        rng = np.random.default_rng()
    snr_values = np.atleast_1d(np.asarray(snr_db_range, dtype=np.float64))

    def awgn(symbols: NDArray[np.complex128], snr: float,
             generator: np.random.Generator) -> NDArray[np.complex128]:
        return np.asarray(Channels.awgn(symbols, snr, rng=generator))

    channel = awgn if channel_fn is None else channel_fn

    bits = rng.integers(0, 2, n_symbols * modulator.bits_per_symbol)
    symbols = modulator.modulate(bits)
    frames = [np.asarray(channel(symbols, float(snr), rng)) for snr in snr_values]
    limit = float(np.max(np.abs(np.concatenate([f.real for f in frames])))) * 1.1

    figure, ax = _figure_axes(figsize=(5.5, 5.5))
    ax.set_aspect('equal')
    ax.axhline(0.0, color=AXIS_COLOR, linewidth=0.8, zorder=0)
    ax.axvline(0.0, color=AXIS_COLOR, linewidth=0.8, zorder=0)
    cloud = ax.scatter([], [], s=6, alpha=0.35, color=SERIES_COLORS[0], edgecolors='none')
    ax.scatter(
        modulator.constellation.real, modulator.constellation.imag, s=70,
        color=SERIES_COLORS[1], edgecolors=AXIS_COLOR, linewidths=1.5, zorder=3,
    )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    _finalize(ax, xlabel='In-phase', ylabel='Quadrature')

    def draw(index: int) -> list[Artist]:
        received = frames[index]
        cloud.set_offsets(np.column_stack([received.real, received.imag]))
        ax.set_title(f'Received constellation at {snr_values[index]:g} dB',
                     color=TEXT_SECONDARY)
        return [cloud]

    return FuncAnimation(
        figure, draw, frames=len(frames), interval=_INTERVAL_MS, blit=False,
    )


def animate_decoding(
    code: object,
    llr: ArrayLike,
    *,
    max_iter: int = 12,
) -> FuncAnimation:
    """Animate an iterative decoder's codeword settling over its iterations.

    Frame `i` is the codeword after `i` iterations, with the bits that flipped
    since the previous iteration highlighted. Watching where the flips cluster
    shows which parity checks the decoder is still fighting over; when the
    highlights stop appearing, it has converged.

    This shows *decisions*, not beliefs: `LDPCCode.decode` and
    `TurboCode.decode` return codewords rather than posterior LLRs, and this
    function deliberately stays on the public API rather than reaching into a
    particular decoder's internals so it works for both.

    Like `plot_decoder_convergence`, the trace comes from re-decoding with a
    rising iteration cap, so it costs `O(max_iter**2 / 2)` decoder work.

    Args:
        code: An iterative decoder such as `LDPCCode` or `TurboCode`.
        llr: Channel LLRs for one received block.
        max_iter: Number of iterations to animate.

    Returns:
        The animation. Keep a reference to it or it stops immediately.

    Raises:
        TypeError: If `code` is not an iterative decoder.
    """
    decode, keyword = _iterative_decoder(code)
    channel_llr = np.asarray(llr, dtype=np.float64)
    codewords = [
        np.asarray(decode(channel_llr, **{keyword: i})[1], dtype=np.float64)
        for i in range(1, max_iter + 1)
    ]

    figure, ax = _figure_axes(figsize=(7.5, 4.5))
    bars = ax.bar(
        np.arange(codewords[0].size), codewords[0], color=SERIES_COLORS[0], width=1.0,
    )
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    _finalize(ax, xlabel='Codeword bit', ylabel='Decoded bit')

    def draw(index: int) -> list[Artist]:
        current = codewords[index]
        flipped = (
            np.zeros(current.size, dtype=bool) if index == 0
            else current != codewords[index - 1]
        )
        for bar, value, changed in zip(bars, current, flipped, strict=True):
            bar.set_height(float(value))
            bar.set_color(SERIES_COLORS[1] if changed else SERIES_COLORS[0])
        ax.set_title(
            f'{type(code).__name__} after {index + 1} iteration(s) '
            f'-- {int(flipped.sum())} bit(s) flipped',
            color=TEXT_SECONDARY,
        )
        return list(bars)

    return FuncAnimation(
        figure, draw, frames=len(codewords), interval=_INTERVAL_MS, blit=False,
    )


def _survivor_segments(
    predecessor: NDArray[np.int64], n_states: int, stage: int,
) -> list[list[tuple[int, int]]]:
    """Return the survivor branches entering `stage`."""
    return [[(stage, int(predecessor[stage, s])), (stage + 1, s)] for s in range(n_states)]


def animate_viterbi(
    trellis: Trellis,
    received: ArrayLike,
    *,
    mode: str = 'hard',
    max_stages: int = 12,
) -> FuncAnimation:
    """Animate the Viterbi algorithm extending its survivors stage by stage.

    Each frame adds one trellis section of survivors, then the final frames
    overlay the traceback -- which is the part worth watching, because it runs
    backwards through decisions the forward pass had already fixed.

    Args:
        trellis: The code's `Trellis`.
        received: Received sequence, as `viterbi_decode` takes it.
        mode: `'hard'` or `'soft'`.
        max_stages: Cap on the number of stages animated.

    Returns:
        The animation. Keep a reference to it or it stops immediately.

    Raises:
        ValueError: If `mode` is unknown or `received` has an inconsistent
            length.
    """
    if mode not in {'hard', 'soft'}:
        msg = f"mode must be 'hard' or 'soft', got {mode!r}."
        raise ValueError(msg)
    values = np.asarray(received, dtype=np.float64)
    if values.size % trellis.n_outputs != 0:
        msg = f'received length must be a multiple of {trellis.n_outputs}.'
        raise ValueError(msg)

    n_symbols = values.size // trellis.n_outputs
    symbols = values.reshape(n_symbols, trellis.n_outputs)
    branch_metrics = (
        _branch_metrics_hard(trellis, symbols)
        if mode == 'hard'
        else _branch_metrics_soft(trellis, symbols)
    )
    predecessor, input_bit, final_metric = _acs_forward(branch_metrics, trellis.next_state)
    decoded = _traceback(predecessor, input_bit, int(np.argmin(final_metric)), n_symbols)

    n_stages = min(n_symbols, max_stages)
    ml_states = [0]
    for bit in decoded[:n_stages]:
        ml_states.append(int(trellis.next_state[ml_states[-1], int(bit)]))

    figure, ax = _figure_axes(figsize=(1.1 * n_stages + 2.0, 0.5 * trellis.n_states + 2.0))
    survivors = LineCollection([], colors=TEXT_MUTED, linewidths=1.0, alpha=0.5)
    ax.add_collection(survivors)
    (path_line,) = ax.plot([], [], '-o', color=SERIES_COLORS[0], linewidth=2.2,
                           markersize=6, zorder=3)
    ax.set_xlim(-0.3, n_stages + 0.3)
    ax.set_ylim(-0.5, trellis.n_states - 0.5)
    ax.set_xticks(np.arange(n_stages + 1))
    ax.set_yticks(np.arange(trellis.n_states))
    _finalize(ax, xlabel='Stage', ylabel='State')

    def draw(frame: int) -> list[Artist]:
        if frame < n_stages:
            drawn: list[list[tuple[int, int]]] = []
            for stage in range(frame + 1):
                drawn.extend(_survivor_segments(predecessor, trellis.n_states, stage))
            survivors.set_segments([np.asarray(segment) for segment in drawn])
            path_line.set_data([], [])
            ax.set_title(f'Forward pass: stage {frame + 1} of {n_stages}',
                         color=TEXT_SECONDARY)
        else:
            kept = frame - n_stages + 1
            tail = ml_states[len(ml_states) - kept:]
            path_line.set_data(np.arange(len(ml_states) - kept, len(ml_states)), tail)
            ax.set_title('Traceback: following the survivors backwards',
                         color=TEXT_SECONDARY)
        return [survivors, path_line]

    return FuncAnimation(
        figure, draw, frames=n_stages + len(ml_states), interval=_INTERVAL_MS,
        blit=False,
    )


__all__ = [
    'animate_constellation',
    'animate_decoding',
    'animate_viterbi',
]
