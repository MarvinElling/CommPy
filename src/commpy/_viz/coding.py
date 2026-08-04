"""Structural plots of forward-error-correcting codes.

These draw what a code *is* -- its parity-check sparsity, its Tanner graph, its
trellis, which of its bit-channels are frozen, how its interleaver permutes --
as opposed to what a decoder *does* with it, which lives in
`commpy._viz.decoding`.
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.typing import NDArray

from commpy._channelCoding.convolutional.trellis import Trellis
from commpy._channelCoding.polar.code import PolarCode
from commpy._channelCoding.polar.construction import (
    bhattacharyya_reliabilities,
    gaussian_approx_reliabilities,
)
from commpy._channelCoding.turbo.rsc import RSCTrellis
from commpy._viz.style import (
    SERIES_COLORS,
    SURFACE,
    TEXT_MUTED,
    TEXT_SECONDARY,
    _axes,
    _finalize,
)


def _as_matrix(source: object) -> NDArray[np.uint8]:
    """Return the parity-check matrix of a code object, or `source` as a matrix."""
    matrix = getattr(source, 'H', source)
    return np.asarray(matrix, dtype=np.uint8)


def plot_parity_check(source: object, *, ax: Axes | None = None) -> Axes:
    """Plot the sparsity pattern of a parity-check matrix.

    The point of an LDPC matrix is that it is *sparse*; this makes that visible
    and puts the measured density in the title.

    Args:
        source: Anything carrying an `H` attribute (`LDPCCode`, `HammingCode`,
            ...) or a binary matrix. Pass `code.G` to inspect a generator
            matrix instead.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the pattern was drawn on.
    """
    matrix = _as_matrix(source)
    m, n = matrix.shape
    density = float(matrix.sum()) / (m * n)

    ax = _axes(ax, figsize=(7.0, 4.5))
    ax.imshow(
        matrix, cmap=ListedColormap([SURFACE, SERIES_COLORS[0]]), vmin=0, vmax=1,
        interpolation='nearest', aspect='auto',
    )
    return _finalize(
        ax,
        title=f'Parity-check matrix ({m}x{n}, density {density:.1%})',
        xlabel='Variable node (column)',
        ylabel='Check node (row)',
        grid=False,
    )


def plot_tanner_graph(source: object, *, ax: Axes | None = None) -> Axes:
    """Plot the bipartite Tanner graph of a parity-check matrix.

    Variable nodes sit on the lower row, check nodes on the upper one, and each
    one of `H` becomes an edge. Readable for teaching-sized codes; past a few
    dozen variable nodes the edges overwhelm the picture and
    `plot_parity_check` is the better view.

    Args:
        source: Anything carrying an `H` attribute (e.g. `LDPCCode`) or a
            binary matrix.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the graph was drawn on.
    """
    matrix = _as_matrix(source)
    m, n = matrix.shape
    check_idx, var_idx = np.nonzero(matrix)

    # Both node rows are spread over the same [0, 1] span so the graph stays
    # centered regardless of how far n and m differ.
    var_x = (np.arange(n) + 0.5) / n
    check_x = (np.arange(m) + 0.5) / m

    ax = _axes(ax, figsize=(8.0, 4.0))
    edges = [
        [(var_x[v], 0.0), (check_x[c], 1.0)]
        for c, v in zip(check_idx, var_idx, strict=True)
    ]
    ax.add_collection(LineCollection(edges, colors=TEXT_MUTED, linewidths=0.7, alpha=0.6))
    ax.scatter(var_x, np.zeros(n), s=60, color=SERIES_COLORS[0], marker='o',
               zorder=3, label='variable nodes')
    ax.scatter(check_x, np.ones(m), s=70, color=SERIES_COLORS[1], marker='s',
               zorder=3, label='check nodes')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.25, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    _finalize(
        ax, title=f'Tanner graph ({n} variable nodes, {m} checks, {check_idx.size} edges)',
        grid=False,
    )
    # Below the axes: the edge bundle fills the frame, leaving no reliably empty
    # corner for a legend box.
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.02), ncols=2,
        frameon=False, labelcolor=TEXT_SECONDARY,
    )
    return ax


def plot_trellis(
    trellis: Trellis | RSCTrellis,
    *,
    n_stages: int = 4,
    labels: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot the state-transition trellis of a convolutional or RSC encoder.

    Solid branches are input bit 0, dashed branches input bit 1 -- so the two
    are told apart by line style as well as color, which survives printing and
    color-vision deficiency.

    Args:
        trellis: A `Trellis` (feedforward) or `RSCTrellis` (recursive
            systematic).
        n_stages: Number of trellis sections to draw.
        labels: Annotate each branch with its output bits. Legible for small
            state counts only.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the trellis was drawn on.

    Raises:
        ValueError: If `n_stages < 1`.
    """
    if n_stages < 1:
        msg = f'n_stages must be at least 1, got {n_stages}.'
        raise ValueError(msg)

    n_states = trellis.n_states
    ax = _axes(ax, figsize=(1.6 * n_stages + 2.0, 0.5 * n_states + 2.0))

    for bit, (color, dash) in enumerate([(SERIES_COLORS[0], '-'), (SERIES_COLORS[1], '--')]):
        edges = [
            [(stage, state), (stage + 1, int(trellis.next_state[state, bit]))]
            for stage in range(n_stages)
            for state in range(n_states)
        ]
        ax.add_collection(
            LineCollection(edges, colors=color, linestyles=dash, linewidths=1.4, alpha=0.9),
        )
        if labels:
            _annotate_branches(ax, trellis, n_stages, bit, color)

    stages = np.arange(n_stages + 1)
    ax.scatter(
        np.repeat(stages, n_states), np.tile(np.arange(n_states), n_stages + 1),
        s=45, color=TEXT_SECONDARY, zorder=3,
    )
    ax.set_xlim(-0.3, n_stages + 0.3)
    ax.set_ylim(-0.5, n_states - 0.5)
    ax.set_xticks(stages)
    ax.set_yticks(np.arange(n_states))
    ax.legend(
        handles=[
            Line2D([], [], color=SERIES_COLORS[0], linestyle='-', label='input 0'),
            Line2D([], [], color=SERIES_COLORS[1], linestyle='--', label='input 1'),
        ],
        frameon=False, labelcolor=TEXT_SECONDARY, loc='upper center',
        bbox_to_anchor=(0.5, -0.12), ncols=2,
    )
    return _finalize(ax, title='Trellis diagram', xlabel='Stage', ylabel='State')


def _annotate_branches(
    ax: Axes, trellis: Trellis | RSCTrellis, n_stages: int, bit: int, color: str,
) -> None:
    """Label each branch of one input bit with the output it emits."""
    for stage in range(n_stages):
        for state in range(trellis.n_states):
            target = int(trellis.next_state[state, bit])
            if isinstance(trellis, Trellis):
                text = ''.join(str(int(b)) for b in trellis.output_bits[state, bit])
            else:
                # RSC: systematic bit is the input itself, so only parity varies.
                text = f'{bit}{int(trellis.parity[state, bit])}'
            ax.annotate(
                text, (stage + 0.5, (state + target) / 2),
                ha='center', va='center', fontsize=7, color=color,
                bbox={'boxstyle': 'round,pad=0.15', 'facecolor': SURFACE, 'edgecolor': 'none'},
                zorder=4,
            )


def plot_polar_reliabilities(
    block_length: int,
    *,
    design_snr_db: float = 1.0,
    ax: Axes | None = None,
) -> Axes:
    """Plot how the two polar constructions rank the synthetic bit-channels.

    Bhattacharyya scores and Gaussian-approximation mean LLRs live on entirely
    different scales, so what is drawn is each construction's *rank* (0 = least
    reliable). Rank is also exactly what the code selection consumes -- the
    frozen set is the argsort of the score -- so where the two curves separate
    is where the constructions would freeze different bits.

    Args:
        block_length: Codeword length `N` (a power of two).
        design_snr_db: Design-point SNR in dB for both constructions.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the ranks were drawn on.
    """
    scores = {
        'Gaussian approximation': gaussian_approx_reliabilities(block_length, design_snr_db),
        'Bhattacharyya': bhattacharyya_reliabilities(block_length, design_snr_db),
    }

    ax = _axes(ax, figsize=(7.5, 4.5))
    for (name, score), color in zip(scores.items(), SERIES_COLORS, strict=False):
        rank = np.empty(block_length, dtype=np.int64)
        rank[np.argsort(score, kind='stable')] = np.arange(block_length)
        ax.plot(rank, '.', color=color, markersize=4, label=name, alpha=0.8)

    return _finalize(
        ax, title=f'Polar bit-channel reliability ranking (N={block_length}, '
                  f'{design_snr_db:g} dB)',
        xlabel='Bit-channel index', ylabel='Reliability rank', legend=True,
    )


def plot_frozen_bits(code: PolarCode, *, ax: Axes | None = None) -> Axes:
    """Plot the frozen/information bit map of a polar code.

    The length-`N` frozen mask is folded into a rectangle so the polarization
    structure -- reliable channels clustering toward high indices -- is visible
    at a glance instead of as one very long strip.

    Args:
        code: The `PolarCode` whose `frozen` mask is drawn.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the map was drawn on.
    """
    frozen = np.asarray(code.frozen)
    exponent = int(np.log2(frozen.size))
    rows = 1 << (exponent // 2)
    grid = frozen.reshape(rows, -1)

    ax = _axes(ax, figsize=(6.5, 5.0))
    ax.imshow(
        grid, cmap=ListedColormap([SERIES_COLORS[0], TEXT_MUTED]), vmin=0, vmax=1,
        interpolation='nearest', aspect='auto',
    )
    ax.legend(
        handles=[
            Patch(facecolor=SERIES_COLORS[0], label='information'),
            Patch(facecolor=TEXT_MUTED, label='frozen'),
        ],
        frameon=False, labelcolor=TEXT_SECONDARY, loc='upper center',
        bbox_to_anchor=(0.5, -0.1), ncols=2,
    )
    return _finalize(
        ax, title=f'Polar frozen-bit map (N={code.n}, K={code.k}, rate {code.rate:.2f})',
        xlabel='Bit index (low bits)', ylabel='Bit index (high bits)', grid=False,
    )


def _as_permutation(source: object) -> NDArray[np.int64]:
    """Return the permutation an interleaver applies, or `source` as one."""
    permutation = getattr(source, 'interleaver', None)  # TurboCode
    if permutation is not None:
        return np.asarray(permutation, dtype=np.int64)

    interleave = getattr(source, 'interleave', None)  # BlockInterleaver
    block_size = getattr(source, 'block_size', None)
    if interleave is not None and block_size is not None:
        return np.asarray(interleave(np.arange(block_size)), dtype=np.int64)

    return np.asarray(source, dtype=np.int64)


def plot_interleaver(source: object, *, ax: Axes | None = None) -> Axes:
    """Plot the permutation an interleaver applies.

    A good interleaver scatters neighbors far apart -- the eye test is that the
    dots show no short-range structure, since any diagonal streak means
    adjacent input bits stay adjacent and burst errors survive deinterleaving.

    Args:
        source: A `TurboCode`, a `BlockInterleaver`, or a permutation array.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the permutation was drawn on.
    """
    perm = _as_permutation(source)

    ax = _axes(ax, figsize=(5.5, 5.5))
    ax.set_aspect('equal')
    ax.scatter(np.arange(perm.size), perm, s=8, color=SERIES_COLORS[0], edgecolors='none')
    return _finalize(
        ax, title=f'Interleaver permutation (length {perm.size})',
        xlabel='Input position', ylabel='Output position',
    )


__all__ = [
    'plot_frozen_bits',
    'plot_interleaver',
    'plot_parity_check',
    'plot_polar_reliabilities',
    'plot_tanner_graph',
    'plot_trellis',
]
