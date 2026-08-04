"""Diagnostic plots of what a decoder does: reliability, convergence, survivors.

Where `commpy._viz.coding` draws a code's fixed structure, everything here
needs channel data: LLRs coming out of a demodulator, the iterations a message
passing decoder takes to settle, the paths a sequence decoder keeps alive.

Throughout, LLRs follow the library-wide convention set by
`Modulator.soft_demodulate`: `L = log(P(0)/P(1))`, so a **positive** LLR favors
bit 0.
"""

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from numpy.typing import ArrayLike, NDArray

from commpy._channelCoding.convolutional.trellis import Trellis
from commpy._channelCoding.convolutional.viterbi import (
    _acs_forward,
    _branch_metrics_hard,
    _branch_metrics_soft,
    _traceback,
)
from commpy._channelCoding.polar.decoder import scl_decode
from commpy._channelCoding.turbo.bcjr import bcjr_decode
from commpy._channelCoding.turbo.rsc import RSCTrellis, rsc_encode
from commpy._viz.style import (
    AXIS_COLOR,
    SERIES_COLORS,
    TEXT_MUTED,
    TEXT_SECONDARY,
    _axes,
    _finalize,
)


def plot_llr_histogram(
    llr: ArrayLike,
    *,
    bits: ArrayLike | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot the distribution of log-likelihood ratios, split by transmitted bit.

    Two well-separated humps mean the demodulator is confident and the decoder
    has an easy job; humps overlapping at zero mean it is guessing. With
    `bits` given, the bit-0 hump must sit on the **positive** side -- if it does
    not, an LLR sign convention is inverted somewhere upstream.

    Args:
        llr: Log-likelihood ratios.
        bits: Optional transmitted bits, same length, used to split the
            distribution. Without them a single pooled histogram is drawn.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the histogram was drawn on.

    Raises:
        ValueError: If `bits` is given with a different length than `llr`.
    """
    values = np.asarray(llr, dtype=np.float64)

    ax = _axes(ax, figsize=(7.0, 4.5))
    if bits is None:
        ax.hist(values, bins=60, color=SERIES_COLORS[0], alpha=0.85)
        legend = False
    else:
        sent = np.asarray(bits, dtype=np.int64)
        if sent.size != values.size:
            msg = f'bits must have the same length as llr ({values.size}), got {sent.size}.'
            raise ValueError(msg)
        bin_edges = np.histogram_bin_edges(values, bins=60).tolist()
        for bit, color in zip((0, 1), SERIES_COLORS, strict=False):
            ax.hist(
                values[sent == bit], bins=bin_edges, color=color, alpha=0.6,
                label=f'sent bit {bit}',
            )
        legend = True

    ax.axvline(0.0, color=AXIS_COLOR, linewidth=1.0)
    return _finalize(
        ax, title='LLR distribution', xlabel='LLR (positive favors bit 0)',
        ylabel='Count', legend=legend,
    )


def _iterative_decoder(code: object) -> tuple[Callable[..., Any], str]:
    """Return a code's `decode` method and the name of its iteration-cap parameter.

    LDPC spells the cap `max_iter` and turbo spells it `iterations`; resolving
    it from the signature keeps this working for either without the caller
    having to say which kind of code it holds.
    """
    decode = getattr(code, 'decode', None)
    if decode is None:
        msg = f'{type(code).__name__} has no decode() method.'
        raise TypeError(msg)
    parameters = inspect.signature(decode).parameters
    for name in ('max_iter', 'iterations'):
        if name in parameters:
            return decode, name
    msg = (
        f'{type(code).__name__}.decode() takes neither max_iter nor iterations, '
        f'so it has no iterative behaviour to trace.'
    )
    raise TypeError(msg)


def plot_decoder_convergence(
    code: object,
    llr: ArrayLike,
    *,
    max_iter: int = 20,
    reference: ArrayLike | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot how a decoder's error count falls as its iterations proceed.

    With `reference` given, the curve is message bit errors; otherwise it is the
    syndrome weight (unsatisfied parity checks), which needs no ground truth but
    requires the code to expose an `H` matrix.

    The trace is produced by re-decoding with an iteration cap of `1, 2, ...`,
    which the decoders permit because they are deterministic. That costs
    `O(max_iter**2 / 2)` decoder work -- fine for a diagnostic on one block,
    not something to put inside a Monte-Carlo loop.

    Args:
        code: An iterative decoder such as `LDPCCode` or `TurboCode`.
        llr: Channel LLRs for one received block.
        max_iter: Highest iteration count to trace.
        reference: Optional transmitted message, enabling a true bit-error
            curve.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the convergence curve was drawn on.

    Raises:
        TypeError: If `code` is not an iterative decoder, or if no `reference`
            is given and the code exposes no `H`.
    """
    decode, keyword = _iterative_decoder(code)
    parity_check = getattr(code, 'H', None)
    if reference is None and parity_check is None:
        msg = (
            f'{type(code).__name__} exposes no H matrix, so syndrome weight cannot be '
            f'measured; pass reference= to plot true bit errors instead.'
        )
        raise TypeError(msg)

    iterations = np.arange(1, max_iter + 1)
    counts = []
    for n_iter in iterations:
        message, codeword, _ = decode(llr, **{keyword: int(n_iter)})
        if reference is None:
            syndrome = (np.asarray(parity_check) @ np.asarray(codeword)) % 2
            counts.append(int(syndrome.sum()))
        else:
            counts.append(int(np.count_nonzero(np.asarray(reference) != message)))

    label = 'message bit errors' if reference is not None else 'unsatisfied checks'
    ax = _axes(ax, figsize=(7.0, 4.5))
    ax.plot(iterations, counts, 'o-', color=SERIES_COLORS[0], markersize=5)
    ax.set_xticks(iterations[:: max(1, max_iter // 10)])
    ax.set_ylim(bottom=0)
    # Both metrics are counts; fractional ticks would be meaningless.
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return _finalize(
        ax, title=f'Decoder convergence ({type(code).__name__})',
        xlabel='Iterations', ylabel=label,
    )


def _mutual_information(llr: NDArray[np.float64], bits: NDArray[np.int64]) -> float:
    """Estimate I(bit; LLR) in bits, assuming consistent LLRs.

    Uses the ergodic estimator `1 - E[log2(1 + exp(-s*L))]` with `s = 1-2b` the
    sign the LLR should carry, so no closed-form J-function approximation is
    needed anywhere in the EXIT chart.
    """
    sign = 1.0 - 2.0 * bits
    return float(1.0 - np.mean(np.logaddexp(0.0, -sign * llr)) / np.log(2.0))


def plot_exit_chart(  # noqa: PLR0913 -- sweep resolution, block size, and seed are independent knobs
    trellis: RSCTrellis,
    *,
    snr_db: float,
    n_points: int = 15,
    n_bits: int = 20000,
    rng: np.random.Generator | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot the extrinsic-information-transfer chart of an RSC constituent decoder.

    Each point feeds `bcjr_decode` a-priori LLRs of a known quality and measures
    the quality of the extrinsic LLRs that come back. Plotted together with its
    own mirror image -- the second constituent decoder in a turbo pair -- the
    gap between the curves is the *decoding tunnel*: while it stays open,
    iterating keeps buying information, and where the curves touch, iteration
    stalls.

    Both axes are measured by Monte-Carlo rather than a fitted J-function, so
    the curve reflects this decoder rather than a Gaussian idealization of it.

    Args:
        trellis: The constituent `RSCTrellis`.
        snr_db: Channel **Es/N0** in dB for the systematic and parity LLRs,
            matching the rest of the library's per-symbol convention. EXIT
            charts are conventionally read in Eb/N0, which for these rate-1/2
            constituent codes is 3 dB higher -- so the interesting region,
            where the tunnel is about to close, sits near `snr_db = -4`, not
            near 0.
        n_points: Number of a-priori operating points to sweep.
        n_bits: Block length per operating point; larger is smoother.
        rng: Optional `np.random.Generator` for reproducibility.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the EXIT chart was drawn on.
    """
    if rng is None:
        rng = np.random.default_rng()

    bits = rng.integers(0, 2, n_bits).astype(np.uint8)
    parity = rsc_encode(trellis, bits)
    noise_var = 1.0 / (2.0 * 10.0 ** (snr_db / 10.0))

    def channel_llr(sent: NDArray[np.uint8]) -> NDArray[np.float64]:
        symbols = 1.0 - 2.0 * sent.astype(np.float64)  # bit 0 -> +1
        received = symbols + np.sqrt(noise_var) * rng.standard_normal(sent.size)
        return np.asarray(2.0 * received / noise_var)

    systematic_llr = channel_llr(bits)
    parity_llr = channel_llr(parity)

    # Sweeping the a-priori LLR standard deviation and *measuring* the resulting
    # I_A avoids inverting a J-function approximation.
    sent_sign = 1.0 - 2.0 * bits.astype(np.float64)
    i_a, i_e = [], []
    for sigma in np.linspace(0.05, 7.0, n_points):
        apriori = sigma**2 / 2.0 * sent_sign + sigma * rng.standard_normal(n_bits)
        extrinsic = bcjr_decode(trellis, apriori, systematic_llr, parity_llr)
        i_a.append(_mutual_information(apriori, bits.astype(np.int64)))
        i_e.append(_mutual_information(extrinsic, bits.astype(np.int64)))

    ax = _axes(ax, figsize=(5.5, 5.5))
    ax.plot(i_a, i_e, '-o', color=SERIES_COLORS[0], markersize=4, label='decoder 1')
    ax.plot(i_e, i_a, '--s', color=SERIES_COLORS[1], markersize=4,
            label='decoder 2 (mirrored)')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect('equal')
    return _finalize(
        ax, title=f'EXIT chart at Es/N0 = {snr_db:g} dB',
        xlabel='$I_A$ (a-priori information)', ylabel='$I_E$ (extrinsic information)',
        legend=True,
    )


def plot_viterbi_paths(
    trellis: Trellis,
    received: ArrayLike,
    *,
    mode: str = 'hard',
    max_stages: int = 12,
    ax: Axes | None = None,
) -> Axes:
    """Plot the Viterbi survivor paths and highlight the maximum-likelihood one.

    At every stage the algorithm keeps exactly one survivor per state; those are
    drawn faintly, and the single path the traceback follows is drawn on top.
    Where survivors merge is where the decoder has committed to a decision.

    Args:
        trellis: The code's `Trellis`.
        received: Received sequence, as `viterbi_decode` takes it -- 0/1 bits
            for `mode='hard'`, LLRs for `mode='soft'`.
        mode: `'hard'` or `'soft'`.
        max_stages: Cap on the number of stages drawn; the trellis becomes
            unreadable long before a real block ends.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the survivor paths were drawn on.

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
    final_state = int(np.argmin(final_metric))
    decoded = _traceback(predecessor, input_bit, final_state, n_symbols)

    n_stages = min(n_symbols, max_stages)
    survivors = [
        [(t, int(predecessor[t, s])), (t + 1, s)]
        for t in range(n_stages)
        for s in range(trellis.n_states)
    ]

    # Replay the winning path forwards from the decoded bits.
    ml_states = [0]
    for bit in decoded[:n_stages]:
        ml_states.append(int(trellis.next_state[ml_states[-1], int(bit)]))

    ax = _axes(ax, figsize=(1.1 * n_stages + 2.0, 0.5 * trellis.n_states + 2.0))
    ax.add_collection(
        LineCollection(survivors, colors=TEXT_MUTED, linewidths=1.0, alpha=0.5),
    )
    ax.plot(np.arange(len(ml_states)), ml_states, '-o', color=SERIES_COLORS[0],
            linewidth=2.2, markersize=6, label='maximum-likelihood path', zorder=3)
    ax.plot([], [], color=TEXT_MUTED, linewidth=1.0, label='survivors')

    ax.set_xlim(-0.3, n_stages + 0.3)
    ax.set_ylim(-0.5, trellis.n_states - 0.5)
    ax.set_xticks(np.arange(n_stages + 1))
    ax.set_yticks(np.arange(trellis.n_states))
    return _finalize(ax, title='Viterbi survivor paths', xlabel='Stage', ylabel='State',
                     legend=True)


def plot_scl_paths(
    llr: ArrayLike,
    frozen: NDArray[np.bool_],
    *,
    list_size: int = 8,
    ax: Axes | None = None,
) -> Axes:
    """Plot the path metrics of a polar list decoder's surviving candidates.

    The winner is whichever survivor ends with the smallest metric. A large gap
    between rank 0 and rank 1 means the decision was comfortable; a cluster of
    near-equal metrics means the list barely resolved the codeword, and a CRC
    would be doing the real work of picking between them.

    Args:
        llr: Channel LLRs, length `N`.
        frozen: Length-`N` boolean frozen mask (`True` = frozen).
        list_size: Number of survivor paths to keep.
        ax: Axes to draw into; a new figure is created when omitted.

    Returns:
        The axes the path metrics were drawn on.
    """
    survivors = scl_decode(llr, frozen, list_size)
    metrics = np.array([metric for metric, _, _ in survivors], dtype=np.float64)
    colors = [SERIES_COLORS[0]] + [TEXT_MUTED] * (metrics.size - 1)

    ax = _axes(ax, figsize=(7.0, 4.5))
    ax.bar(np.arange(metrics.size), metrics, color=colors, width=0.7)
    ax.set_xticks(np.arange(metrics.size))
    ax.annotate(
        'selected', (0, metrics[0]), textcoords='offset points', xytext=(0, 6),
        ha='center', fontsize=9, color=TEXT_SECONDARY,
    )
    return _finalize(
        ax, title=f'SCL survivor path metrics (list size {list_size})',
        xlabel='Survivor rank (0 = best)', ylabel='Path metric (lower is better)',
    )


__all__ = [
    'plot_decoder_convergence',
    'plot_exit_chart',
    'plot_llr_histogram',
    'plot_scl_paths',
    'plot_viterbi_paths',
]
