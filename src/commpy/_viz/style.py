"""Shared visual language for CommPy's plotting functions.

Every `plot_*`/`animate_*` function in `commpy` draws from the palette and axes
chrome defined here, so figures from different corners of the library compose
into one coherent look.

The categorical palette is eight fixed hues assigned in a **fixed order** -- slot
1 is always blue, slot 2 always orange, and so on -- validated so that every
adjacent pair stays separable under protanopia, deuteranopia, and tritanopia.
Color therefore tracks the entity, not its rank: dropping a curve from a
comparison never repaints the survivors.

Nothing here mutates matplotlib's global state. The plotting functions pass
colors explicitly, and `commpy_style()` merely *returns* an rcParams mapping for
callers who want the same look for their own figures::

    import matplotlib.pyplot as plt
    from commpy import commpy_style

    with plt.rc_context(commpy_style()):
        ...
"""

import warnings
from contextlib import AbstractContextManager
from typing import Any, cast

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots, in assignment order. Validated as a set: worst adjacent
# color-vision-deficient separation dE 9.1 (OKLab x100, >= 8 target), worst
# adjacent normal-vision separation dE 19.6 (>= 15 floor).
SERIES_COLORS: tuple[str, ...] = (
    '#2a78d6',  # 1 blue
    '#eb6834',  # 2 orange
    '#1baf7a',  # 3 aqua
    '#eda100',  # 4 yellow
    '#e87ba4',  # 5 magenta
    '#008300',  # 6 green
    '#4a3aa7',  # 7 violet
    '#e34948',  # 8 red
)

# Single-hue blue ramp, light -> dark. Sequential encoding (magnitude) uses one
# hue by design: a rainbow map invents perceptual boundaries the data does not
# have.
_BLUE_RAMP: tuple[str, ...] = (
    '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
    '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b',
)

# Diverging ramp for signed quantities (LLRs above all): two opposed hues with a
# neutral -- not merely pale -- midpoint, so "zero" reads as "nothing" rather
# than as a weak version of one pole.
_DIVERGING_RAMP: tuple[str, ...] = (
    '#0d366b', '#256abf', '#3987e5', '#86b6ef', '#cde2fb',
    '#f0efec',
    '#f9d0cf', '#f19d9c', '#e34948', '#b32c2b', '#7d1f1e',
)

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list('commpy_sequential', _BLUE_RAMP)
DIVERGING_CMAP = LinearSegmentedColormap.from_list('commpy_diverging', _DIVERGING_RAMP)

SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
TEXT_MUTED = '#898781'
GRID_COLOR = '#e1e0d9'
AXIS_COLOR = '#c3c2b7'

# Slot count past which hues would have to repeat.
_MAX_SERIES = len(SERIES_COLORS)


_STYLE_RCPARAMS: dict[str, Any] = {
    'axes.prop_cycle': cycler(color=list(SERIES_COLORS)),
    'axes.edgecolor': AXIS_COLOR,
    'axes.facecolor': SURFACE,
    'axes.grid': True,
    'axes.labelcolor': TEXT_SECONDARY,
    'axes.linewidth': 0.8,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.titlecolor': TEXT_PRIMARY,
    'figure.facecolor': SURFACE,
    'grid.color': GRID_COLOR,
    'grid.linewidth': 0.6,
    'image.cmap': SEQUENTIAL_CMAP,
    'legend.frameon': False,
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
    'savefig.facecolor': SURFACE,
    'text.color': TEXT_PRIMARY,
    'xtick.color': TEXT_MUTED,
    'ytick.color': TEXT_MUTED,
}


def commpy_style() -> AbstractContextManager[None]:
    """Return a context manager that applies CommPy's plot styling.

    CommPy's own plotting functions do not need it -- they style the axes they
    create directly -- but entering it makes *your* figures match theirs::

        with commpy_style():
            fig, ax = plt.subplots()
            ax.plot(snr_db, ber)

    Returns:
        A context manager applying the palette cycle, the recessive grid and
        axes, and the muted tick/label ink for its duration.
    """
    # matplotlib's stubs narrow rc keys to a Literal union of every known
    # parameter name, which a plain dict[str, Any] cannot satisfy.
    return plt.rc_context(rc=cast('dict[Any, Any]', _STYLE_RCPARAMS))


def series_colors(n: int) -> list[str]:
    """Return the first `n` categorical colors, in fixed slot order.

    Args:
        n: Number of distinct series to color.

    Returns:
        `n` hex colors. Requesting more than the eight available slots forces
        hues to repeat, which makes two series indistinguishable; that case
        warns and cycles rather than failing, but the fix is to split the
        figure or fold the tail into an "other" curve.
    """
    if n > _MAX_SERIES:
        msg = (
            f'{n} series requested but only {_MAX_SERIES} distinguishable colors exist; '
            f'hues will repeat. Split the comparison across figures instead.'
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
    return [SERIES_COLORS[i % _MAX_SERIES] for i in range(n)]


def _apply_chrome(ax: Axes) -> None:
    """Apply the recessive axes chrome to a freshly created axes."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS_COLOR)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_MUTED, labelcolor=TEXT_MUTED)


def _axes(ax: Axes | None, figsize: tuple[float, float] = (7.0, 5.0)) -> Axes:
    """Return `ax`, or a new styled axes when it is `None`.

    A caller-supplied axes is returned untouched: the caller owns its styling,
    and restyling it would break composition into an existing figure.

    Args:
        ax: Axes to draw into, or `None` to create one.
        figsize: Figure size used only when creating a new figure.

    Returns:
        The axes to draw into.
    """
    if ax is not None:
        return ax
    _, new_ax = plt.subplots(figsize=figsize)
    _apply_chrome(new_ax)
    return new_ax


def _finalize(  # noqa: PLR0913 -- one keyword per axes decoration; grouping them would only add indirection
    ax: Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = False,
    grid: bool = True,
) -> Axes:
    """Label an axes and apply the shared grid treatment.

    Args:
        ax: Axes to decorate.
        title: Axes title, if any.
        xlabel: X-axis label, if any.
        ylabel: Y-axis label, if any.
        legend: Whether to draw a legend. Ignored when nothing is labeled.
        grid: Whether to draw the recessive grid behind the data.

    Returns:
        The same axes, for chaining.
    """
    if title is not None:
        ax.set_title(title, color=TEXT_PRIMARY)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    if grid:
        ax.grid(True, color=GRID_COLOR, linewidth=0.6)
        ax.set_axisbelow(True)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, labelcolor=TEXT_SECONDARY)
    return ax


__all__ = [
    'DIVERGING_CMAP',
    'SEQUENTIAL_CMAP',
    'SERIES_COLORS',
    'commpy_style',
    'series_colors',
]
