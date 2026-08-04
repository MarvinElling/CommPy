"""Interactive Plotly counterparts of the most-used static plots.

The matplotlib functions elsewhere in `commpy._viz` are the primary API; these
exist for the cases where hovering, zooming, and toggling series pays for
itself -- exploring a noisy constellation point by point, or embedding a live
figure in HTML documentation.

Plotly is an optional dependency (`pip install commpy[viz]`). It is imported
inside each function rather than at module scope, so `import commpy` keeps
working -- and stays fast -- without it.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from commpy._modulation.base import Modulator
from commpy._simulation.link_simulation import SimulationResult
from commpy._viz.style import GRID_COLOR, SERIES_COLORS, SURFACE, TEXT_MUTED, TEXT_SECONDARY

if TYPE_CHECKING:  # pragma: no cover -- import-time typing only
    from plotly.graph_objects import Figure

_INSTALL_HINT = (
    "plotly is required for commpy's interactive plots; install it with "
    "`pip install commpy[viz]`."
)


def _graph_objects() -> Any:  # noqa: ANN401 -- the plotly module object, typed only when installed
    """Import `plotly.graph_objects`, with an actionable error when it is absent."""
    try:
        import plotly.graph_objects as go  # noqa: PLC0415 -- optional dependency, imported on use
    except ImportError as exc:  # pragma: no cover -- exercised only without the extra
        raise ImportError(_INSTALL_HINT) from exc
    return go


def _layout(title: str, xaxis: str, yaxis: str) -> dict[str, Any]:
    """Return the shared Plotly layout, matching the matplotlib chrome."""
    axis = {
        'gridcolor': GRID_COLOR,
        'linecolor': TEXT_MUTED,
        'zeroline': False,
        'title_font': {'color': TEXT_SECONDARY},
        'tickfont': {'color': TEXT_MUTED},
    }
    return {
        'title': {'text': title},
        'xaxis': {'title': {'text': xaxis}, **axis},
        'yaxis': {'title': {'text': yaxis}, **axis},
        'plot_bgcolor': SURFACE,
        'paper_bgcolor': SURFACE,
        'font': {'color': TEXT_SECONDARY},
        'margin': {'l': 70, 'r': 30, 't': 60, 'b': 60},
    }


def plotly_constellation(
    source: Modulator | ArrayLike,
    *,
    received: ArrayLike | None = None,
) -> 'Figure':
    """Build an interactive constellation diagram.

    Hovering a received symbol reports its exact I/Q coordinates, which is the
    thing a static scatter cannot give you when the cloud is dense.

    Args:
        source: A `Modulator` or an array of complex reference points.
        received: Received symbols to scatter behind the reference points.

    Returns:
        A Plotly `Figure`.

    Raises:
        ImportError: If plotly is not installed.
    """
    go = _graph_objects()
    points = (
        source.constellation if isinstance(source, Modulator)
        else np.asarray(source, dtype=np.complex128)
    )

    traces = []
    if received is not None:
        rx = np.asarray(received, dtype=np.complex128)
        traces.append(go.Scattergl(
            x=rx.real, y=rx.imag, mode='markers', name='received',
            marker={'size': 4, 'color': SERIES_COLORS[0], 'opacity': 0.45},
        ))
    traces.append(go.Scatter(
        x=points.real, y=points.imag, mode='markers', name='constellation',
        marker={'size': 12, 'color': SERIES_COLORS[1],
                'line': {'width': 1.5, 'color': TEXT_MUTED}},
    ))

    figure = go.Figure(data=traces, layout=_layout('Constellation', 'In-phase', 'Quadrature'))
    figure.update_yaxes(scaleanchor='x', scaleratio=1)
    return figure


def plotly_eye_diagram(
    signal: ArrayLike,
    sps: int,
    *,
    n_traces: int = 200,
) -> 'Figure':
    """Build an interactive eye diagram.

    Args:
        signal: Oversampled baseband samples; the real part is drawn.
        sps: Samples per symbol.
        n_traces: Maximum number of overlaid windows.

    Returns:
        A Plotly `Figure`.

    Raises:
        ImportError: If plotly is not installed.
        ValueError: If `sps < 2` or the signal is too short for one window.
    """
    go = _graph_objects()
    if sps < 2:
        msg = f'sps must be at least 2 to form an eye, got {sps}.'
        raise ValueError(msg)

    samples = np.asarray(signal).real
    window = 2 * sps + 1
    n_windows = (samples.size - window) // sps + 1
    if n_windows < 1:
        msg = f'Signal too short: need at least {window} samples for one window.'
        raise ValueError(msg)
    n_windows = min(n_windows, n_traces)

    starts = sps * np.arange(n_windows)
    traces = samples[starts[:, None] + np.arange(window)[None, :]]
    x = np.arange(window) / sps

    # One trace with NaN separators rather than n_windows traces: plotly renders
    # a single path far faster, and the legend stays a single entry.
    xs = np.tile(np.append(x, np.nan), n_windows)
    ys = np.concatenate([np.append(row, np.nan) for row in traces])

    figure = go.Figure(
        data=[go.Scattergl(
            x=xs, y=ys, mode='lines', name='traces',
            line={'width': 1, 'color': SERIES_COLORS[0]}, opacity=0.35,
        )],
        layout=_layout('Eye diagram', 'Time (symbol periods)', 'Amplitude'),
    )
    figure.update_layout(showlegend=False)
    return figure


def plotly_psd(signal: ArrayLike, fs: float, *, nperseg: int = 256) -> 'Figure':
    """Build an interactive two-sided power spectral density plot.

    Args:
        signal: Baseband samples.
        fs: Sampling rate in Hz.
        nperseg: Welch segment length; clamped to the signal length.

    Returns:
        A Plotly `Figure`.

    Raises:
        ImportError: If plotly is not installed.
    """
    go = _graph_objects()
    from scipy.signal import welch  # noqa: PLC0415 -- keeps this module import-light

    samples = np.asarray(signal)
    freqs, psd = welch(
        samples, fs=fs, nperseg=min(nperseg, samples.size), return_onesided=False,
    )
    order = np.argsort(freqs)
    power_db = 10 * np.log10(np.maximum(psd[order], np.finfo(float).tiny))

    return go.Figure(
        data=[go.Scattergl(x=freqs[order], y=power_db, mode='lines',
                           line={'color': SERIES_COLORS[0]}, name='PSD')],
        layout=_layout('Power spectral density', 'Frequency (Hz)', 'PSD (dB/Hz)'),
    )


def plotly_waterfall(results: dict[str, SimulationResult]) -> 'Figure':
    """Build an interactive BER/FER waterfall with confidence-interval bars.

    Args:
        results: Mapping of label to `SimulationResult`. Clicking a legend
            entry hides its curve, which is the fastest way to compare two
            codes out of a crowded figure.

    Returns:
        A Plotly `Figure`.

    Raises:
        ImportError: If plotly is not installed.
        ValueError: If `results` is empty.
    """
    go = _graph_objects()
    if not results:
        msg = 'results must contain at least one curve.'
        raise ValueError(msg)

    traces = []
    for (label, result), color in zip(results.items(), SERIES_COLORS, strict=False):
        traces.append(go.Scatter(
            x=result.snr_db, y=result.error_rate, mode='lines+markers', name=label,
            line={'color': color},
            error_y={
                'type': 'data',
                'array': np.clip(result.ci_upper - result.error_rate, 0.0, None),
                'arrayminus': np.clip(result.error_rate - result.ci_lower, 0.0, None),
                'visible': True,
            },
        ))

    figure = go.Figure(data=traces, layout=_layout('Waterfall curve', 'SNR (dB)', 'Error rate'))
    figure.update_yaxes(type='log')
    return figure


def plotly_tanner_graph(source: object) -> 'Figure':
    """Build an interactive Tanner graph.

    Args:
        source: Anything carrying an `H` attribute (e.g. `LDPCCode`) or a
            binary matrix. Hovering a node reports its degree, which is what
            you actually want when hunting for irregular columns.

    Returns:
        A Plotly `Figure`.

    Raises:
        ImportError: If plotly is not installed.
    """
    go = _graph_objects()
    matrix: NDArray[np.uint8] = np.asarray(getattr(source, 'H', source), dtype=np.uint8)
    m, n = matrix.shape
    check_idx, var_idx = np.nonzero(matrix)

    var_x = (np.arange(n) + 0.5) / n
    check_x = (np.arange(m) + 0.5) / m

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for check, variable in zip(check_idx, var_idx, strict=True):
        edge_x.extend([float(var_x[variable]), float(check_x[check]), None])
        edge_y.extend([0.0, 1.0, None])

    figure = go.Figure(
        data=[
            go.Scattergl(x=edge_x, y=edge_y, mode='lines', name='edges',
                         line={'width': 0.7, 'color': TEXT_MUTED}, opacity=0.6,
                         hoverinfo='skip'),
            go.Scatter(x=var_x, y=np.zeros(n), mode='markers', name='variable nodes',
                       marker={'size': 10, 'color': SERIES_COLORS[0]},
                       text=[f'v{i}, degree {d}' for i, d in enumerate(matrix.sum(axis=0))],
                       hoverinfo='text'),
            go.Scatter(x=check_x, y=np.ones(m), mode='markers', name='check nodes',
                       marker={'size': 11, 'color': SERIES_COLORS[1], 'symbol': 'square'},
                       text=[f'c{i}, degree {d}' for i, d in enumerate(matrix.sum(axis=1))],
                       hoverinfo='text'),
        ],
        layout=_layout(f'Tanner graph ({n} variable nodes, {m} checks)', '', ''),
    )
    figure.update_xaxes(showticklabels=False, showgrid=False)
    figure.update_yaxes(showticklabels=False, showgrid=False, range=[-0.25, 1.25])
    return figure


__all__ = [
    'plotly_constellation',
    'plotly_eye_diagram',
    'plotly_psd',
    'plotly_tanner_graph',
    'plotly_waterfall',
]
