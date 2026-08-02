"""Spatial-multiplexing MIMO detectors: ZF, MMSE, ML, and K-best.

Given a receive vector `y = H x + n`, these recover the transmit vector `x`:

- `zf_detector` (zero-forcing) inverts the channel (`x_hat = H^+ y`) -- simple,
  but amplifies noise on ill-conditioned channels.
- `mmse_detector` balances interference nulling against noise enhancement.
- `ml_detector` searches every transmit vector for the maximum-likelihood one
  (exact, but exponential in `n_tx` -- for small problems and cross-checking).
- `kbest_detector` is a breadth-first sphere decoder that keeps the `k` best
  partial paths through the QR-triangularized channel -- near-ML at a fraction
  of the cost, and equal to ML for a large enough `k`.

ZF/MMSE return soft (unquantized) estimates; ML/K-best return constellation
points. All accept a single receive vector `(n_rx,)` or a block `(n_rx, T)`.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_block(received: ArrayLike) -> tuple[NDArray[np.complex128], bool]:
    """Return `(y, single)` where `y` is 2-D `(n_rx, T)` and `single` flags 1-D input."""
    y = np.asarray(received, dtype=np.complex128)
    single = y.ndim == 1
    return (y[:, None] if single else y), single


def zf_detector(received: ArrayLike, channel: ArrayLike) -> NDArray[np.complex128]:
    """Zero-forcing detection: `x_hat = pinv(H) @ y`.

    Args:
        received: Receive vector `(n_rx,)` or block `(n_rx, T)`.
        channel: Channel matrix `H`, shape `(n_rx, n_tx)`.

    Returns:
        Soft transmit estimates, shape `(n_tx,)` or `(n_tx, T)`.
    """
    y, single = _as_block(received)
    H = np.asarray(channel, dtype=np.complex128)
    x = np.linalg.pinv(H) @ y
    return np.asarray(x[:, 0] if single else x, dtype=np.complex128)


def mmse_detector(
    received: ArrayLike, channel: ArrayLike, noise_var: float,
) -> NDArray[np.complex128]:
    """Linear MMSE detection: `x_hat = (H^H H + noise_var I)^-1 H^H y`.

    Args:
        received: Receive vector `(n_rx,)` or block `(n_rx, T)`.
        channel: Channel matrix `H`, shape `(n_rx, n_tx)`.
        noise_var: Complex noise variance per receive sample (e.g. from
            `commpy` MIMO `noise_variance(snr_db)`).

    Returns:
        Soft transmit estimates, shape `(n_tx,)` or `(n_tx, T)`.
    """
    y, single = _as_block(received)
    H = np.asarray(channel, dtype=np.complex128)
    hermitian = H.conj().T
    filt = np.linalg.inv(hermitian @ H + noise_var * np.eye(H.shape[1])) @ hermitian
    x = filt @ y
    return np.asarray(x[:, 0] if single else x, dtype=np.complex128)


def _candidate_vectors(
    constellation: NDArray[np.complex128], n_tx: int,
) -> NDArray[np.complex128]:
    """All `M**n_tx` transmit vectors over a constellation, shape `(n_tx, M**n_tx)`."""
    grids = np.indices((constellation.size,) * n_tx).reshape(n_tx, -1)
    return constellation[grids]


def ml_detector(
    received: ArrayLike, channel: ArrayLike, constellation: ArrayLike,
) -> NDArray[np.complex128]:
    """Maximum-likelihood detection by exhaustive search over transmit vectors.

    Args:
        received: Receive vector `(n_rx,)` or block `(n_rx, T)`.
        channel: Channel matrix `H`, shape `(n_rx, n_tx)`.
        constellation: The transmit constellation points (`M` complex values).

    Returns:
        Detected constellation points, shape `(n_tx,)` or `(n_tx, T)`.
    """
    y, single = _as_block(received)
    H = np.asarray(channel, dtype=np.complex128)
    points = np.asarray(constellation, dtype=np.complex128)
    candidates = _candidate_vectors(points, H.shape[1])  # (n_tx, C)
    hypotheses = H @ candidates  # (n_rx, C)

    detected = np.empty((H.shape[1], y.shape[1]), dtype=np.complex128)
    for t in range(y.shape[1]):
        distances = np.sum(np.abs(y[:, t:t + 1] - hypotheses) ** 2, axis=0)
        detected[:, t] = candidates[:, int(np.argmin(distances))]
    return np.asarray(detected[:, 0] if single else detected, dtype=np.complex128)


def _kbest_column(
    z: NDArray[np.complex128], upper: NDArray[np.complex128],
    points: NDArray[np.complex128], list_size: int,
) -> NDArray[np.complex128]:
    """K-best search for one column: `z = R x`, `R = upper` triangular."""
    n_tx = upper.shape[0]
    # Each path: (symbols decided so far as [x_{n-1}, x_{n-2}, ...], accumulated metric).
    paths: list[tuple[list[np.complex128], float]] = [([], 0.0)]
    for layer in range(n_tx - 1, -1, -1):
        expanded: list[tuple[list[np.complex128], float]] = []
        for decided, metric in paths:
            interference = sum(
                upper[layer, n_tx - 1 - k] * decided[k] for k in range(len(decided))
            )
            for point in points:
                residual = z[layer] - upper[layer, layer] * point - interference
                expanded.append(([*decided, point], metric + float(np.abs(residual) ** 2)))
        expanded.sort(key=lambda path: path[1])
        paths = expanded[:list_size]
    best = paths[0][0]  # [x_{n-1}, ..., x_0]
    return np.array(best[::-1], dtype=np.complex128)


def kbest_detector(
    received: ArrayLike, channel: ArrayLike, constellation: ArrayLike, list_size: int = 4,
) -> NDArray[np.complex128]:
    """K-best (breadth-first sphere) detection over the QR-triangularized channel.

    Args:
        received: Receive vector `(n_rx,)` or block `(n_rx, T)`.
        channel: Channel matrix `H`, shape `(n_rx, n_tx)` with `n_rx >= n_tx`.
        constellation: The transmit constellation points (`M` complex values).
        list_size: Number of survivor paths `k` (larger is closer to ML).

    Returns:
        Detected constellation points, shape `(n_tx,)` or `(n_tx, T)`.

    Raises:
        ValueError: If `list_size < 1` or `n_rx < n_tx`.
    """
    if list_size < 1:
        msg = f'list_size must be >= 1, got {list_size}.'
        raise ValueError(msg)
    y, single = _as_block(received)
    H = np.asarray(channel, dtype=np.complex128)
    if H.shape[0] < H.shape[1]:
        msg = f'K-best needs n_rx >= n_tx, got H shape {H.shape}.'
        raise ValueError(msg)
    points = np.asarray(constellation, dtype=np.complex128)

    q, upper = np.linalg.qr(H)
    z = q.conj().T @ y  # (n_tx, T)
    detected = np.empty((H.shape[1], y.shape[1]), dtype=np.complex128)
    for t in range(y.shape[1]):
        detected[:, t] = _kbest_column(z[:, t], upper, points, list_size)
    return np.asarray(detected[:, 0] if single else detected, dtype=np.complex128)


__all__ = ['kbest_detector', 'ml_detector', 'mmse_detector', 'zf_detector']
