"""Multiple-input multiple-output (MIMO) flat-fading channel models.

A narrowband MIMO channel with `n_tx` transmit and `n_rx` receive antennas maps
a transmit vector `x` (length `n_tx`) to a receive vector `y = H x + n` (length
`n_rx`), where `H` is the `(n_rx, n_tx)` channel matrix and `n` is complex AWGN.
The SNR convention here is per transmit stream (`Es/N0` with unit-energy
symbols), so the complex noise variance per receive sample is `10**(-snr_db/10)`
-- the same value the MMSE detector needs.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def rayleigh_channel_matrix(
    n_rx: int, n_tx: int, *, rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """Draw an i.i.d. Rayleigh MIMO channel matrix `H`, shape `(n_rx, n_tx)`.

    Each entry is a unit-variance circularly-symmetric complex Gaussian
    (`CN(0, 1)`): real and imaginary parts are independent `N(0, 1/2)`.

    Args:
        n_rx: Number of receive antennas.
        n_tx: Number of transmit antennas.
        rng: Optional `np.random.Generator` for reproducibility.

    Returns:
        The `(n_rx, n_tx)` complex channel matrix.
    """
    if rng is None:
        rng = np.random.default_rng()
    real = rng.standard_normal((n_rx, n_tx))
    imag = rng.standard_normal((n_rx, n_tx))
    return np.asarray((real + 1j * imag) / np.sqrt(2.0), dtype=np.complex128)


def mimo_noise_variance(snr_db: float) -> float:
    """Complex noise variance per receive sample for a per-stream SNR (`Es/N0`, `Es = 1`)."""
    return float(10.0 ** (-snr_db / 10.0))


def mimo_awgn(
    symbols: ArrayLike,
    channel: ArrayLike,
    snr_db: float,
    *,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """Pass transmit symbols through a fixed MIMO channel and add AWGN.

    Args:
        symbols: Transmit symbols, shape `(n_tx,)` (one channel use) or
            `(n_tx, T)` (a block of `T` uses); unit-energy per antenna.
        channel: Channel matrix `H`, shape `(n_rx, n_tx)`.
        snr_db: Per-stream SNR in dB (`Es/N0`); the added complex noise has
            variance `mimo_noise_variance(snr_db)` per receive sample.
        rng: Optional `np.random.Generator` for reproducibility.

    Returns:
        Received symbols, shape `(n_rx,)` or `(n_rx, T)` matching the input.

    Raises:
        ValueError: If the transmit dimension of `symbols` and `channel` differ.
    """
    x = np.asarray(symbols, dtype=np.complex128)
    H = np.asarray(channel, dtype=np.complex128)
    single = x.ndim == 1
    x2 = x[:, None] if single else x
    if x2.shape[0] != H.shape[1]:
        msg = f'transmit dimension mismatch: symbols have {x2.shape[0]}, H has {H.shape[1]}.'
        raise ValueError(msg)
    if rng is None:
        rng = np.random.default_rng()

    received = H @ x2
    std = np.sqrt(mimo_noise_variance(snr_db) / 2.0)
    received = received + std * (
        rng.standard_normal(received.shape) + 1j * rng.standard_normal(received.shape)
    )
    return np.asarray(received[:, 0] if single else received, dtype=np.complex128)


__all__ = ['mimo_awgn', 'mimo_noise_variance', 'rayleigh_channel_matrix']
