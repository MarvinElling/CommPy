"""Linear FIR channel equalizers: zero-forcing (ZF) and MMSE.

Both design an `n_taps`-length FIR filter `w` that, convolved with a known
channel impulse response, approximates a unit impulse at some `delay` --
i.e. an inverse-channel filter that removes intersymbol interference (ISI).
ZF minimizes residual ISI alone (least squares, ignoring noise); MMSE adds a
noise-variance regularization term (a ridge-regression-style penalty),
trading a little residual ISI for much better noise suppression -- the
classical linear-equalizer tradeoff.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_toeplitz


def _convolution_matrix(channel_taps: NDArray[np.float64], n_taps: int) -> NDArray[np.float64]:
    """`(len(channel_taps) + n_taps - 1, n_taps)` matrix `H` such that `H @ w == convolve(h, w)`."""
    n_channel = len(channel_taps)
    n_out = n_channel + n_taps - 1
    conv_matrix = np.zeros((n_out, n_taps))
    for j in range(n_taps):
        conv_matrix[j:j + n_channel, j] = channel_taps
    return conv_matrix


def zf_equalizer(
    channel_taps: ArrayLike, n_taps: int, delay: int | None = None,
) -> NDArray[np.float64]:
    """Design a zero-forcing FIR equalizer for `channel_taps` (least squares if not invertible).

    Args:
        channel_taps: Known FIR channel impulse response.
        n_taps: Number of equalizer taps.
        delay: Target impulse position in the combined channel+equalizer
            response (`len(channel_taps) + n_taps - 1` samples long).
            Defaults to the midpoint, giving the equalizer some "pre-cursor"
            taps to work with -- generally better-conditioned than delay=0.

    Returns:
        Equalizer FIR taps, length `n_taps`.
    """
    channel = np.asarray(channel_taps, dtype=np.float64)
    conv_matrix = _convolution_matrix(channel, n_taps)
    n_out = conv_matrix.shape[0]
    if delay is None:
        delay = n_out // 2
    target = np.zeros(n_out)
    target[delay] = 1.0
    taps, *_ = np.linalg.lstsq(conv_matrix, target, rcond=None)
    return np.asarray(taps, dtype=np.float64)


def mmse_equalizer(
    channel_taps: ArrayLike, n_taps: int, noise_var: float, delay: int | None = None,
) -> NDArray[np.float64]:
    """Design an MMSE FIR equalizer for `channel_taps` at the given noise level.

    Solves the Wiener-Hopf normal equations `(H^T H + noise_var*I) w = H^T target`.
    `H^T H` is the (symmetric) autocorrelation Toeplitz matrix of the
    channel, so the system is solved via `scipy.linalg.solve_toeplitz`
    (O(n^2)) rather than a generic O(n^3) solver.

    Args:
        channel_taps: Known FIR channel impulse response.
        n_taps: Number of equalizer taps.
        noise_var: Noise variance (regularization strength; 0 reduces to ZF).
        delay: Target impulse position; defaults to the midpoint (see
            `zf_equalizer`).

    Returns:
        Equalizer FIR taps, length `n_taps`.
    """
    channel = np.asarray(channel_taps, dtype=np.float64)
    conv_matrix = _convolution_matrix(channel, n_taps)
    n_out = conv_matrix.shape[0]
    if delay is None:
        delay = n_out // 2
    target = np.zeros(n_out)
    target[delay] = 1.0

    gram = conv_matrix.T @ conv_matrix + noise_var * np.eye(n_taps)
    rhs = conv_matrix.T @ target
    return np.asarray(solve_toeplitz(gram[:, 0], rhs))


__all__ = ['mmse_equalizer', 'zf_equalizer']
