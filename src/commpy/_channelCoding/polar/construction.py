"""Polar-code construction: choosing which synthetic bit-channels to freeze.

Polarization turns `N = 2**n` uses of a channel into `N` synthetic bit-channels
of differing reliability; a polar code carries information on the most reliable
ones and freezes the rest to zero. This module scores the bit-channels (in the
natural, non-bit-reversed index order used by the encoder/decoder) by either:

- `bhattacharyya_reliabilities`: Arikan's Bhattacharyya-parameter recursion
  (exact for the binary erasure channel, a standard closed-form design), or
- `gaussian_approx_reliabilities`: Gaussian-approximation density evolution
  (the practical AWGN design, tracking each channel's mean LLR).

`frozen_mask` turns a score into the boolean frozen/free partition the code uses.
"""

import numpy as np
from numpy.typing import NDArray

_PHI_THRESHOLD = 10.0
_PHI_INV_CAP = 1.0e9


def _check_power_of_two(n: int) -> int:
    """Return `log2(n)`, raising if `n` is not a power of two >= 1."""
    if n < 1 or (n & (n - 1)) != 0:
        msg = f'block length must be a power of two >= 1, got {n}.'
        raise ValueError(msg)
    return int(n).bit_length() - 1


def bhattacharyya_reliabilities(block_length: int, design_snr_db: float) -> NDArray[np.float64]:
    """Score each bit-channel by (minus) its Bhattacharyya parameter (higher = better).

    Uses the erasure-channel recursion `Z -> (2Z - Z**2, Z**2)` seeded with an
    initial erasure probability derived from the design SNR
    (`Z0 = exp(-10**(design_snr_db / 10))`).

    Args:
        block_length: Codeword length `N` (a power of two).
        design_snr_db: Design-point SNR in dB used to seed the recursion.

    Returns:
        A length-`N` score array (natural bit-channel order); larger is more
        reliable.
    """
    n = _check_power_of_two(block_length)
    z = np.array([np.exp(-(10.0 ** (design_snr_db / 10.0)))], dtype=np.float64)
    for _ in range(n):
        upper = 2.0 * z - z**2  # worse (less reliable) child
        lower = z**2  # better child
        z = np.empty(2 * z.size, dtype=np.float64)
        z[0::2] = upper
        z[1::2] = lower
    return -z  # smaller Bhattacharyya parameter -> more reliable -> larger score


def _phi(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Gaussian-approximation `phi` function (mean-LLR density-evolution kernel)."""
    out = np.empty_like(x)
    small = x <= _PHI_THRESHOLD
    xs = x[small]
    out[small] = np.exp(-0.4527 * np.power(xs, 0.859) + 0.0218)
    xl = x[~small]
    out[~small] = np.sqrt(np.pi / xl) * np.exp(-xl / 4.0) * (1.0 - 10.0 / (7.0 * xl))
    out[x == 0.0] = 1.0
    return out


def _phi_inv(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of `_phi` via vectorized bisection (`phi` is monotonically decreasing)."""
    target = np.clip(y, 1e-12, 1.0 - 1e-12)
    lo = np.zeros_like(target)
    hi = np.full_like(target, _PHI_INV_CAP)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        greater = _phi(mid) > target
        lo = np.where(greater, mid, lo)
        hi = np.where(greater, hi, mid)
    return 0.5 * (lo + hi)


def gaussian_approx_reliabilities(block_length: int, design_snr_db: float) -> NDArray[np.float64]:
    """Score each bit-channel by its mean LLR under Gaussian-approximation DE.

    Tracks the mean log-likelihood ratio of each synthetic channel through the
    recursion `m -> (phi_inv(1 - (1 - phi(m))**2), 2 m)`, seeded with the AWGN
    channel LLR mean `m0 = 4 * 10**(design_snr_db / 10)`.

    Args:
        block_length: Codeword length `N` (a power of two).
        design_snr_db: Design-point SNR in dB (interpreted as `Es/N0`).

    Returns:
        A length-`N` score array (natural bit-channel order); a larger mean LLR
        means a more reliable channel.
    """
    n = _check_power_of_two(block_length)
    m = np.array([4.0 * 10.0 ** (design_snr_db / 10.0)], dtype=np.float64)
    for _ in range(n):
        upper = np.minimum(_phi_inv(1.0 - (1.0 - _phi(m)) ** 2), _PHI_INV_CAP)
        lower = 2.0 * m
        m = np.empty(2 * m.size, dtype=np.float64)
        m[0::2] = upper
        m[1::2] = lower
    return m


def frozen_mask(
    block_length: int, n_free: int, *, method: str = 'gaussian', design_snr_db: float = 1.0,
) -> NDArray[np.bool_]:
    """Return the boolean frozen mask selecting the `n_free` most reliable channels.

    Args:
        block_length: Codeword length `N` (a power of two).
        n_free: Number of non-frozen (information + CRC) positions.
        method: `'gaussian'` (Gaussian-approximation) or `'bhattacharyya'`.
        design_snr_db: Design-point SNR in dB for the chosen construction.

    Returns:
        A length-`N` boolean array; `True` marks a frozen position.

    Raises:
        ValueError: If `n_free` is not in `[1, N]` or `method` is unknown.
    """
    if not 1 <= n_free <= block_length:
        msg = f'n_free must be in [1, {block_length}], got {n_free}.'
        raise ValueError(msg)
    if method == 'gaussian':
        score = gaussian_approx_reliabilities(block_length, design_snr_db)
    elif method == 'bhattacharyya':
        score = bhattacharyya_reliabilities(block_length, design_snr_db)
    else:
        msg = f"method must be 'gaussian' or 'bhattacharyya', got {method!r}."
        raise ValueError(msg)

    free_positions = np.argsort(score, kind='stable')[-n_free:]
    mask = np.ones(block_length, dtype=bool)
    mask[free_positions] = False
    return mask


__all__ = [
    'bhattacharyya_reliabilities',
    'frozen_mask',
    'gaussian_approx_reliabilities',
]
