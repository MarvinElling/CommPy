"""Alamouti space-time block coding (the rate-1, 2-transmit-antenna scheme).

Alamouti coding sends two symbols over two time slots from two antennas so that,
with simple linear combining at the receiver, each symbol is recovered with full
(order-`2 * n_rx`) diversity and *no* matrix inversion -- the transmit-diversity
scheme in UMTS/LTE. Over two time slots the antennas send

    slot 1:  antenna 0 -> s0,        antenna 1 -> s1
    slot 2:  antenna 0 -> -conj(s1), antenna 1 -> conj(s0)

which is orthogonal, so the combiner is a per-symbol matched filter.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def alamouti_encode(symbols: ArrayLike) -> NDArray[np.complex128]:
    """Alamouti-encode a symbol stream into a `(2, len(symbols))` transmit matrix.

    Args:
        symbols: Complex symbols, an even-length sequence. Consecutive pairs
            `(s0, s1)` are mapped onto two time slots.

    Returns:
        The `(2, T)` transmit matrix (`row = antenna`, `column = time slot`),
        where `T = len(symbols)`.

    Raises:
        ValueError: If `len(symbols)` is odd.
    """
    s = np.asarray(symbols, dtype=np.complex128)
    if s.size % 2 != 0:
        msg = f'Alamouti coding needs an even number of symbols, got {s.size}.'
        raise ValueError(msg)
    s0 = s[0::2]
    s1 = s[1::2]
    transmit = np.empty((2, s.size), dtype=np.complex128)
    transmit[0, 0::2] = s0
    transmit[1, 0::2] = s1
    transmit[0, 1::2] = -np.conj(s1)
    transmit[1, 1::2] = np.conj(s0)
    return transmit


def alamouti_decode(received: ArrayLike, channel: ArrayLike) -> NDArray[np.complex128]:
    """Combine an Alamouti-coded reception back into soft symbol estimates.

    Args:
        received: Received matrix, shape `(n_rx, T)` with `T` even, from a
            channel that is constant across each two-slot pair.
        channel: Channel matrix `H`, shape `(n_rx, 2)`.

    Returns:
        The `T` recovered soft symbol estimates (scaled to unit channel gain).

    Raises:
        ValueError: If `received` has an odd number of time slots, or its
            receive dimension does not match `channel`.
    """
    y = np.atleast_2d(np.asarray(received, dtype=np.complex128))
    H = np.asarray(channel, dtype=np.complex128)
    if y.shape[1] % 2 != 0:
        msg = f'received must have an even number of time slots, got {y.shape[1]}.'
        raise ValueError(msg)
    if y.shape[0] != H.shape[0]:
        msg = f'receive-antenna mismatch: received has {y.shape[0]}, H has {H.shape[0]}.'
        raise ValueError(msg)

    h0 = H[:, 0][:, None]  # (n_rx, 1), broadcast over the pairs
    h1 = H[:, 1][:, None]
    r1 = y[:, 0::2]  # first slot of each pair, (n_rx, T/2)
    r2 = y[:, 1::2]  # second slot of each pair
    gain = np.sum(np.abs(h0) ** 2 + np.abs(h1) ** 2)
    s0 = np.sum(np.conj(h0) * r1 + h1 * np.conj(r2), axis=0) / gain
    s1 = np.sum(np.conj(h1) * r1 - h0 * np.conj(r2), axis=0) / gain

    out = np.empty(y.shape[1], dtype=np.complex128)
    out[0::2] = s0
    out[1::2] = s1
    return out


__all__ = ['alamouti_decode', 'alamouti_encode']
