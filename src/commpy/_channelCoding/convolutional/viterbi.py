"""Viterbi maximum-likelihood decoding of a convolutional trellis.

Branch-metric computation (for every state, input bit, and time step at
once) is fully vectorized in numpy. The add-compare-select (ACS) trellis
traversal that follows is inherently sequential -- each time step's path
metrics depend on the previous one's -- so it is the one hot loop in this
package accelerated via the optional `numba` extra (see `_utils.numba_compat`);
without `numba` installed it still runs correctly, just slower.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from commpy._channelCoding.convolutional.trellis import Trellis
from commpy._utils.numba_compat import njit

_INF = 1.0e18


@njit(cache=True)
def _acs_forward(
    branch_metrics: NDArray[np.float64], next_state: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Forward add-compare-select pass. Returns `(predecessor, input_bit, final_metric)`."""
    n_symbols, n_states, _ = branch_metrics.shape
    path_metric = np.full(n_states, _INF)
    path_metric[0] = 0.0
    predecessor = np.zeros((n_symbols, n_states), dtype=np.int64)
    input_bit = np.zeros((n_symbols, n_states), dtype=np.int64)

    for t in range(n_symbols):
        new_metric = np.full(n_states, _INF)
        for s in range(n_states):
            pm = path_metric[s]
            if pm >= _INF:
                continue
            for b in range(2):
                ns = next_state[s, b]
                cand = pm + branch_metrics[t, s, b]
                if cand < new_metric[ns]:
                    new_metric[ns] = cand
                    predecessor[t, ns] = s
                    input_bit[t, ns] = b
        path_metric = new_metric

    return predecessor, input_bit, path_metric


def _traceback(
    predecessor: NDArray[np.int64], input_bit: NDArray[np.int64], final_state: int, n_symbols: int,
) -> NDArray[np.int64]:
    decoded = np.zeros(n_symbols, dtype=np.int64)
    state = final_state
    for t in range(n_symbols - 1, -1, -1):
        decoded[t] = input_bit[t, state]
        state = int(predecessor[t, state])
    return decoded


def _branch_metrics_hard(trellis: Trellis, symbols: NDArray[np.float64]) -> NDArray[np.float64]:
    diff = trellis.output_bits[None, :, :, :] != symbols[:, None, None, :]
    return np.asarray(np.sum(diff, axis=3).astype(np.float64))


def _branch_metrics_soft(trellis: Trellis, symbols: NDArray[np.float64]) -> NDArray[np.float64]:
    # expected_bit=0 favored by a positive LLR -> lower cost; expected_bit=1 -> higher cost.
    signed = (2 * trellis.output_bits - 1).astype(np.float64)
    return np.asarray(np.sum(signed[None, :, :, :] * symbols[:, None, None, :], axis=3))


def viterbi_decode(
    trellis: Trellis, received: ArrayLike, mode: str = 'hard', *, terminated: bool = True,
) -> NDArray[np.int64]:
    """Maximum-likelihood Viterbi-decode a received sequence.

    Args:
        trellis: The code's `Trellis`.
        received: Length `n_symbols * trellis.n_outputs`. For `mode='hard'`,
            0/1 bits. For `mode='soft'`, LLRs (positive favors bit 0,
            matching `Modulator.soft_demodulate`'s convention).
        mode: `'hard'` or `'soft'`.
        terminated: Whether the encoder was zero-tail-terminated (see
            `ConvolutionalEncoder.encode`); if so, the trellis.memory
            trailing zero bits are dropped from the returned message, and
            decoding assumes (rather than searches for) a final state of 0.

    Returns:
        Decoded message bits.

    Raises:
        ValueError: If `mode` is invalid, or `len(received)` isn't a
            multiple of `trellis.n_outputs`.
    """
    if mode not in ('hard', 'soft'):
        msg = f"mode must be 'hard' or 'soft', got {mode!r}."
        raise ValueError(msg)
    received_arr = np.asarray(received, dtype=np.float64)
    if received_arr.size % trellis.n_outputs != 0:
        msg = f'received length must be a multiple of {trellis.n_outputs}.'
        raise ValueError(msg)

    n_symbols = received_arr.size // trellis.n_outputs
    symbols = received_arr.reshape(n_symbols, trellis.n_outputs)
    branch_metrics = (
        _branch_metrics_hard(trellis, symbols)
        if mode == 'hard'
        else _branch_metrics_soft(trellis, symbols)
    )

    predecessor, input_bit, final_metric = _acs_forward(branch_metrics, trellis.next_state)
    final_state = 0 if terminated else int(np.argmin(final_metric))
    decoded = _traceback(predecessor, input_bit, final_state, n_symbols)

    if terminated:
        decoded = decoded[:n_symbols - trellis.memory]
    return decoded


__all__ = ['viterbi_decode']
