"""Recursive systematic convolutional (RSC) codes -- the turbo building block.

Unlike the feedforward `Trellis` used for Viterbi decoding, a turbo constituent
code is *recursive*: the encoder feeds a linear combination of its shift-register
state back to the register input. Each step outputs the systematic bit (the
input itself) plus one parity bit. This module builds the state-transition and
parity tables of such an encoder from its feedback/feedforward polynomials and
provides the streaming encoder.

Polynomials are integers with bit `i` weighting `D**i` (bit 0 = the constant
term), e.g. the classic rate-1/2 RSC has feedback `0o7` (1 + D + D**2) and
feedforward `0o5` (1 + D**2).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _rsc_step(state: int, u: int, feedback: int, feedforward: int, memory: int) -> tuple[int, int]:
    """One RSC step: return `(next_state, parity_bit)` for `input u` from `state`.

    State bit `memory-1` holds the most-recent register value `r_1`, down to bit
    0 for the oldest `r_memory`.
    """
    fed_back = u
    for i in range(1, memory + 1):
        r_i = (state >> (memory - i)) & 1
        fed_back ^= ((feedback >> i) & 1) & r_i
    # parity = g_0 * fed_back  XOR  sum_{i=1..memory} g_i * r_i
    parity = (feedforward & 1) & fed_back
    for i in range(1, memory + 1):
        r_i = (state >> (memory - i)) & 1
        parity ^= ((feedforward >> i) & 1) & r_i
    next_state = ((fed_back << (memory - 1)) | (state >> 1)) & ((1 << memory) - 1)
    return next_state, parity & 1


class RSCTrellis:
    """State-transition / parity tables of a rate-1/2 recursive systematic encoder."""

    def __init__(self, constraint_length: int, feedback: int, feedforward: int) -> None:
        """Build the RSC trellis.

        Args:
            constraint_length: Total register length (current bit + memory), `>= 2`.
            feedback: Feedback polynomial (bit `i` = `D**i` coefficient); bit 0
                must be set.
            feedforward: Feedforward (parity) polynomial.

        Raises:
            ValueError: If `constraint_length < 2` or `feedback`'s constant term
                (bit 0) is unset.
        """
        if constraint_length < 2:
            msg = 'constraint_length must be >= 2.'
            raise ValueError(msg)
        if feedback & 1 == 0:
            msg = 'feedback polynomial must have its constant term (bit 0) set.'
            raise ValueError(msg)

        memory = constraint_length - 1
        n_states = 1 << memory
        next_state = np.zeros((n_states, 2), dtype=np.int64)
        parity = np.zeros((n_states, 2), dtype=np.uint8)
        for s in range(n_states):
            for u in (0, 1):
                ns, p = _rsc_step(s, u, feedback, feedforward, memory)
                next_state[s, u] = ns
                parity[s, u] = p

        self.constraint_length = constraint_length
        self.memory = memory
        self.n_states = n_states
        self.feedback = feedback
        self.feedforward = feedforward
        self.next_state = next_state
        self.parity = parity


def rsc_encode(trellis: RSCTrellis, bits: ArrayLike) -> NDArray[np.uint8]:
    """Encode `bits` through the RSC, returning the parity stream (systematic = `bits`).

    Args:
        trellis: The `RSCTrellis`.
        bits: Information bits; the encoder starts from the all-zero state.

    Returns:
        The parity bit stream (same length as `bits`).
    """
    data = np.asarray(bits, dtype=np.uint8)
    parity = np.zeros(data.size, dtype=np.uint8)
    state = 0
    for t, u in enumerate(data):
        parity[t] = trellis.parity[state, u]
        state = int(trellis.next_state[state, u])
    return parity


__all__ = ['RSCTrellis', 'rsc_encode']
