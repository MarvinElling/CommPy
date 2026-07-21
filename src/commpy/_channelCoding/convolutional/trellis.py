"""Trellis representation of a rate-1/n binary convolutional code."""

from collections.abc import Sequence

import numpy as np


class Trellis:
    """Trellis diagram for a rate-1/n binary convolutional code.

    States are integers in `[0, 2**memory)`, `memory = constraint_length - 1`,
    where bit `memory-1` holds the most-recently-shifted-in input bit and bit
    0 the oldest. Each generator polynomial is given as an integer with bit
    `constraint_length-1` (the MSB) weighting the *current* input bit, down to
    bit 0 weighting the oldest memory bit -- the convention used by e.g.
    MATLAB's `poly2trellis`.
    """

    def __init__(self, constraint_length: int, generators: Sequence[int]) -> None:
        """Build the trellis for a rate-1/n code.

        Args:
            constraint_length: Total shift-register length (current bit +
                memory), `>= 2`.
            generators: One tap-pattern integer per output bit, each in
                `[1, 2**constraint_length)`.

        Raises:
            ValueError: If `constraint_length < 2`, `generators` is empty, or
                any generator is out of range.
        """
        if constraint_length < 2:
            msg = 'constraint_length must be >= 2.'
            raise ValueError(msg)
        if not generators:
            msg = 'at least one generator polynomial is required.'
            raise ValueError(msg)
        limit = 1 << constraint_length
        if any(g <= 0 or g >= limit for g in generators):
            msg = f'each generator must be in [1, {limit}).'
            raise ValueError(msg)

        k = constraint_length
        m = k - 1
        n_states = 1 << m
        n_outputs = len(generators)

        next_state = np.zeros((n_states, 2), dtype=np.int64)
        output_bits = np.zeros((n_states, 2, n_outputs), dtype=np.int64)

        for s in range(n_states):
            for b in (0, 1):
                window = (b << m) | s
                for j, g in enumerate(generators):
                    output_bits[s, b, j] = (window & g).bit_count() % 2
                next_state[s, b] = (b << (m - 1)) | (s >> 1) if m > 0 else 0

        self.constraint_length = k
        self.memory = m
        self.n_states = n_states
        self.n_outputs = n_outputs
        self.generators = tuple(generators)
        self.next_state = next_state
        self.output_bits = output_bits


__all__ = ['Trellis']
