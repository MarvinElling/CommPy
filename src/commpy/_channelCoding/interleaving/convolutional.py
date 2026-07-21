"""Ramsey type-II convolutional interleaver (multiplexed shift-register lanes).

Unlike a block interleaver, this is a streaming, stateful transform: symbols
round-robin across `n_lanes` FIFOs, lane `i` holding `i * delay_increment`
symbols. Because a given lane is only revisited once every `n_lanes` symbols,
a symbol taking lane `i` is delayed `n_lanes * i * delay_increment` symbols
before it reappears at that same instance's output. The deinterleaver uses
the complementary depths (`(n_lanes - 1 - i) * delay_increment`), so a symbol
taking lane `i` in the interleaver and (necessarily) the same lane `i` in the
deinterleaver experiences total delay
`n_lanes * i * delay_increment + n_lanes * (n_lanes - 1 - i) * delay_increment
= n_lanes * (n_lanes - 1) * delay_increment` -- independent of `i`, so every
symbol is delayed by the same fixed amount and order is restored.
"""

from collections import deque

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ConvolutionalInterleaver:
    """One side (interleave or deinterleave) of a Ramsey type-II convolutional interleaver.

    Stateful across calls to `process` -- construct one instance for the
    transmitter (`is_deinterleaver=False`) and a separate one for the
    receiver (`is_deinterleaver=True`).
    """

    def __init__(
        self, n_lanes: int, delay_increment: int, *, is_deinterleaver: bool = False,
    ) -> None:
        """Construct an interleaver or deinterleaver with `n_lanes` lanes.

        Args:
            n_lanes: Number of round-robin FIFO lanes, `>= 1`.
            delay_increment: Per-lane delay step `D`; lane `i` holds
                `i * D` (or, for the deinterleaver, `(n_lanes-1-i) * D`)
                symbols.
            is_deinterleaver: Use the complementary lane depths that invert
                a matching interleaver's effect.

        Raises:
            ValueError: If `n_lanes < 1` or `delay_increment < 0`.
        """
        if n_lanes < 1:
            msg = 'n_lanes must be >= 1.'
            raise ValueError(msg)
        if delay_increment < 0:
            msg = 'delay_increment must be >= 0.'
            raise ValueError(msg)
        self.n_lanes = n_lanes
        self.delay_increment = delay_increment
        self.total_delay = n_lanes * (n_lanes - 1) * delay_increment
        depths = (
            [(n_lanes - 1 - i) * delay_increment for i in range(n_lanes)]
            if is_deinterleaver
            else [i * delay_increment for i in range(n_lanes)]
        )
        self._lanes = [deque([0] * depth) for depth in depths]
        self._next_lane = 0

    def process(self, data: ArrayLike) -> NDArray[np.generic]:
        """Push `data` through the lanes, round-robin, returning one output symbol per input.

        Faster (low-delay) lanes start emitting real data immediately; slower
        lanes still emit their initial zero-fill until enough symbols have
        passed. By output position `total_delay`, every lane has flushed its
        zero-fill and all outputs correspond to real input from that point on.
        """
        arr = np.asarray(data)
        out = np.empty(arr.shape, dtype=arr.dtype)
        for idx in range(arr.size):
            lane = self._lanes[self._next_lane]
            lane.append(arr.flat[idx])
            out.flat[idx] = lane.popleft()
            self._next_lane = (self._next_lane + 1) % self.n_lanes
        return out


__all__ = ['ConvolutionalInterleaver']
