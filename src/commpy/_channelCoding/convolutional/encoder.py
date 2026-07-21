"""Convolutional encoder driven by a `Trellis`."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from commpy._channelCoding.convolutional.trellis import Trellis


class ConvolutionalEncoder:
    """Encodes a bitstream by walking a `Trellis`.

    Encoding is inherently sequential (state depends on all prior bits) but
    O(n); unlike Viterbi decoding, it needs no JIT acceleration.
    """

    def __init__(self, trellis: Trellis) -> None:
        """Construct an encoder for the given trellis."""
        self.trellis = trellis

    def encode(
        self, bits: ArrayLike, initial_state: int = 0, *, terminate: bool = True,
    ) -> tuple[NDArray[np.int64], int]:
        """Encode a message, optionally zero-tail-terminating it.

        Args:
            bits: Message bits.
            initial_state: Encoder shift-register state before the first bit.
            terminate: If True, appends `trellis.memory` zero bits so the
                encoder flushes back to state 0, enabling exact (rather than
                best-effort) Viterbi termination on decode.

        Returns:
            `(output_bits, final_state)`: the encoded bitstream (length
            `len(bits_padded) * trellis.n_outputs`) and the encoder's final
            shift-register state.
        """
        bits_arr = np.asarray(bits, dtype=np.int64)
        if terminate:
            bits_arr = np.concatenate([bits_arr, np.zeros(self.trellis.memory, dtype=np.int64)])

        state = initial_state
        outputs = np.empty((bits_arr.size, self.trellis.n_outputs), dtype=np.int64)
        for i, b in enumerate(bits_arr):
            outputs[i] = self.trellis.output_bits[state, int(b)]
            state = int(self.trellis.next_state[state, int(b)])

        return outputs.reshape(-1), state


__all__ = ['ConvolutionalEncoder']
