"""Tests for commpy.Trellis and commpy.ConvolutionalEncoder."""

import numpy as np
import pytest

from commpy import ConvolutionalEncoder, Trellis


@pytest.mark.parametrize(
    ('constraint_length', 'generators'),
    [(3, (0b111, 0b101)), (3, (0b110, 0b101, 0b111)), (4, (0b1101, 0b1011))],
)
def test_trellis_structural_invariants(constraint_length, generators):
    trellis = Trellis(constraint_length, generators)
    assert trellis.n_states == 1 << (constraint_length - 1)
    assert trellis.n_outputs == len(generators)
    assert np.all(trellis.next_state >= 0)
    assert np.all(trellis.next_state < trellis.n_states)
    # All-zero input from state 0 keeps the encoder at state 0 with all-zero output.
    assert trellis.next_state[0, 0] == 0
    assert np.all(trellis.output_bits[0, 0] == 0)


def test_rejects_short_constraint_length():
    with pytest.raises(ValueError, match='constraint_length'):
        Trellis(1, (0b1,))


def test_rejects_empty_generators():
    with pytest.raises(ValueError, match='generator'):
        Trellis(3, ())


def test_rejects_out_of_range_generator():
    with pytest.raises(ValueError, match='generator'):
        Trellis(3, (0b1000,))


def test_nasa_voyager_code_dimensions():
    # The classic (7, 1/2) NASA/Voyager code, generators 171/133 (octal).
    trellis = Trellis(7, (0o171, 0o133))
    assert trellis.n_states == 64
    assert trellis.n_outputs == 2


def test_encode_output_length():
    trellis = Trellis(3, (0b111, 0b101))
    encoder = ConvolutionalEncoder(trellis)
    msg = [1, 0, 1, 1, 0]
    codeword, final_state = encoder.encode(msg, terminate=True)
    assert len(codeword) == (len(msg) + trellis.memory) * trellis.n_outputs
    assert final_state == 0  # zero-tail termination flushes back to state 0


def test_encode_without_termination_does_not_pad():
    trellis = Trellis(3, (0b111, 0b101))
    encoder = ConvolutionalEncoder(trellis)
    msg = [1, 0, 1, 1, 0]
    codeword, _ = encoder.encode(msg, terminate=False)
    assert len(codeword) == len(msg) * trellis.n_outputs


def test_all_zero_message_encodes_to_all_zeros():
    trellis = Trellis(3, (0b111, 0b101))
    encoder = ConvolutionalEncoder(trellis)
    codeword, final_state = encoder.encode(np.zeros(10, dtype=int), terminate=True)
    np.testing.assert_array_equal(codeword, 0)
    assert final_state == 0
