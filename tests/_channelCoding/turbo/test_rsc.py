"""Tests for the recursive systematic convolutional (RSC) constituent encoder.

The structural invariant a valid convolutional trellis must satisfy is that, for
each fixed input bit, the state-transition map is a bijection (a permutation of
the states) -- this is what makes the code linear and losslessly decodable. Also
checks the recursive encoder's defining behaviors (zero input from the zero state
stays at zero; parity bits are binary) and argument validation.
"""

import numpy as np
import pytest

from commpy._channelCoding.turbo.rsc import RSCTrellis, rsc_encode


@pytest.mark.parametrize(
    ('constraint_length', 'feedback', 'feedforward'),
    [(3, 0o7, 0o5), (4, 0o13, 0o15), (3, 0o5, 0o7)],
)
def test_state_transition_is_a_bijection_per_input(constraint_length, feedback, feedforward):
    trellis = RSCTrellis(constraint_length, feedback, feedforward)
    for u in (0, 1):
        transitions = trellis.next_state[:, u]
        np.testing.assert_array_equal(np.sort(transitions), np.arange(trellis.n_states))
    assert set(np.unique(trellis.parity).tolist()) <= {0, 1}


def test_zero_input_stays_in_zero_state():
    trellis = RSCTrellis(3, 0o7, 0o5)
    # From the all-zero state, input 0 must return to the zero state with 0 parity.
    assert trellis.next_state[0, 0] == 0
    assert trellis.parity[0, 0] == 0
    parity = rsc_encode(trellis, np.zeros(20, dtype=np.uint8))
    np.testing.assert_array_equal(parity, np.zeros(20, dtype=np.uint8))


def test_encode_is_deterministic_and_length_preserving(rng):
    trellis = RSCTrellis(3, 0o7, 0o5)
    bits = rng.integers(0, 2, 50).astype(np.uint8)
    first = rsc_encode(trellis, bits)
    second = rsc_encode(trellis, bits)
    assert first.shape == bits.shape
    np.testing.assert_array_equal(first, second)


def test_invalid_construction_raises():
    with pytest.raises(ValueError, match='constraint_length'):
        RSCTrellis(1, 0o7, 0o5)
    with pytest.raises(ValueError, match='constant term'):
        RSCTrellis(3, 0o6, 0o5)  # feedback 0o6 = 110b has no constant term
