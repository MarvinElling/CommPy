"""Parametrized contract tests every concrete `Modulator` subclass must satisfy.

New schemes built on the generic `Modulator` ABC get this coverage for free.
"""

import numpy as np
import pytest

from commpy import MPAMModulator, MPSKModulator, MQAMModulator

MODULATORS = [
    MPSKModulator(2),
    MPSKModulator(4),
    MPSKModulator(8),
    MPAMModulator(2),
    MPAMModulator(4),
    MPAMModulator(8),
    MQAMModulator(4),
    MQAMModulator(16),
    MQAMModulator(64),
]


def _ids(mod: object) -> str:
    return f'{type(mod).__name__}(M={mod.M})'  # type: ignore[attr-defined]


@pytest.mark.parametrize('mod', MODULATORS, ids=_ids)
def test_constellation_has_unit_average_energy(mod):
    avg_energy = np.mean(np.abs(mod.constellation)**2)
    assert avg_energy == pytest.approx(1.0)


@pytest.mark.parametrize('mod', MODULATORS, ids=_ids)
def test_every_bit_pattern_is_assigned_exactly_once(mod):
    weights = 1 << np.arange(mod.bits_per_symbol - 1, -1, -1)
    label_values = sorted((mod.bit_labels @ weights).tolist())
    assert label_values == list(range(mod.M))


@pytest.mark.parametrize('mod', MODULATORS, ids=_ids)
def test_modulate_demodulate_round_trip_at_zero_noise(mod, rng):
    n_symbols = 200
    bits = rng.integers(0, 2, n_symbols * mod.bits_per_symbol)
    symbols = mod.modulate(bits)
    recovered = mod.demodulate(symbols)
    np.testing.assert_array_equal(recovered, bits)


@pytest.mark.parametrize('mod', MODULATORS, ids=_ids)
def test_modulate_accepts_list_tuple_and_ndarray_identically(mod):
    bits_list = [0, 1] * (mod.bits_per_symbol * 2)
    bits_tuple = tuple(bits_list)
    bits_array = np.array(bits_list)
    np.testing.assert_array_equal(mod.modulate(bits_list), mod.modulate(bits_tuple))
    np.testing.assert_array_equal(mod.modulate(bits_list), mod.modulate(bits_array))


@pytest.mark.parametrize('mod', MODULATORS, ids=_ids)
def test_soft_demodulate_llr_sign_matches_hard_decision_far_from_boundary(mod, rng):
    # Push each constellation point far from the origin (low relative noise)
    # so the LLR sign should unambiguously agree with the hard decision.
    bits = rng.integers(0, 2, 100 * mod.bits_per_symbol)
    symbols = mod.modulate(bits) * 10.0
    llrs = mod.soft_demodulate(symbols, noise_var=1.0)
    hard_bits = mod.demodulate(symbols)
    # LLR > 0 favors bit 0, LLR < 0 favors bit 1.
    llr_hard_decision = np.where(llrs > 0, 0, 1)
    np.testing.assert_array_equal(llr_hard_decision, hard_bits)


@pytest.mark.parametrize('mod', MODULATORS, ids=_ids)
def test_modulate_rejects_bitstream_not_multiple_of_bits_per_symbol(mod):
    if mod.bits_per_symbol == 1:
        pytest.skip('every length is a multiple of 1 bit/symbol; nothing invalid to construct')
    with pytest.raises(ValueError, match='multiple'):
        mod.modulate([0] * (mod.bits_per_symbol + 1))
