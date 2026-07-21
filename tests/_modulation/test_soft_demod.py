"""Tests for commpy._modulation.soft_demod.compute_llr."""

import numpy as np

from commpy import MPSKModulator
from commpy._modulation.soft_demod import compute_llr


def test_llr_sign_convention_bpsk():
    mod = MPSKModulator(2)
    # BPSK: bit 0 -> some constellation point, bit 1 -> the other.
    zero_symbol = mod.constellation[mod.bit_labels[:, 0] == 0][0]
    one_symbol = mod.constellation[mod.bit_labels[:, 0] == 1][0]

    llr_near_zero = compute_llr(
        np.array([zero_symbol]), mod.constellation, mod.bit_labels, noise_var=1.0,
    )
    llr_near_one = compute_llr(
        np.array([one_symbol]), mod.constellation, mod.bit_labels, noise_var=1.0,
    )
    assert llr_near_zero[0] > 0  # received exactly the bit-0 symbol -> favors bit 0
    assert llr_near_one[0] < 0  # received exactly the bit-1 symbol -> favors bit 1


def test_llr_magnitude_grows_with_lower_noise_variance():
    mod = MPSKModulator(2)
    zero_symbol = mod.constellation[mod.bit_labels[:, 0] == 0][0]
    symbols = np.array([zero_symbol])

    llr_low_noise = compute_llr(symbols, mod.constellation, mod.bit_labels, noise_var=0.1)
    llr_high_noise = compute_llr(symbols, mod.constellation, mod.bit_labels, noise_var=10.0)
    assert abs(llr_low_noise[0]) > abs(llr_high_noise[0])


def test_llr_output_length_matches_bits():
    mod = MPSKModulator(8)
    symbols = mod.constellation[:4]
    llrs = compute_llr(symbols, mod.constellation, mod.bit_labels, noise_var=1.0)
    assert llrs.shape == (4 * mod.bits_per_symbol,)
