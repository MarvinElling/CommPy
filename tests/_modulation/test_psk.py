"""Tests for commpy.MPSKModulator, including a theoretical BER-vs-SNR check."""

import numpy as np
from scipy.special import erfc

from commpy import Channels, MPSKModulator


def test_bpsk_constellation_is_plus_minus_one():
    mod = MPSKModulator(2)
    np.testing.assert_allclose(sorted(mod.constellation.real), [-1.0, 1.0])


def test_qpsk_constellation_at_45_degree_offsets():
    mod = MPSKModulator(4)
    angles = sorted(np.angle(mod.constellation) % (2 * np.pi))
    expected = sorted((2 * np.pi * np.arange(4) / 4) % (2 * np.pi))
    np.testing.assert_allclose(angles, expected, atol=1e-9)


def test_adjacent_constellation_points_differ_by_one_bit():
    mod = MPSKModulator(8)
    for i in range(mod.M):
        j = (i + 1) % mod.M
        diff = mod.bit_labels[i] != mod.bit_labels[j]
        assert diff.sum() == 1


def test_bpsk_measured_ber_matches_theoretical_curve(rng):
    mod = MPSKModulator(2)
    snr_db = 6.0
    n_bits = 200_000

    bits = rng.integers(0, 2, n_bits)
    symbols = mod.modulate(bits)
    received = Channels.awgn(symbols, snr_db, rng=rng)
    recovered = mod.demodulate(received)

    measured_ber = np.mean(recovered != bits)
    snr_lin = 10**(snr_db / 10)
    theoretical_ber = 0.5 * erfc(np.sqrt(snr_lin))

    assert abs(measured_ber - theoretical_ber) < 0.002
