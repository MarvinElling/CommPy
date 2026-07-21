"""Tests for commpy OFDM: OFDMModulator/OFDMDemodulator and PAPR analysis."""

import numpy as np
import pytest

from commpy import MQAMModulator, OFDMDemodulator, OFDMModulator, papr, papr_ccdf, papr_db


def test_round_trip_all_subcarriers(rng):
    n_fft, cp_len = 64, 16
    mod = OFDMModulator(n_fft, cp_len)
    demod = OFDMDemodulator(n_fft, cp_len)

    symbols = (rng.integers(0, 2, n_fft * 5) * 2 - 1).astype(complex)
    time_domain = mod.modulate(symbols)
    recovered = demod.demodulate(time_domain)
    np.testing.assert_allclose(recovered, symbols, atol=1e-9)


def test_round_trip_with_active_subcarrier_subset(rng):
    n_fft, cp_len = 64, 16
    active = np.arange(4, 60)  # null the DC region and edge guard bands
    mod = OFDMModulator(n_fft, cp_len, active_subcarriers=active)
    demod = OFDMDemodulator(n_fft, cp_len, active_subcarriers=active)

    qam = MQAMModulator(16)
    bits = rng.integers(0, 2, len(active) * qam.bits_per_symbol * 3)
    symbols = qam.modulate(bits)
    time_domain = mod.modulate(symbols)
    recovered = demod.demodulate(time_domain)
    np.testing.assert_allclose(recovered, symbols, atol=1e-9)


def test_cyclic_prefix_matches_tail_of_ifft():
    n_fft, cp_len = 16, 4
    mod = OFDMModulator(n_fft, cp_len)
    symbols = np.arange(1, n_fft + 1).astype(complex)
    out = mod.modulate(symbols)
    cp = out[:cp_len]
    tail_of_symbol = out[n_fft:n_fft + cp_len]
    np.testing.assert_allclose(cp, tail_of_symbol)


def test_output_length():
    n_fft, cp_len = 32, 8
    mod = OFDMModulator(n_fft, cp_len)
    symbols = np.ones(n_fft * 3, dtype=complex)
    out = mod.modulate(symbols)
    assert len(out) == 3 * (n_fft + cp_len)


@pytest.mark.parametrize(('n_fft', 'cp_len'), [(0, 0), (-4, 0)])
def test_rejects_invalid_n_fft(n_fft, cp_len):
    with pytest.raises(ValueError, match='n_fft'):
        OFDMModulator(n_fft, cp_len)


def test_rejects_invalid_cp_len():
    with pytest.raises(ValueError, match='cp_len'):
        OFDMModulator(16, 16)  # cp_len must be < n_fft
    with pytest.raises(ValueError, match='cp_len'):
        OFDMModulator(16, -1)


def test_rejects_out_of_range_active_subcarriers():
    with pytest.raises(ValueError, match='active_subcarriers'):
        OFDMModulator(16, 4, active_subcarriers=[0, 1, 20])


def test_rejects_duplicate_active_subcarriers():
    with pytest.raises(ValueError, match='duplicates'):
        OFDMModulator(16, 4, active_subcarriers=[0, 1, 1])


def test_modulate_rejects_wrong_symbol_count():
    mod = OFDMModulator(16, 4)
    with pytest.raises(ValueError, match='multiple'):
        mod.modulate(np.ones(10, dtype=complex))


def test_demodulate_rejects_wrong_sample_count():
    demod = OFDMDemodulator(16, 4)
    with pytest.raises(ValueError, match='multiple'):
        demod.demodulate(np.ones(10, dtype=complex))


def test_constant_subcarriers_give_worst_case_papr():
    # IFFT of a constant frequency-domain signal is a single time-domain
    # impulse: the theoretical worst-case PAPR, exactly n_fft.
    n_fft = 64
    mod = OFDMModulator(n_fft, cp_len=0)
    symbols = np.ones(n_fft, dtype=complex)
    time_domain = mod.modulate(symbols)
    assert papr(time_domain) == pytest.approx(n_fft, rel=1e-6)


def test_papr_db_consistent_with_linear_papr():
    n_fft = 64
    mod = OFDMModulator(n_fft, cp_len=0)
    rng = np.random.default_rng(5)
    symbols = (rng.integers(0, 2, n_fft) * 2 - 1).astype(complex)
    time_domain = mod.modulate(symbols)
    assert papr_db(time_domain) == pytest.approx(10 * np.log10(papr(time_domain)))


def test_papr_ccdf_is_monotonically_non_increasing(rng):
    n_fft = 64
    mod = OFDMModulator(n_fft, cp_len=0)
    n_blocks = 300
    blocks = np.empty((n_blocks, n_fft), dtype=complex)
    for i in range(n_blocks):
        symbols = (rng.integers(0, 2, n_fft) * 2 - 1).astype(complex)
        blocks[i] = mod.modulate(symbols)

    thresholds = np.linspace(0, 12, 25)
    ccdf = papr_ccdf(blocks, thresholds)
    assert np.all(np.diff(ccdf) <= 1e-12)
    assert ccdf[0] <= 1.0
    assert ccdf[-1] >= 0.0
