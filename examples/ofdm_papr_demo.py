"""OFDM modulate/demodulate round trip, plus PAPR/PAPR-CCDF analysis.

Demonstrates commpy.OFDMModulator/OFDMDemodulator and the PAPR helpers,
including the theoretical worst case (all-subcarriers-equal -> a single
time-domain impulse, PAPR == n_fft).
"""

import numpy as np

from commpy import MQAMModulator, OFDMDemodulator, OFDMModulator, papr, papr_ccdf, papr_db


def main() -> None:
    """Run the OFDM round-trip and PAPR analysis demo."""
    rng = np.random.default_rng(0)
    n_fft, cp_len = 64, 16
    active = np.arange(4, 60)  # null the DC region and edge guard bands
    mod = OFDMModulator(n_fft, cp_len, active_subcarriers=active)
    demod = OFDMDemodulator(n_fft, cp_len, active_subcarriers=active)
    qam = MQAMModulator(16)

    bits = rng.integers(0, 2, len(active) * qam.bits_per_symbol * 20)
    symbols = qam.modulate(bits)
    tx = mod.modulate(symbols)
    rx_symbols = demod.demodulate(tx)
    assert np.allclose(rx_symbols, symbols, atol=1e-9)
    print(f'OFDM round trip OK: {len(symbols)} symbols across {len(active)} active subcarriers.')

    # Worst-case PAPR: constant subcarriers -> a single time-domain impulse.
    worst_case = OFDMModulator(n_fft, cp_len=0).modulate(np.ones(n_fft, dtype=complex))
    print(f'Worst-case PAPR (constant subcarriers): {papr(worst_case):.1f} == n_fft={n_fft}.')

    # Typical PAPR distribution over many random OFDM symbols.
    plain_mod = OFDMModulator(n_fft, cp_len=0)
    n_symbols = 2000
    blocks = np.empty((n_symbols, n_fft), dtype=complex)
    for i in range(n_symbols):
        sym_bits = rng.integers(0, 2, n_fft * qam.bits_per_symbol)
        blocks[i] = plain_mod.modulate(qam.modulate(sym_bits))

    mean_papr_db = np.mean([papr_db(b) for b in blocks])
    thresholds = [6.0, 8.0, 10.0]
    ccdf = papr_ccdf(blocks, thresholds)
    print(f'Mean PAPR over {n_symbols} random OFDM symbols: {mean_papr_db:.1f} dB')
    for th, p in zip(thresholds, ccdf, strict=True):
        print(f'  P(PAPR > {th:.0f} dB) = {p:.3f}')


if __name__ == '__main__':
    main()
