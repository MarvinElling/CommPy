"""BER-vs-SNR sweep for 16-QAM, compared to a theoretical reference curve.

Demonstrates commpy's generic Modulator engine plus commpy.Channels.awgn.
"""

import numpy as np
from scipy.special import erfc

from commpy import Channels, MQAMModulator


def main() -> None:
    """Run the 16-QAM BER-vs-SNR sweep."""
    rng = np.random.default_rng(0)
    mod = MQAMModulator(16)
    n_bits = 200_000

    print(f'{"SNR (dB)":>10}  {"measured BER":>14}  {"theoretical BER":>16}')
    for snr_db in [6, 8, 10, 12, 14, 16]:
        bits = rng.integers(0, 2, n_bits - n_bits % mod.bits_per_symbol)
        symbols = mod.modulate(bits)
        received = Channels.awgn(symbols, snr_db, rng=rng)
        recovered = mod.demodulate(received)
        measured_ber = np.mean(recovered != bits)

        # Standard approximate square-QAM BER formula (Gray-coded, high-SNR),
        # in terms of Es/N0 = snr_lin (matching Channels.awgn's convention).
        m, k = mod.M, mod.bits_per_symbol
        snr_lin = 10**(snr_db / 10)
        theoretical_ber = (
            (4 / k) * (1 - 1 / np.sqrt(m)) * 0.5 * erfc(np.sqrt(3 * snr_lin / (2 * (m - 1))))
        )
        print(f'{snr_db:>10}  {measured_ber:>14.5f}  {theoretical_ber:>16.5f}')


if __name__ == '__main__':
    main()
