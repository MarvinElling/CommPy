"""MIMO demo: spatial-multiplexing detectors, Alamouti diversity, and capacity.

Compares zero-forcing, MMSE, and maximum-likelihood detection for 2x2 spatial
multiplexing (QPSK over i.i.d. Rayleigh fading), shows the transmit-diversity
gain of Alamouti 2x1 over an uncoded SISO link, and prints the ergodic capacity
growing with the antenna count.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from commpy import (
    MQAMModulator,
    alamouti_decode,
    alamouti_encode,
    ergodic_mimo_capacity,
    mimo_awgn,
    mimo_noise_variance,
    ml_detector,
    mmse_detector,
    rayleigh_channel_matrix,
    zf_detector,
)


def _spatial_multiplexing_ber(
    mod: MQAMModulator,
    snr_db_range: list[float],
    n_symbols: int,
    rng: np.random.Generator,
) -> dict[str, NDArray[np.float64]]:
    """Average BER of ZF/MMSE/ML detectors for 2x2 spatial multiplexing."""
    errors = {'ZF': np.zeros(len(snr_db_range)), 'MMSE': np.zeros(len(snr_db_range)),
              'ML': np.zeros(len(snr_db_range))}
    total = 0
    for _ in range(n_symbols):
        H = rayleigh_channel_matrix(2, 2, rng=rng)
        bits = rng.integers(0, 2, 2 * mod.bits_per_symbol)
        x = mod.modulate(bits).reshape(2, 1)
        for i, snr_db in enumerate(snr_db_range):
            y = mimo_awgn(x, H, snr_db, rng=rng)
            detections = {
                'ZF': zf_detector(y, H),
                'MMSE': mmse_detector(y, H, mimo_noise_variance(snr_db)),
                'ML': ml_detector(y, H, mod.constellation),
            }
            for name, x_hat in detections.items():
                errors[name][i] += int(np.sum(mod.demodulate(x_hat.reshape(-1)) != bits))
        total += bits.size
    return {name: err / total for name, err in errors.items()}


def main() -> None:
    """Run the MIMO detector, diversity, and capacity demonstrations."""
    rng = np.random.default_rng(0)
    mod = MQAMModulator(4)  # QPSK
    snr_db_range = [0.0, 4.0, 8.0, 12.0, 16.0]

    ber = _spatial_multiplexing_ber(mod, snr_db_range, n_symbols=600, rng=rng)
    print('2x2 spatial multiplexing (QPSK), BER by detector:')
    print(f'{"SNR (dB)":>9}  {"ZF":>9}  {"MMSE":>9}  {"ML":>9}')
    for i, snr_db in enumerate(snr_db_range):
        print(f'{snr_db:>9.0f}  {ber["ZF"][i]:>9.4f}  {ber["MMSE"][i]:>9.4f}  {ber["ML"][i]:>9.4f}')

    # Transmit diversity: Alamouti 2x1 vs an uncoded SISO link at the same SNR.
    diversity_snr = 8.0
    siso_errors = alamouti_errors = total_bits = 0
    for _ in range(400):
        bits = rng.integers(0, 2, 4 * mod.bits_per_symbol)  # even symbol count
        symbols = mod.modulate(bits)
        h_siso = rayleigh_channel_matrix(1, 1, rng=rng)
        siso_rx = mimo_awgn(symbols[None, :], h_siso, diversity_snr, rng=rng)
        siso_errors += int(np.sum(mod.demodulate((siso_rx / h_siso[0, 0]).reshape(-1)) != bits))
        h_alamouti = rayleigh_channel_matrix(1, 2, rng=rng)
        alamouti_rx = mimo_awgn(alamouti_encode(symbols), h_alamouti, diversity_snr, rng=rng)
        recovered = mod.demodulate(alamouti_decode(alamouti_rx, h_alamouti))
        alamouti_errors += int(np.sum(recovered != bits))
        total_bits += bits.size
    print(
        f'\nTransmit diversity @ {diversity_snr:.0f} dB (QPSK): '
        f'SISO BER {siso_errors / total_bits:.4f}  vs  '
        f'Alamouti 2x1 BER {alamouti_errors / total_bits:.4f}',
    )

    print('\nErgodic capacity at 10 dB (bits/use):')
    for n in (1, 2, 4):
        print(f'  {n}x{n}: {ergodic_mimo_capacity(n, n, 10.0, n_trials=500, rng=rng):.2f}')

    _, ax = plt.subplots(figsize=(7, 5))
    for name, marker in (('ZF', 'o-'), ('MMSE', 's-'), ('ML', '^-')):
        ax.semilogy(snr_db_range, np.maximum(ber[name], 1e-5), marker, label=name)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Bit error rate')
    ax.set_title('2x2 MIMO spatial-multiplexing detectors (QPSK, Rayleigh)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
