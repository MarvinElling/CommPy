"""LDPC coding-gain demo: coded vs. uncoded BPSK over an AWGN channel.

Builds a regular (3, 6) Gallager LDPC code, then uses `simulate_coded_ber`
(modulate -> AWGN -> soft_demodulate -> belief-propagation decode) and the
plain `simulate_ber` reference to show the coded curve sitting to the left of
the uncoded one -- i.e. the LDPC coding gain -- on a single waterfall plot.
"""

import matplotlib.pyplot as plt
import numpy as np

from commpy import (
    Channels,
    LDPCCode,
    MPSKModulator,
    plot_waterfall,
    simulate_ber,
    simulate_coded_ber,
)


def main() -> None:
    """Run the LDPC coded-vs-uncoded BPSK BER comparison."""
    rng = np.random.default_rng(0)
    mod = MPSKModulator(2)  # BPSK
    code = LDPCCode.from_gallager(n=48, w_c=3, w_r=6, rng=rng)
    snr_db_range = [1.0, 2.0, 3.0, 4.0]

    coded = simulate_coded_ber(
        code, mod, Channels.awgn, snr_db_range,
        blocks_per_batch=25, target_errors=40, max_trials=8_000, rng=rng,
    )
    uncoded = simulate_ber(
        mod, Channels.awgn, snr_db_range,
        bits_per_batch=4_000, target_errors=200, max_trials=200_000, rng=rng,
    )

    print(f'LDPC({code.n}, {code.k}), rate {code.rate:.3f}\n')
    print(f'{"SNR (dB)":>10}  {"coded BER":>12}  {"uncoded BER":>12}')
    for i, snr_db in enumerate(snr_db_range):
        print(f'{snr_db:>10.1f}  {coded.error_rate[i]:>12.5f}  {uncoded.error_rate[i]:>12.5f}')

    ax = plot_waterfall(coded)
    ax.plot(uncoded.snr_db, uncoded.error_rate, 's--', label='uncoded BPSK')
    ax.set_title(f'LDPC({code.n}, {code.k}) coding gain over AWGN')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
