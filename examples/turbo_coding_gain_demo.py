"""Turbo code demo: iterative BCJR decoding vs. uncoded BPSK over AWGN.

Builds a rate-1/3 parallel-concatenated (turbo) code and sweeps its BER with
`simulate_coded_ber` (iterative log-MAP decoding), against an uncoded BPSK
reference, showing the steep turbo waterfall -- the coding gain that first
brought practical codes close to the Shannon limit.
"""

import matplotlib.pyplot as plt
import numpy as np

from commpy import (
    Channels,
    MPSKModulator,
    TurboCode,
    plot_waterfall,
    simulate_ber,
    simulate_coded_ber,
)


def main() -> None:
    """Run the turbo coded-vs-uncoded BPSK BER comparison."""
    rng = np.random.default_rng(0)
    mod = MPSKModulator(2)  # BPSK
    code = TurboCode(k=400, rng=np.random.default_rng(1))
    snr_db_range = [-4.0, -2.0, 0.0, 2.0]

    coded = simulate_coded_ber(
        code, mod, Channels.awgn, snr_db_range,
        blocks_per_batch=6, target_errors=40, max_trials=8_000, rng=rng,
    )
    uncoded = simulate_ber(
        mod, Channels.awgn, snr_db_range,
        bits_per_batch=4_000, target_errors=200, max_trials=400_000, rng=rng,
    )

    print(f'Turbo({code.n}, {code.k}), rate {code.rate:.3f}\n')
    print(f'{"SNR (dB)":>10}  {"turbo BER":>12}  {"uncoded BER":>12}')
    for i, snr_db in enumerate(snr_db_range):
        print(f'{snr_db:>10.1f}  {coded.error_rate[i]:>12.5f}  {uncoded.error_rate[i]:>12.5f}')

    ax = plot_waterfall(coded)
    ax.plot(uncoded.snr_db, uncoded.error_rate, 's--', label='uncoded BPSK')
    ax.set_title(f'Turbo({code.n}, {code.k}) coding gain over AWGN')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
