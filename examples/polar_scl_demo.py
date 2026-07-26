"""Polar code demo: CRC-aided SCL vs. plain SC vs. uncoded BPSK over AWGN.

Builds a length-128 polar code with a CRC-8 for CRC-aided successive-
cancellation list (CA-SCL) decoding -- the 5G-NR-style decoder -- and compares
its coded BER waterfall against plain successive cancellation (list size 1) and
an uncoded BPSK reference, illustrating both the polar coding gain and the extra
gain the CRC-aided list brings.
"""

import matplotlib.pyplot as plt
import numpy as np

from commpy import (
    CRC,
    Channels,
    MPSKModulator,
    PolarCode,
    plot_waterfall,
    simulate_ber,
    simulate_coded_ber,
)


def main() -> None:
    """Run the CA-SCL vs. SC vs. uncoded BPSK comparison."""
    rng = np.random.default_rng(0)
    mod = MPSKModulator(2)  # BPSK
    code = PolarCode(block_length=128, k=64, crc=CRC.crc8(), design_snr_db=2.0)
    snr_db_range = [0.0, 1.0, 2.0, 3.0]

    ca_scl = simulate_coded_ber(
        code, mod, Channels.awgn, snr_db_range,
        blocks_per_batch=30, target_errors=40, max_trials=12_000, rng=rng,
    )
    # Plain SC on the same code: a list size of 1 leaves the CRC unused.
    sc = simulate_coded_ber(
        PolarCode(block_length=128, k=64, design_snr_db=2.0), mod, Channels.awgn, snr_db_range,
        blocks_per_batch=30, target_errors=40, max_trials=12_000, rng=rng,
    )
    uncoded = simulate_ber(
        mod, Channels.awgn, snr_db_range,
        bits_per_batch=4_000, target_errors=200, max_trials=400_000, rng=rng,
    )

    print(f'Polar({code.n}, {code.k}) + CRC-8, rate {code.rate:.3f}\n')
    print(f'{"SNR (dB)":>10}  {"CA-SCL BER":>12}  {"SC BER":>12}  {"uncoded BER":>12}')
    for i, snr_db in enumerate(snr_db_range):
        print(
            f'{snr_db:>10.1f}  {ca_scl.error_rate[i]:>12.5f}  '
            f'{sc.error_rate[i]:>12.5f}  {uncoded.error_rate[i]:>12.5f}',
        )

    ax = plot_waterfall(ca_scl)
    ax.plot(sc.snr_db, sc.error_rate, '^--', label='polar SC (L=1)')
    ax.plot(uncoded.snr_db, uncoded.error_rate, 's--', label='uncoded BPSK')
    ax.set_title(f'Polar({code.n}, {code.k}) CA-SCL coding gain over AWGN')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
