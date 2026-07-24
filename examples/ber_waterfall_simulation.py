"""Monte-Carlo BER waterfall simulation with confidence intervals.

Demonstrates commpy.simulate_ber (early-stopping Monte-Carlo sweep) and
commpy.plot_waterfall, compared against the closed-form square-QAM BER
formula also used in `qam_ber_curve.py`.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

from commpy import Channels, MQAMModulator, plot_waterfall, simulate_ber


def theoretical_qam_ber(snr_db: np.ndarray, m: int, k: int) -> np.ndarray:
    """Standard approximate square-QAM BER formula (Gray-coded, high-SNR)."""
    snr_lin = 10**(snr_db / 10)
    ber = (4 / k) * (1 - 1 / np.sqrt(m)) * 0.5 * erfc(np.sqrt(3 * snr_lin / (2 * (m - 1))))
    return np.asarray(ber)


def main() -> None:
    """Run the 16-QAM Monte-Carlo BER waterfall demo."""
    rng = np.random.default_rng(0)
    mod = MQAMModulator(16)
    snr_db_range = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0]

    result = simulate_ber(
        mod, Channels.awgn, snr_db_range,
        bits_per_batch=20_000, target_errors=200, max_trials=20_000_000, rng=rng,
    )

    print(f'{"SNR (dB)":>10}  {"measured BER":>14}  {"95% CI":>21}  {"n_bits":>12}')
    for i, snr_db in enumerate(result.snr_db):
        ci = f'[{result.ci_lower[i]:.5f}, {result.ci_upper[i]:.5f}]'
        print(
            f'{snr_db:>10.0f}  {result.error_rate[i]:>14.5f}  {ci:>21}  {result.n_trials[i]:>12}',
        )

    ax = plot_waterfall(
        result, theoretical=lambda snr: theoretical_qam_ber(snr, mod.M, mod.bits_per_symbol),
    )
    ax.set_title('16-QAM BER waterfall (AWGN)')
    plt.show()


if __name__ == '__main__':
    main()
