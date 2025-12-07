"""Channel models for discrete and analog channels.

This module implements common channel models used in communications
simulations: Binary Symmetric Channel (BSC), Binary Erasure Channel (BEC),
and Additive White Gaussian Noise (AWGN) channel.

Functions are provided as static methods of the `Channels` class so callers
can use `Channels.bsc(...)`, `Channels.bec(...)`, and `Channels.awgn(...)`.
"""

from typing import Any, Optional

import numpy as np


class Channels:
    @staticmethod
    def bsc(bits: Any, p: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Binary Symmetric Channel (BSC).

        Flips each bit independently with probability `p`.

        Parameters
        - bits: array-like of 0/1 or boolean values.
        - p: flip probability in [0, 1].
        - rng: optional `np.random.Generator` for reproducibility.

        Returns
        - ndarray of the same shape as `bits` containing the (possibly)
          flipped bits.
        """
        bits_arr = np.asarray(bits)
        if rng is None:
            rng = np.random.default_rng()
        flips = rng.random(bits_arr.shape) < float(p)

        # Preserve boolean dtype when possible
        if bits_arr.dtype == bool:
            return np.logical_xor(bits_arr, flips)

        # For numeric arrays (0/1), flip by doing 1 - bit where flips=True
        out = bits_arr.copy()
        out = np.where(flips, 1 - out, out)
        return out

    @staticmethod
    def bec(bits: Any, p: float, erasure_value: Any = -1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Binary Erasure Channel (BEC).

        Erases each symbol independently with probability `p`. Erased entries
        are replaced with `erasure_value` (default -1).

        Parameters
        - bits: array-like of symbols (commonly 0/1).
        - p: erasure probability in [0, 1].
        - erasure_value: value used to indicate erasure.
        - rng: optional `np.random.Generator` for reproducibility.

        Returns
        - ndarray (dtype float) with erased entries set to `erasure_value`.
        """
        bits_arr = np.asarray(bits)
        if rng is None:
            rng = np.random.default_rng()
        erasures = rng.random(bits_arr.shape) < float(p)
        out = bits_arr.astype(float).copy()
        out[erasures] = erasure_value
        return out

    @staticmethod
    def awgn(x: Any, snr_db: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Additive White Gaussian Noise (AWGN) channel.

        Adds Gaussian noise to the input signal `x` such that the resulting
        signal-to-noise ratio (SNR) in dB is approximately `snr_db`.

        The function measures the average signal power as `mean(|x|^2)` and
        uses that to determine the noise power: noise_power = signal_power / snr_lin.

        For real-valued inputs, noise is real Gaussian with variance = noise_power.
        For complex-valued inputs, noise has independent real/imag parts each
        with variance = noise_power/2.

        Parameters
        - x: input signal (array-like, real or complex).
        - snr_db: desired SNR in dB (linear SNR = 10**(snr_db/10)).
        - rng: optional `np.random.Generator` for reproducibility.

        Returns
        - noisy signal as ndarray with same shape and dtype as `x`.
        """
        x_arr = np.asarray(x)
        if rng is None:
            rng = np.random.default_rng()

        # average signal power per sample
        signal_power = float(np.mean(np.abs(x_arr) ** 2))
        if signal_power == 0:
            # avoid divide-by-zero; just add unit-variance noise scaled by snr
            signal_power = 1e-16

        snr_lin = 10.0 ** (float(snr_db) / 10.0)
        noise_power = signal_power / snr_lin

        if np.iscomplexobj(x_arr):
            sigma = np.sqrt(noise_power / 2.0)
            noise = sigma * (rng.normal(size=x_arr.shape) + 1j * rng.normal(size=x_arr.shape))
        else:
            sigma = np.sqrt(noise_power)
            noise = sigma * rng.normal(size=x_arr.shape)

        return x_arr + noise


__all__ = ["Channels"]
