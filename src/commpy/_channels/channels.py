"""Channel models for discrete and analog channels.

This module implements common channel models used in communications
simulations: Binary Symmetric Channel (BSC), Binary Erasure Channel (BEC),
Additive White Gaussian Noise (AWGN), fading, and bursty-error channels.

Functions are provided as static methods of the `Channels` class so callers
can use `Channels.bsc(...)`, `Channels.bec(...)`, `Channels.awgn(...)`, etc.
"""

import numpy as np
from numpy.typing import ArrayLike


class Channels:
    """Namespace of discrete and analog channel impairment models."""

    @staticmethod
    def bsc(bits: ArrayLike, p: float, rng: np.random.Generator | None = None) -> np.ndarray:
        """Binary Symmetric Channel (BSC).

        Flips each bit independently with probability `p`.

        Args:
            bits: Array-like of 0/1 or boolean values.
            p: Flip probability in [0, 1].
            rng: Optional `np.random.Generator` for reproducibility.

        Returns:
            Array of the same shape as `bits` containing the (possibly)
            flipped bits.
        """
        bits_arr = np.asarray(bits)
        if rng is None:
            rng = np.random.default_rng()
        flips = rng.random(bits_arr.shape) < float(p)

        # Preserve boolean dtype when possible.
        if bits_arr.dtype == bool:
            return np.asarray(np.logical_xor(bits_arr, flips))

        # For numeric arrays (0/1), flip by doing 1 - bit where flips=True.
        return np.where(flips, 1 - bits_arr, bits_arr)

    @staticmethod
    def bec(
        bits: ArrayLike,
        p: float,
        erasure_value: float = -1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Binary Erasure Channel (BEC).

        Erases each symbol independently with probability `p`. Erased entries
        are replaced with `erasure_value` (default -1).

        Args:
            bits: Array-like of symbols (commonly 0/1).
            p: Erasure probability in [0, 1].
            erasure_value: Value used to indicate erasure.
            rng: Optional `np.random.Generator` for reproducibility.

        Returns:
            Array (dtype float) with erased entries set to `erasure_value`.
        """
        bits_arr = np.asarray(bits)
        if rng is None:
            rng = np.random.default_rng()
        erasures = rng.random(bits_arr.shape) < float(p)
        out = bits_arr.astype(float)
        out[erasures] = erasure_value
        return out

    @staticmethod
    def awgn(
        x: ArrayLike,
        snr_db: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Additive White Gaussian Noise (AWGN) channel.

        Adds Gaussian noise to the input signal `x` such that the resulting
        signal-to-noise ratio (SNR) in dB is approximately `snr_db`.

        The function measures the average signal power as `mean(|x|^2)` and
        uses that to determine the noise power: `noise_power = signal_power / snr_lin`.

        For real-valued inputs, noise is real Gaussian with variance = noise_power.
        For complex-valued inputs, noise has independent real/imag parts each
        with variance = noise_power/2.

        Args:
            x: Input signal (array-like, real or complex).
            snr_db: Desired SNR in dB (linear SNR = 10**(snr_db/10)).
            rng: Optional `np.random.Generator` for reproducibility.

        Returns:
            Noisy signal as an array with the same shape and dtype as `x`.
        """
        x_arr = np.asarray(x)
        if rng is None:
            rng = np.random.default_rng()

        # Average signal power per sample.
        signal_power = float(np.mean(np.abs(x_arr)**2))
        if signal_power == 0:
            # Avoid divide-by-zero; just add unit-variance noise scaled by snr.
            signal_power = 1e-16

        snr_lin = 10.0**(float(snr_db) / 10.0)
        noise_power = signal_power / snr_lin

        if np.iscomplexobj(x_arr):
            sigma = np.sqrt(noise_power / 2.0)
            noise = sigma * (rng.normal(size=x_arr.shape) + 1j * rng.normal(size=x_arr.shape))
        else:
            sigma = np.sqrt(noise_power)
            noise = sigma * rng.normal(size=x_arr.shape)

        return np.asarray(x_arr + noise)

    @staticmethod
    def rayleigh(
        x: ArrayLike,
        snr_db: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Rayleigh fading channel + AWGN.

        Args:
            x: Input signal (array-like, real or complex).
            snr_db: Desired SNR in dB.
            rng: Optional random generator.

        Returns:
            Output after Rayleigh fading and AWGN.
        """
        x_arr = np.asarray(x)
        if rng is None:
            rng = np.random.default_rng()

        # Rayleigh fading coefficients (complex or real, unit average power).
        if np.iscomplexobj(x_arr):
            h = (rng.normal(size=x_arr.shape) + 1j * rng.normal(size=x_arr.shape)) / np.sqrt(2)
        else:
            h = rng.rayleigh(scale=1.0, size=x_arr.shape)

        faded = x_arr * h
        return Channels.awgn(faded, snr_db, rng)

    @staticmethod
    def rician(
        x: ArrayLike,
        snr_db: float,
        k_factor: float = 10.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Rician fading channel + AWGN.

        Args:
            x: Input signal (array-like, real or complex).
            snr_db: Desired SNR in dB.
            k_factor: Rician K-factor (power ratio of LOS to scattered path).
            rng: Optional random generator.

        Returns:
            Output after Rician fading and AWGN.
        """
        x_arr = np.asarray(x)
        if rng is None:
            rng = np.random.default_rng()

        # Rician fading: specular (LOS) + scattered component.
        k_lin = float(k_factor)
        s = np.sqrt(k_lin / (k_lin + 1))
        sigma = np.sqrt(1 / (2 * (k_lin + 1)))

        if np.iscomplexobj(x_arr):
            h = s + sigma * (rng.normal(size=x_arr.shape) + 1j * rng.normal(size=x_arr.shape))
        else:
            h = s + sigma * rng.normal(size=x_arr.shape)

        faded = x_arr * h
        return Channels.awgn(faded, snr_db, rng)

    @staticmethod
    def z_channel(bits: ArrayLike, p: float, rng: np.random.Generator | None = None) -> np.ndarray:
        """Z-Channel.

        Flips 1 to 0 with probability `p`; 0 always stays 0.

        Args:
            bits: Array-like (0/1 or bool).
            p: Error probability for 1 -> 0.
            rng: Optional random generator.

        Returns:
            Array of the same shape as `bits`.
        """
        bits_arr = np.asarray(bits)
        if rng is None:
            rng = np.random.default_rng()
        flips = (bits_arr == 1) & (rng.random(bits_arr.shape) < p)
        out = bits_arr.copy()
        out[flips] = 0
        return out

    @staticmethod
    def gilbert_elliott(  # noqa: PLR0913 -- each parameter names a distinct, physically meaningful Markov-model quantity
        bits: ArrayLike,
        p_gb: float,
        p_bg: float,
        p_good: float = 0.0,
        p_bad: float = 0.2,
        rng: np.random.Generator | None = None,
        init_state: str = 'good',
    ) -> np.ndarray:
        """Gilbert-Elliott two-state bursty channel.

        Args:
            bits: Input bits, array-like (0/1 or bool).
            p_gb: Probability of switching Good -> Bad.
            p_bg: Probability of switching Bad -> Good.
            p_good: Error probability in the 'good' state.
            p_bad: Error probability in the 'bad' state.
            rng: Optional random number generator.
            init_state: Initial state, `"good"` or `"bad"`.

        Returns:
            Output bits after passing through the channel.
        """
        bits_arr = np.asarray(bits)
        if rng is None:
            rng = np.random.default_rng()
        n = bits_arr.size

        state = 0 if init_state == 'good' else 1
        states = np.zeros(n, dtype=int)
        for i in range(1, n):
            if state == 0 and rng.random() < p_gb:
                state = 1
            elif state == 1 and rng.random() < p_bg:
                state = 0
            states[i] = state

        errors = np.where(states == 0, rng.random(n) < p_good, rng.random(n) < p_bad)
        if bits_arr.dtype == bool:
            return np.asarray(np.logical_xor(bits_arr, errors))
        return np.where(errors, 1 - bits_arr, bits_arr)

    @staticmethod
    def quantize(
        x: ArrayLike,
        bits: int = 8,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> np.ndarray:
        """Uniform quantization of an analog signal.

        Args:
            x: Input real signal.
            bits: Number of quantization bits.
            vmin: Optional lower clipping bound (defaults to `min(x)`).
            vmax: Optional upper clipping bound (defaults to `max(x)`).

        Returns:
            Quantized signal, same shape as `x`.
        """
        x_arr = np.asarray(x)
        if vmin is None:
            vmin = np.min(x_arr)
        if vmax is None:
            vmax = np.max(x_arr)
        levels = 2**bits
        x_clipped = np.clip(x_arr, vmin, vmax)
        q = np.round((x_clipped - vmin) / (vmax - vmin) * (levels - 1))
        return np.asarray(vmin + q / (levels - 1) * (vmax - vmin))


__all__ = ['Channels']
