"""OFDM (Orthogonal Frequency-Division Multiplexing) modulation and PAPR analysis.

`OFDMModulator`/`OFDMDemodulator` need no up-conversion machinery of their
own: the time-domain samples they produce are ordinary complex baseband
symbols and can be fed straight into `IQWaveform(I, Q, ...)` for RF
up-conversion/plotting, exactly like any other modulator's output.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import fft, ifft


class OFDMModulator:
    """Maps frequency-domain symbols onto subcarriers, IFFTs, and adds a cyclic prefix."""

    def __init__(
        self, n_fft: int, cp_len: int, active_subcarriers: ArrayLike | None = None,
    ) -> None:
        """Construct an OFDM modulator.

        Args:
            n_fft: IFFT size (total number of subcarriers).
            cp_len: Cyclic prefix length in samples, `0 <= cp_len < n_fft`.
            active_subcarriers: Indices (in `[0, n_fft)`) that carry data;
                the rest are nulled. Defaults to all `n_fft` subcarriers.

        Raises:
            ValueError: If `n_fft <= 0`, `cp_len` is out of range, or
                `active_subcarriers` contains an out-of-range or duplicate index.
        """
        if n_fft <= 0:
            msg = 'n_fft must be positive.'
            raise ValueError(msg)
        if not 0 <= cp_len < n_fft:
            msg = f'cp_len must be in [0, {n_fft}), got {cp_len}.'
            raise ValueError(msg)
        self.n_fft = n_fft
        self.cp_len = cp_len
        if active_subcarriers is None:
            active_subcarriers = np.arange(n_fft)
        active = np.asarray(active_subcarriers, dtype=np.int64)
        if active.size == 0 or np.any(active < 0) or np.any(active >= n_fft):
            msg = f'active_subcarriers must be non-empty indices in [0, {n_fft}).'
            raise ValueError(msg)
        if len(np.unique(active)) != active.size:
            msg = 'active_subcarriers must not contain duplicates.'
            raise ValueError(msg)
        self.active_subcarriers = active
        self.n_active = active.size

    def modulate(self, symbols: ArrayLike) -> NDArray[np.complex128]:
        """Modulate frequency-domain symbols into time-domain OFDM samples.

        Args:
            symbols: Complex symbols, length a multiple of `n_active`; each
                consecutive group of `n_active` becomes one OFDM symbol.

        Returns:
            Time-domain samples: `(n_fft + cp_len)` per OFDM symbol, concatenated.

        Raises:
            ValueError: If `len(symbols)` is not a multiple of `n_active`.
        """
        syms = np.asarray(symbols, dtype=np.complex128)
        if syms.size % self.n_active != 0:
            msg = f'symbols length must be a multiple of n_active={self.n_active}.'
            raise ValueError(msg)
        n_ofdm_symbols = syms.size // self.n_active
        syms = syms.reshape(n_ofdm_symbols, self.n_active)

        freq_domain = np.zeros((n_ofdm_symbols, self.n_fft), dtype=np.complex128)
        freq_domain[:, self.active_subcarriers] = syms
        time_domain = ifft(freq_domain, axis=1)

        if self.cp_len > 0:
            with_cp = np.concatenate([time_domain[:, -self.cp_len:], time_domain], axis=1)
        else:
            with_cp = time_domain
        return np.asarray(with_cp.reshape(-1))


class OFDMDemodulator:
    """Strips the cyclic prefix, FFTs, and extracts the active-subcarrier symbols."""

    def __init__(
        self, n_fft: int, cp_len: int, active_subcarriers: ArrayLike | None = None,
    ) -> None:
        """Construct an OFDM demodulator matching an `OFDMModulator`'s parameters.

        Raises:
            ValueError: Same conditions as `OFDMModulator.__init__`.
        """
        # Reuse OFDMModulator's validation by constructing one; only its
        # (identical) parameter bookkeeping is needed here.
        reference = OFDMModulator(n_fft, cp_len, active_subcarriers)
        self.n_fft = reference.n_fft
        self.cp_len = reference.cp_len
        self.active_subcarriers = reference.active_subcarriers
        self.n_active = reference.n_active

    def demodulate(self, samples: ArrayLike) -> NDArray[np.complex128]:
        """Demodulate time-domain OFDM samples back into frequency-domain symbols.

        Args:
            samples: Time-domain samples, length a multiple of `n_fft + cp_len`.

        Returns:
            Recovered complex symbols from the active subcarriers.

        Raises:
            ValueError: If `len(samples)` is not a multiple of `n_fft + cp_len`.
        """
        arr = np.asarray(samples, dtype=np.complex128)
        symbol_len = self.n_fft + self.cp_len
        if arr.size % symbol_len != 0:
            msg = f'samples length must be a multiple of n_fft + cp_len = {symbol_len}.'
            raise ValueError(msg)
        n_ofdm_symbols = arr.size // symbol_len
        arr = arr.reshape(n_ofdm_symbols, symbol_len)
        time_domain = arr[:, self.cp_len:]
        freq_domain = fft(time_domain, axis=1)
        return np.asarray(freq_domain[:, self.active_subcarriers].reshape(-1))


def papr(signal: ArrayLike) -> float:
    """Peak-to-Average Power Ratio (linear, not dB) of a time-domain signal block."""
    arr = np.asarray(signal, dtype=np.complex128)
    power = np.abs(arr) ** 2
    return float(np.max(power) / np.mean(power))


def papr_db(signal: ArrayLike) -> float:
    """Peak-to-Average Power Ratio in dB."""
    return float(10.0 * np.log10(papr(signal)))


def papr_ccdf(ofdm_symbols: ArrayLike, thresholds_db: ArrayLike) -> NDArray[np.float64]:
    """Empirical complementary CDF of PAPR across many OFDM symbol blocks.

    Args:
        ofdm_symbols: Shape `(n_symbols, n_fft)`: one time-domain OFDM symbol
            (no cyclic prefix) per row.
        thresholds_db: PAPR thresholds (dB) to evaluate the CCDF at.

    Returns:
        `P(PAPR > threshold)` for each threshold, same shape as `thresholds_db`.
    """
    arr = np.asarray(ofdm_symbols, dtype=np.complex128)
    power = np.abs(arr) ** 2
    peak = np.max(power, axis=1)
    avg = np.mean(power, axis=1)
    papr_values_db = 10.0 * np.log10(peak / avg)
    thresholds = np.asarray(thresholds_db, dtype=np.float64)
    return np.array([np.mean(papr_values_db > th) for th in thresholds])


__all__ = ['OFDMDemodulator', 'OFDMModulator', 'papr', 'papr_ccdf', 'papr_db']
