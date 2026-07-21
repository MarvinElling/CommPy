"""Legacy per-scheme digital modulator classes.

These classes predate the generic, Gray-coded M-PSK/M-QAM/M-PAM engine in
`commpy._modulation` (`MPSKModulator`, `MQAMModulator`, `MPAMModulator`) and
are kept only for backward compatibility with their original call signatures
and numeric conventions (e.g. OOK's asymmetric 0/amplitude levels, and
ASK's arbitrary caller-supplied amplitude lists, are not expressible as
instances of the generic unit-average-energy engine, so these classes are
kept as direct implementations rather than thin wrappers around it). New
code should prefer the generic modulators, which additionally provide
soft-decision (LLR) demodulation.

The original, historically underscore-cased class names (`OOK_Modulator`,
`BPSK_Modulator`, ...) are part of the public, documented API and are kept
as-is rather than renamed to strict CapWords, to avoid breaking every caller.
"""

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


class OOK_Modulator:  # noqa: N801 -- public API name kept for backward compatibility
    """On-Off Keying (OOK) modulator."""

    @staticmethod
    def modulate(data: ArrayLike, amplitude: float = 1) -> NDArray[np.complex128]:
        """Modulate binary data using OOK.

        Args:
            data: Binary data to be modulated (0s and 1s).
            amplitude: Amplitude of the signal.

        Returns:
            Modulated complex baseband signal.
        """
        return (np.array(data) * amplitude).astype(np.complex128)

    @staticmethod
    def demodulate(signal: Iterable[complex], threshold: float = 0.5) -> list[complex]:
        """Demodulate an OOK signal back to binary data.

        Args:
            signal: OOK modulated signal.
            threshold: Threshold above which a sample is decided as '1'.

        Returns:
            Demodulated binary data (1+0j / 0j per symbol).
        """
        # OOK carries no phase information, so the decision is on the real part.
        return [1 + 0j if sample.real >= threshold else 0j for sample in signal]


class BPSK_Modulator:  # noqa: N801 -- public API name kept for backward compatibility
    """Binary Phase Shift Keying (BPSK) modulator."""

    @staticmethod
    def modulate(bitstream: Iterable[int]) -> NDArray[np.complex128]:
        """Modulate binary data using BPSK.

        Args:
            bitstream: Binary data to be modulated (0s and 1s).

        Returns:
            Modulated complex baseband signal.
        """
        return np.array([(1 + 0j) if bit == 1 else (-1 + 0j) for bit in bitstream])

    @staticmethod
    def demodulate(signal: Iterable[complex], threshold: float = 0) -> NDArray[np.int_]:
        """Demodulate a BPSK signal back to binary data.

        Args:
            signal: BPSK modulated signal.
            threshold: Decision threshold on the real axis.

        Returns:
            Demodulated bits.
        """
        return np.array([1 if sample.real > threshold else 0 for sample in signal])


class ASK_2_Modulator:  # noqa: N801 -- public API name kept for backward compatibility
    """2-level Amplitude Shift Keying (ASK) modulator."""

    @staticmethod
    def modulate(
        bitstream: Iterable[int],
        amplitudes: Sequence[float] = (-1, 1),
    ) -> NDArray[np.complex128]:
        """Modulate binary data using 2-level ASK.

        Args:
            bitstream: Binary data to be modulated (0s and 1s).
            amplitudes: Amplitudes `(a0, a1)` for bit 0 and bit 1.

        Returns:
            Modulated complex baseband signal.
        """
        return np.array(
            [(amplitudes[1] if bit == 1 else amplitudes[0]) for bit in bitstream],
            dtype=np.complex128,
        )

    @staticmethod
    def demodulate(signal: Iterable[complex], threshold: float = 0) -> NDArray[np.int_]:
        """Demodulate a 2-level ASK signal back to binary data.

        Args:
            signal: ASK modulated signal.
            threshold: Decision threshold on the real axis.

        Returns:
            Demodulated bits.
        """
        return np.array([1 if sample.real >= threshold else 0 for sample in signal])


class ASK_4_Modulator:  # noqa: N801 -- public API name kept for backward compatibility
    """4-level Amplitude Shift Keying (ASK) modulator."""

    @staticmethod
    def modulate(
        bitstream: ArrayLike,
        amplitudes: Sequence[float] = (-3, -1, 1, 3),
    ) -> NDArray[np.complex128]:
        """Modulate binary data using 4-level ASK.

        Args:
            bitstream: Binary data to be modulated (0s and 1s), length a multiple of 2.
            amplitudes: Amplitudes for the four levels, indexed by the 2-bit group value.

        Returns:
            Modulated complex baseband signal.
        """
        return np.array(
            [amplitudes[int(''.join(map(str, bit)), 2)] for bit in np.reshape(bitstream, (-1, 2))],
            dtype=np.complex128,
        )

    @staticmethod
    def demodulate(
        signal: Iterable[complex],
        levels: Sequence[float] = (-3, -1, 1, 3),
    ) -> NDArray[np.str_]:
        """Demodulate a 4-level ASK signal back to binary data.

        Args:
            signal: 4-level ASK modulated signal.
            levels: The four amplitude levels used at modulation time.

        Returns:
            Demodulated 2-bit groups, each as a `'00'..'11'` string.
        """
        return np.array([format(levels.index(int(sample.real)), '02b') for sample in signal])


class QPSK_Modulator:  # noqa: N801 -- public API name kept for backward compatibility
    """Quadrature Phase Shift Keying (QPSK) modulator."""

    @staticmethod
    def modulate(bitstream: ArrayLike, amplitude: float = 1) -> NDArray[np.complex128]:
        """Modulate binary data using QPSK.

        Args:
            bitstream: Binary data to be modulated (0s and 1s), length a multiple of 2.
                Accepts a list, tuple, or ndarray.
            amplitude: Symbol amplitude.

        Returns:
            Modulated complex baseband signal.

        Raises:
            ValueError: If `len(bitstream)` is not even.
        """
        bits = np.asarray(bitstream, dtype=np.int64)
        if bits.size % 2 != 0:
            msg = 'Bitstream length must be even for QPSK modulation.'
            raise ValueError(msg)

        symbols = []
        for i in range(0, bits.size, 2):
            # Comparing against a tuple of plain ints (rather than the bits
            # slice directly) makes this correct for list, tuple, *and*
            # ndarray input -- a bare `bits[i:i+2] == [0, 0]` only works for
            # list input, since ndarray `==` compares elementwise instead.
            bit_pair = (int(bits[i]), int(bits[i + 1]))
            if bit_pair == (0, 0):
                symbols.append(complex(amplitude, 0))  # 0 degrees
            elif bit_pair == (0, 1):
                symbols.append(complex(0, amplitude))  # 90 degrees
            elif bit_pair == (1, 0):
                symbols.append(complex(-amplitude, 0))  # 180 degrees
            elif bit_pair == (1, 1):
                symbols.append(complex(0, -amplitude))  # 270 degrees

        return np.array(symbols, dtype=np.complex128)

    @staticmethod
    def demodulate(signal: Iterable[complex], amplitude: float = 1) -> NDArray[np.int_]:  # noqa: ARG004 -- kept for signature symmetry with modulate()
        """Demodulate a QPSK signal back to binary data.

        Args:
            signal: QPSK modulated signal.
            amplitude: Unused; kept for signature symmetry with `modulate()`.

        Returns:
            Demodulated bits.
        """
        bitstream: list[int] = []
        for sample in signal:
            if sample.real > 0 and sample.imag >= 0:
                bitstream.extend([0, 0])  # 0 degrees
            elif sample.real <= 0 and sample.imag > 0:
                bitstream.extend([0, 1])  # 90 degrees
            elif sample.real < 0 and sample.imag <= 0:
                bitstream.extend([1, 0])  # 180 degrees
            elif sample.real >= 0 and sample.imag < 0:
                bitstream.extend([1, 1])  # 270 degrees

        return np.array(bitstream)


class PSK_8_Modulator:  # noqa: N801 -- public API name kept for backward compatibility
    """8-Phase Shift Keying (8-PSK) modulator."""

    @staticmethod
    def modulate(bitstream: ArrayLike, amplitude: float = 1) -> NDArray[np.complex128]:
        """Modulate binary data using 8-PSK.

        Args:
            bitstream: Binary data to be modulated (0s and 1s), length a multiple of 3.
            amplitude: Symbol amplitude.

        Returns:
            Modulated complex baseband signal.

        Raises:
            ValueError: If `len(bitstream)` is not a multiple of 3.
        """
        bits = np.asarray(bitstream, dtype=np.int64)
        if bits.size % 3 != 0:
            msg = 'Bitstream length must be a multiple of 3 for 8-PSK modulation.'
            raise ValueError(msg)

        symbols = []
        for i in range(0, bits.size, 3):
            bit_triplet = bits[i:i + 3]
            index = int(''.join(str(b) for b in bit_triplet), 2)
            angle = index * np.pi / 4  # 45 degrees per symbol
            symbols.append(complex(amplitude * np.cos(angle), amplitude * np.sin(angle)))

        return np.array(symbols, dtype=np.complex128)

    @staticmethod
    def demodulate(signal: Iterable[complex], amplitude: float = 1) -> NDArray[np.str_]:  # noqa: ARG004 -- kept for signature symmetry with modulate()
        """Demodulate an 8-PSK signal back to binary data.

        Args:
            signal: 8-PSK modulated signal.
            amplitude: Unused; kept for signature symmetry with `modulate()`.

        Returns:
            Demodulated bits, as an array of individual `'0'`/`'1'` characters.
        """
        bitstream: list[str] = []
        for sample in signal:
            angle = np.angle(sample) % (2 * np.pi)
            index = round(angle / (np.pi / 4)) % 8
            bitstream.extend(format(index, '03b'))  # 3-bit binary string, one char per bit

        return np.array(bitstream)


__all__ = [
    'ASK_2_Modulator',
    'ASK_4_Modulator',
    'BPSK_Modulator',
    'OOK_Modulator',
    'PSK_8_Modulator',
    'QPSK_Modulator',
]
