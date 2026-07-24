"""SDR IQ recording interoperability: raw binary and SigMF file I/O.

Demonstrates commpy.write_iq/read_iq (headerless, GNU-Radio-compatible raw
complex samples) and commpy.write_sigmf/read_sigmf (SigMF-style
`.sigmf-data` + `.sigmf-meta` recordings, https://github.com/sigmf/SigMF).
"""

import tempfile
from pathlib import Path

import numpy as np

from commpy import MQAMModulator, read_iq, read_sigmf, write_iq, write_sigmf


def main() -> None:
    """Run the IQ file round-trip demo."""
    rng = np.random.default_rng(0)
    mod = MQAMModulator(16)
    bits = rng.integers(0, 2, mod.bits_per_symbol * 5_000)
    symbols = mod.modulate(bits)

    with tempfile.TemporaryDirectory() as tmp:
        # Raw binary: headerless complex64 dump, readable by GNU Radio's
        # blocks.file_source (or any tool expecting a raw IQ sample stream).
        raw_path = Path(tmp) / 'recording.cf32'
        write_iq(raw_path, symbols)
        raw_recovered = read_iq(raw_path)
        assert np.array_equal(raw_recovered, symbols.astype(np.complex64))
        n_bytes = raw_path.stat().st_size
        print(f'Raw IQ round trip OK: {raw_recovered.size} samples ({n_bytes} bytes).')

        # SigMF: adds a JSON sidecar with sample rate / center frequency / description.
        sigmf_base = Path(tmp) / 'capture'
        write_sigmf(
            sigmf_base, symbols, sample_rate=1e6, center_freq=915e6,
            description='16-QAM test capture', author='commpy',
        )
        sigmf_recovered, meta = read_sigmf(sigmf_base)
        assert np.array_equal(sigmf_recovered, symbols.astype(np.complex64))
        print(f'SigMF round trip OK: {sigmf_recovered.size} samples.')
        print(f'  core:datatype    = {meta["global"]["core:datatype"]}')
        print(f'  core:sample_rate = {meta["global"]["core:sample_rate"]:.0f} Hz')
        print(f'  core:frequency   = {meta["captures"][0]["core:frequency"]:.0f} Hz')


if __name__ == '__main__':
    main()
