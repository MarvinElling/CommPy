[![PyPI version](https://img.shields.io/pypi/v/commpy.svg)](https://pypi.org/project/commpy/)
[![Python versions](https://img.shields.io/pypi/pyversions/commpy.svg)](https://pypi.org/project/commpy/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/MarvinElling/CommPy/actions/workflows/ci.yml/badge.svg)](https://github.com/MarvinElling/CommPy/actions/workflows/ci.yml)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/commpy?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/commpy)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MarvinElling/CommPy/blob/main/examples/quickstart.ipynb)

# CommPy

**CommPy** is a general-purpose Python library for communications engineering (Nachrichtentechnik): channel coding, digital modulation, OFDM, information theory, queuing theory, and link-level simulation, built on NumPy/SciPy with an optional Numba-accelerated fast path.

## Overview

CommPy covers the classic communications-engineering stack, end to end:

- **Channel coding (FEC)**: CRC (8/16/32), Hamming, generic cyclic codes, BCH, Reed-Solomon (error + erasure decoding), convolutional codes with hard/soft-decision Viterbi decoding, **LDPC codes** (belief-propagation decoding; Gallager and quasi-cyclic constructions), **polar codes** (successive-cancellation and CRC-aided list decoding), **turbo codes** (parallel-concatenated RSC with iterative log-MAP/BCJR decoding), block and convolutional interleaving.
- **Digital modulation**: a generic, Gray-coded M-PSK/M-QAM/M-PAM engine with soft-decision (LLR) demodulation, plus the original per-scheme classes (OOK, BPSK, ASK-2/4, QPSK, 8-PSK) kept for backward compatibility.
- **Physical layer**: raised-cosine/root-raised-cosine pulse shaping, ZF/MMSE linear equalization, symbol-timing and carrier-frequency/-phase synchronization (Gardner TED, M-th-power CFO estimation, a Costas loop).
- **OFDM**: modulator/demodulator with configurable active subcarriers and cyclic prefix, PAPR/PAPR-CCDF analysis.
- **Channel models**: BSC, BEC, AWGN, Rayleigh/Rician fading, Z-channel, Gilbert-Elliott bursty channel, uniform quantization.
- **Information theory**: Shannon/binary entropy, mutual information, channel capacity (closed-form BSC/AWGN and Blahut-Arimoto for general DMCs), Huffman and arithmetic source coding, binary rate-distortion.
- **Queuing theory**: M/M/1, M/M/1/K, M/M/c closed-form performance models.
- **Finite-field arithmetic**: prime fields GF(p) and binary extension fields GF(2^m), the algebraic foundation for BCH/Reed-Solomon.
- **Waveform synthesis**: pulse-shaped, optionally up-converted IQ waveforms with eye-diagram/spectrum plotting.
- **SDR interoperability**: read/write raw complex IQ recordings (GNU Radio-compatible) and SigMF (`.sigmf-data`/`.sigmf-meta`) recordings.
- **Link-level simulation**: early-stopping Monte-Carlo BER/FER sweeps (uncoded, plus `simulate_coded_ber` for any soft-input code) with Wilson-score confidence intervals and waterfall-curve plotting.

## Features

- **Modular by design** — each topic (coding, modulation, OFDM, info theory, queuing) is an independent subpackage; shared abstractions (`Modulator`, `FiniteField`) mean adding a new scheme reuses existing, tested machinery instead of duplicating it.
- **Resource-efficient** — vectorized NumPy throughout; SciPy where it's a genuine win (FFT for OFDM, `solve_toeplitz` for MMSE equalization); the one inherently sequential hot loop (Viterbi decoding) gets optional Numba JIT acceleration via `pip install commpy[fast]`, with a correctness-preserving pure-Python fallback when it's not installed.
- **Rigorously tested** — 390+ tests, including exhaustive brute-force cross-validation for algebraic decoders (BCH, Reed-Solomon), Viterbi and polar list decoding against maximum-likelihood search, and statistical BER-vs-SNR checks (coded and uncoded) against theoretical curves.
- **Fully typed** — complete type hints throughout, checked with `mypy --strict`.

## Installation

```bash
pip install commpy
```

With optional JIT acceleration for Viterbi decoding:

```bash
pip install commpy[fast]
```

Or install from source:

```bash
git clone https://github.com/MarvinElling/CommPy.git
cd CommPy
pip install -e ".[dev]"
```

## Quick Start

### Channel coding: Reed-Solomon

```python
from commpy import ReedSolomonCode
import numpy as np

code = ReedSolomonCode(m=8, k=223)  # RS(255, 223), the classic CCSDS code
message = np.arange(223) % code.field.order
codeword = code.encode(message)

corrupted = codeword.copy()
corrupted[[10, 50, 100]] ^= 1  # 3 symbol errors, well within t=16
decoded, _, n_errors = code.decode(corrupted)
assert np.array_equal(decoded, message)
```

### Digital modulation with soft-decision demodulation

```python
from commpy import MQAMModulator, Channels
import numpy as np

mod = MQAMModulator(16)  # 16-QAM, Gray-coded, unit average energy
bits = np.random.randint(0, 2, mod.bits_per_symbol * 1000)
symbols = mod.modulate(bits)

received = Channels.awgn(symbols, snr_db=15)
llrs = mod.soft_demodulate(received, noise_var=1.0)  # feed straight into a Viterbi decoder
hard_bits = mod.demodulate(received)
```

### Convolutional coding + Viterbi decoding

```python
from commpy import Trellis, ConvolutionalEncoder, viterbi_decode
import numpy as np

trellis = Trellis(constraint_length=7, generators=(0o171, 0o133))  # the classic Voyager code
encoder = ConvolutionalEncoder(trellis)

message = np.random.randint(0, 2, 100)
codeword, _ = encoder.encode(message, terminate=True)
decoded = viterbi_decode(trellis, codeword, mode='hard', terminated=True)
assert np.array_equal(decoded, message)
```

### OFDM

```python
from commpy import OFDMModulator, OFDMDemodulator, MQAMModulator, papr_db
import numpy as np

mod, demod = OFDMModulator(n_fft=64, cp_len=16), OFDMDemodulator(n_fft=64, cp_len=16)
qam = MQAMModulator(4)

bits = np.random.randint(0, 2, 64 * qam.bits_per_symbol * 10)
symbols = qam.modulate(bits)
tx = mod.modulate(symbols)
print(f"PAPR: {papr_db(tx[:64]):.1f} dB")

rx_symbols = demod.demodulate(tx)
assert np.allclose(rx_symbols, symbols)
```

### Information theory

```python
from commpy import channel_capacity_awgn, huffman_codes, huffman_encode

capacity = channel_capacity_awgn(snr_linear=10)  # bits/channel use

codes = huffman_codes({'a': 0.5, 'b': 0.25, 'c': 0.25})
encoded = huffman_encode(['a', 'a', 'b', 'c'], codes)
```

### Monte-Carlo BER simulation & SigMF file I/O

```python
from commpy import Channels, MQAMModulator, plot_waterfall, simulate_ber, write_sigmf

mod = MQAMModulator(16)
result = simulate_ber(mod, Channels.awgn, snr_db_range=[6, 8, 10, 12], target_errors=200)
print(result.error_rate, result.ci_lower, result.ci_upper)  # early-stopped, with 95% CIs
plot_waterfall(result)

write_sigmf('capture', mod.modulate([0, 1] * 1000), sample_rate=1e6, center_freq=915e6)
```

More end-to-end examples, including a full transmit chain composing several of these pieces, live in [`examples/`](examples/) — or try [`examples/quickstart.ipynb`](examples/quickstart.ipynb) straight in your browser via the Colab badge above.

## Module Structure

```
commpy/
├── _channelCoding/
│   ├── block/            # CRC, Hamming, cyclic, BCH, Reed-Solomon
│   ├── convolutional/    # Trellis, encoder, Viterbi (hard/soft)
│   └── interleaving/     # Block and convolutional interleavers
├── _channels/            # Channel impairment models (BSC, BEC, AWGN, fading, ...)
├── _fields/               # GF(p) and GF(2^m) arithmetic, polynomials
├── _informationTheory/   # Entropy, capacity, source coding, rate-distortion
├── _modulation/          # Generic M-PSK/M-QAM/M-PAM engine, legacy classes,
│                          # pulse shaping, equalization, synchronization
├── _networking/          # M/M/1-family queuing models
├── _ofdm/                 # OFDM modulator/demodulator, PAPR analysis
├── _sdr/                  # Raw IQ and SigMF file I/O
├── _simulation/           # Monte-Carlo BER/FER simulation, waterfall plotting
├── _utils/                # Math helpers, optional-Numba shim
├── _waves/                # IQ waveform synthesis and plotting
└── __init__.py            # Public API (flat re-export; everything else is private)
```

Only names exported from `commpy/__init__.py` are public API; submodules (anything starting with `_`) may be reorganized without notice.

## Documentation

- [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) — tutorials, one per major feature area.
- [`docs/API.md`](docs/API.md) — full API reference.
- [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) — theory background and best practices.
- [`docs/QUICK_REFERENCE.md`](docs/QUICK_REFERENCE.md) — cheat sheet.
- [`docs/FAQ.md`](docs/FAQ.md), [`docs/CHANGELOG.md`](docs/CHANGELOG.md).
- [`examples/`](examples/) — runnable scripts, one per major feature plus a full-chain capstone.

## Requirements

- Python ≥ 3.10
- NumPy, SciPy, Matplotlib
- Optional: Numba (`pip install commpy[fast]`), for JIT-accelerated Viterbi decoding

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) file for details.

## Author

Marvin Elling

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests. See [CONTRIBUTING.md](CONTRIBUTING.md).
