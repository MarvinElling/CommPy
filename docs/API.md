# CommPy API Reference

Complete API documentation for the CommPy communication engineering library.

## Table of Contents

1. [Modulation (legacy per-scheme classes)](#modulation)
2. [Generic Modulation Engine](#generic-modulation-engine)
3. [Pulse Shaping](#pulse-shaping)
4. [Equalization](#equalization)
5. [Synchronization](#synchronization)
6. [Channels](#channels)
7. [Finite Fields](#finite-fields)
8. [Channel Coding: Block Codes](#channel-coding-block-codes)
9. [Channel Coding: Convolutional Codes](#channel-coding-convolutional-codes)
10. [Channel Coding: Interleaving](#channel-coding-interleaving)
11. [OFDM](#ofdm)
12. [Information Theory](#information-theory)
13. [Source Coding](#source-coding)
14. [Queuing Theory](#queuing-theory)
15. [Waveforms](#waveforms)
16. [SDR Interoperability](#sdr-interoperability)
17. [Link-Level Simulation](#link-level-simulation)
18. [Utilities](#utilities)

---

## Modulation

> These classes predate the [generic modulation engine](#generic-modulation-engine) below and are
> kept only for backward compatibility with their original call signatures. New code should
> prefer `MPSKModulator`/`MQAMModulator`/`MPAMModulator`.

All modulation classes provide static `modulate()` and `demodulate()` methods.

### BPSK_Modulator

**Binary Phase Shift Keying**

Maps binary data to phase-shifted symbols:
- 0 → -1 (magnitude 1, phase π)
- 1 → +1 (magnitude 1, phase 0)

```python
from commpy import BPSK_Modulator

# Modulation
bitstream = [0, 1, 0, 1, 1, 0]
symbols = BPSK_Modulator.modulate(bitstream)
# Output: [-1+0j, 1+0j, -1+0j, 1+0j, 1+0j, -1+0j]

# Demodulation
recovered = BPSK_Modulator.demodulate(symbols)
# Output: [0, 1, 0, 1, 1, 0]
```

**Methods:**

- `modulate(bitstream: array-like) -> ndarray`
  - Modulates binary data using BPSK
  - **Parameters:** `bitstream` - Array of 0s and 1s
  - **Returns:** Complex128 array of BPSK symbols

- `demodulate(signal: array-like, threshold: float = 0) -> ndarray`
  - Demodulates BPSK signal
  - **Parameters:** 
    - `signal` - Complex array of received symbols
    - `threshold` - Decision threshold for real part
  - **Returns:** Array of recovered bits (0 or 1)

---

### QPSK_Modulator

**Quadrature Phase Shift Keying**

Encodes 2 bits per symbol using 4-point constellation.

```python
from commpy import QPSK_Modulator

# Input: 2-bit values (0-3)
bits = [0, 1, 2, 3]
symbols = QPSK_Modulator.modulate(bits)

recovered = QPSK_Modulator.demodulate(symbols)
```

**Methods:**

- `modulate(bits: array-like) -> ndarray`
  - **Parameters:** `bits` - 2-bit values (0, 1, 2, or 3)
  - **Returns:** Complex128 array of QPSK symbols

- `demodulate(signal: array-like) -> ndarray`
  - **Parameters:** `signal` - Complex array of received symbols
  - **Returns:** Array with values 0-3

---

### OOK_Modulator

**On-Off Keying (Amplitude Shift Keying - 2 levels)**

Simple binary modulation: 0 → 0, 1 → amplitude

```python
from commpy import OOK_Modulator

data = [0, 1, 0, 1]
symbols = OOK_Modulator.modulate(data, amplitude=1.0)

recovered = OOK_Modulator.demodulate(symbols, threshold=0.5)
```

**Methods:**

- `modulate(data: array-like, amplitude: float = 1) -> ndarray`
  - **Parameters:**
    - `data` - Binary data (0s and 1s)
    - `amplitude` - Signal amplitude for bit 1
  - **Returns:** Complex128 array

- `demodulate(signal: array-like, threshold: float = 0.5) -> list`
  - **Parameters:**
    - `signal` - Received signal
    - `threshold` - Decision threshold
  - **Returns:** List of demodulated bits

---

### ASK_2_Modulator

**Amplitude Shift Keying - 2 Levels**

Similar to BPSK but uses amplitude variation instead of phase.

```python
from commpy import ASK_2_Modulator

bitstream = [0, 1, 0, 1]
symbols = ASK_2_Modulator.modulate(bitstream, amplitudes=[-1, 1])

recovered = ASK_2_Modulator.demodulate(symbols, threshold=0)
```

**Methods:**

- `modulate(bitstream: array-like, amplitudes: list = [-1, 1]) -> ndarray`
  - **Parameters:**
    - `bitstream` - Binary data
    - `amplitudes` - [amplitude_for_0, amplitude_for_1]
  - **Returns:** Complex128 array

- `demodulate(signal: array-like, threshold: float = 0) -> ndarray`
  - **Parameters:**
    - `signal` - Received signal
    - `threshold` - Decision threshold
  - **Returns:** Binary array

---

### ASK_4_Modulator

**Amplitude Shift Keying - 4 Levels**

Encodes 2 bits per symbol using amplitude levels.

```python
from commpy import ASK_4_Modulator

values = [0, 1, 2, 3]  # 2-bit values
symbols = ASK_4_Modulator.modulate(values)

recovered = ASK_4_Modulator.demodulate(symbols)
```

**Methods:**

- `modulate(values: array-like, levels: list = None) -> ndarray`
  - **Parameters:**
    - `values` - Input symbols (0-3)
    - `levels` - Amplitude levels for each symbol
  - **Returns:** Complex128 array

- `demodulate(signal: array-like) -> ndarray`
  - **Returns:** Array with values 0-3

---

### PSK_8_Modulator

**Phase Shift Keying - 8 Symbols**

Encodes 3 bits per symbol using 8-point constellation.

```python
from commpy import PSK_8_Modulator

values = [0, 1, 2, 3, 4, 5, 6, 7]  # 3-bit values
symbols = PSK_8_Modulator.modulate(values)

recovered = PSK_8_Modulator.demodulate(symbols)
```

**Methods:**

- `modulate(values: array-like) -> ndarray`
  - **Parameters:** `values` - Input symbols (0-7)
  - **Returns:** Complex128 array of 8-PSK symbols

- `demodulate(signal: array-like) -> ndarray`
  - **Returns:** Array with values 0-7

---

## Generic Modulation Engine

A `Modulator` abstract base class with `MPSKModulator`, `MQAMModulator`, and `MPAMModulator`
subclasses. All three build a unit-average-energy, Gray-coded constellation once at construction;
`modulate`/`demodulate`/`soft_demodulate` are then generic and fully vectorized — new schemes get
correct nearest-neighbor demodulation and LLR computation for free.

```python
from commpy import MQAMModulator, Channels

mod = MQAMModulator(16)          # 16-QAM, Gray-coded, unit average energy
print(mod.bits_per_symbol)       # 4
print(mod.constellation)         # 16 complex points, mean(|s|^2) == 1

bits = [0, 1, 1, 0] * 100
symbols = mod.modulate(bits)
received = Channels.awgn(symbols, snr_db=15)

hard_bits = mod.demodulate(received)
llrs = mod.soft_demodulate(received, noise_var=1.0)  # positive LLR favors bit 0
```

**Constructors:**
- `MPSKModulator(M)` — M-ary PSK, any `M` a power of two `>= 2`.
- `MQAMModulator(M)` — square M-QAM, `M` a power of four (4, 16, 64, 256, ...). Non-square/cross
  constellations (e.g. 32-QAM) are out of scope.
- `MPAMModulator(M)` — M-ary PAM, any `M` a power of two `>= 2`.

**Attributes** (set at construction): `M`, `bits_per_symbol`, `constellation` (shape `(M,)`),
`bit_labels` (shape `(M, bits_per_symbol)`).

**Methods:**
- `modulate(bits) -> ndarray[complex128]` — maps a bitstream (list, tuple, or ndarray; length a
  multiple of `bits_per_symbol`) onto constellation symbols.
- `demodulate(symbols) -> ndarray[int64]` — hard-decision nearest-neighbor demodulation.
- `soft_demodulate(symbols, noise_var) -> ndarray[float64]` — max-log-approximate LLRs, one per
  bit, in the same ordering as `demodulate`'s output. Feed directly into `viterbi_decode(mode='soft')`.

---

## Pulse Shaping

`raised_cosine_filter` / `root_raised_cosine_filter` build callables compatible with
`IQWaveform`'s `pulse_shape` parameter (causal, peaking at `span*T/2`).

```python
from commpy import raised_cosine_filter, IQWaveform
import numpy as np

pulse = raised_cosine_filter(symbol_period=1e-3, rolloff=0.35, span=8)
wf = IQWaveform(I=np.array([1., -1., 1.]), Q=np.zeros(3), T=1e-3, fs=20_000, pulse_shape=pulse, span=8)
```

**Signatures:**
```python
raised_cosine_filter(symbol_period: float, rolloff: float, span: int) -> Callable
root_raised_cosine_filter(symbol_period: float, rolloff: float, span: int) -> Callable
```

- `rolloff` — excess bandwidth factor `beta` in `[0, 1]`.
- `span` — truncation window in multiples of `symbol_period` (pass the same value to `IQWaveform`).
- The raised-cosine pulse satisfies the Nyquist zero-ISI criterion (`g(k*T) == 0` for nonzero
  integer `k`). A matched RRC/RRC pair's combined response equals the RC pulse.

---

## Equalization

Linear FIR channel equalizers that invert a known channel impulse response.

```python
from commpy import zf_equalizer, mmse_equalizer

channel = [0.2, 1.0, 0.3, -0.1]      # a short multipath channel
w_zf = zf_equalizer(channel, n_taps=15)
w_mmse = mmse_equalizer(channel, n_taps=15, noise_var=0.1)
```

**Signatures:**
```python
zf_equalizer(channel_taps, n_taps: int, delay: int | None = None) -> ndarray[float64]
mmse_equalizer(channel_taps, n_taps: int, noise_var: float, delay: int | None = None) -> ndarray[float64]
```

- `zf_equalizer` minimizes residual ISI alone (least squares); `mmse_equalizer` adds a
  noise-variance regularization term (`noise_var=0` reduces exactly to `zf_equalizer`), trading a
  little residual ISI for much better noise suppression under real (noisy) conditions.
- `mmse_equalizer` solves the Wiener-Hopf normal equations via `scipy.linalg.solve_toeplitz`
  (the channel's autocorrelation Gram matrix is Toeplitz), an O(n²) solve instead of O(n³).

---

## Synchronization

Narrow-scope, single-algorithm synchronization primitives (not a configurable PLL framework).

```python
from commpy import gardner_timing_error, estimate_cfo_mth_power, costas_loop_bpsk

# Symbol timing error (2 samples/symbol input): zero at the correct sampling instant.
error = gardner_timing_error(samples_at_2sps)

# Carrier frequency offset via the M-th power method (removes M-PSK modulation).
cfo_hz = estimate_cfo_mth_power(signal, fs=1000.0, m_order=4)  # QPSK -> m_order=4

# Minimal first-order Costas loop for BPSK carrier phase recovery.
corrected, phase_history = costas_loop_bpsk(signal, loop_gain=0.05)
```

- `gardner_timing_error(samples)` — the classic Gardner (1986) non-data-aided timing-error
  detector; a discriminator, not a closed loop (feed its output into your own loop filter).
- `estimate_cfo_mth_power(signal, fs, m_order)` — raises the signal to the `m_order`-th power to
  strip M-PSK modulation, leaving a pure tone at `m_order * cfo`, located via FFT peak search.
- `costas_loop_bpsk(signal, loop_gain=0.05)` — fixed proportional-gain feedback loop; recovers
  carrier phase up to the usual BPSK 180-degree ambiguity.

---

## Channels

### Channels (Static Class)

Channel models for digital communication systems.

#### bsc() - Binary Symmetric Channel

Flips each bit independently with probability `p`.

```python
from commpy import Channels
import numpy as np

bits = np.array([1, 0, 1, 1, 0])
corrupted = Channels.bsc(bits, p=0.1)  # 10% bit error rate
```

**Signature:**
```python
Channels.bsc(
    bits: array-like,
    p: float,
    rng: np.random.Generator | None = None
) -> ndarray
```

**Parameters:**
- `bits` - Array-like of 0/1 or boolean values
- `p` - Bit flip probability in [0, 1]
- `rng` - Optional `np.random.Generator` for reproducibility. If `None`, uses default RNG.

**Returns:** 
- ndarray of same shape/dtype as input with (possibly) flipped bits

**Example:**
```python
rng = np.random.default_rng(seed=42)
bits = [1, 1, 0, 0, 1]
output = Channels.bsc(bits, p=0.2, rng=rng)
```

---

#### bec() - Binary Erasure Channel

Erases (replaces) each symbol independently with probability `p`.

```python
from commpy import Channels

bits = np.array([1, 0, 1, 1, 0])
erased = Channels.bec(bits, p=0.1, erasure_value=-1)
```

**Signature:**
```python
Channels.bec(
    bits: array-like,
    p: float,
    erasure_value: any = -1,
    rng: np.random.Generator | None = None
) -> ndarray
```

**Parameters:**
- `bits` - Array-like of symbols (commonly 0/1)
- `p` - Erasure probability in [0, 1]
- `erasure_value` - Symbol used to mark erased positions (default: -1)
- `rng` - Optional `np.random.Generator`

**Returns:**
- Float array with erased entries set to `erasure_value`

**Example:**
```python
# Simulate 20% erasures with None marker
output = Channels.bec([1, 0, 1], p=0.2, erasure_value=None)
```

---

#### awgn() - Additive White Gaussian Noise

Adds Gaussian noise to achieve target SNR in dB.

```python
from commpy import Channels
import numpy as np

signal = np.array([1+1j, -1-1j, 1-1j])
noisy = Channels.awgn(signal, snr_db=10)
```

**Signature:**
```python
Channels.awgn(
    x: array-like,
    snr_db: float,
    rng: np.random.Generator | None = None
) -> ndarray
```

**Parameters:**
- `x` - Input signal (array-like, real or complex)
- `snr_db` - Target SNR in dB: `SNR_linear = 10^(snr_db/10)`
- `rng` - Optional `np.random.Generator`

**Noise Generation:**
- Signal power: `P_s = mean(|x|²)`
- Noise power: `P_n = P_s / SNR_linear`
- For real signals: Real Gaussian noise with variance `P_n`
- For complex signals: Complex Gaussian with noise power `P_n` split equally between I/Q

**Returns:**
- ndarray with same shape/dtype as input, containing noisy signal

**Example:**
```python
# Modulate, transmit through AWGN at varying SNR
bits = np.random.randint(0, 2, 1000)
symbols = BPSK_Modulator.modulate(bits)

for snr in [0, 5, 10, 15]:
    noisy = Channels.awgn(symbols, snr_db=snr)
    recovered = BPSK_Modulator.demodulate(noisy)
    ber = np.mean(recovered != bits)
    print(f"SNR={snr} dB: BER={ber}")
```

---

#### Other channel models

```python
from commpy import Channels
import numpy as np

x = np.array([1+0j, -1+0j, 1+0j])
Channels.rayleigh(x, snr_db=15)               # Rayleigh fading + AWGN
Channels.rician(x, snr_db=15, k_factor=10.0)  # Rician fading + AWGN (LOS + scatter)
Channels.z_channel([1, 0, 1, 1], p=0.1)       # asymmetric: only 1 -> 0 flips
Channels.gilbert_elliott(                     # 2-state bursty (Markov) channel
    np.zeros(1000, dtype=int), p_gb=0.05, p_bg=0.2, p_good=0.0, p_bad=0.3,
)
Channels.quantize(np.linspace(-1, 1, 100), bits=8)  # uniform scalar quantization
```

- `rayleigh(x, snr_db, rng=None)` / `rician(x, snr_db, k_factor=10.0, rng=None)` — multiplicative
  fading (complex Gaussian / LOS-plus-scatter) followed by `awgn`.
- `z_channel(bits, p, rng=None)` — asymmetric binary channel: `1` flips to `0` with probability
  `p`; `0` never flips.
- `gilbert_elliott(bits, p_gb, p_bg, p_good=0.0, p_bad=0.2, rng=None, init_state='good')` — a
  2-state Markov chain alternating between low- and high-error-rate states, modeling bursty
  errors (unlike BSC's memoryless errors).
- `quantize(x, bits=8, vmin=None, vmax=None)` — uniform scalar quantization of an analog signal.

---

## Finite Fields

`PrimeField` (GF(p)) and `GF2m` (GF(2^m)) share a common `FiniteField` interface
(`add`/`subtract`/`multiply`/`divide`/`power`/`negate`). `GF2m` is the algebraic foundation for
BCH and Reed-Solomon codes below.

```python
from commpy import PrimeField, GF2m

gf7 = PrimeField(7)
gf7.multiply(3, 5)         # 1  ((3*5) mod 7)
gf7.divide(6, 3)           # 2

gf256 = GF2m(m=8)          # GF(2^8), used by RS(255, k) codes
gf256.multiply(0x53, 0xCA) # O(1) via log/antilog tables, vectorizes over arrays
gf256.power(2, 100)        # alpha^100 for the field's primitive element (== 2)
```

**`PrimeField(p)`** — raises `ValueError` if `p` isn't prime. `primitive_roots()` returns all
generators of the multiplicative group.

**`GF2m(m, modulus_poly=None)`** — `2**m`-element field; if `modulus_poly` is omitted, the
numerically smallest primitive polynomial of degree `m` is found automatically (any primitive
polynomial defines an isomorphic copy of the field, so this doesn't need to match another
library's conventional choice). `exp(k)` returns `alpha**k` for an array of (possibly negative)
exponents `k`.

---

## Channel Coding: Block Codes

### CRC

Parametrized ("Rocksoft model") CRC, verified against `zlib.crc32`/`binascii.crc_hqx`.

```python
from commpy import CRC

CRC.crc32().compute(b'123456789')       # 0xCBF43926
CRC.crc16_xmodem().compute(b'hello')
CRC.crc8().compute(b'hello')
```

Custom polynomials via `CRCConfig(width, poly, init, xorout, refin, refout)` passed to `CRC(config)`.

### HammingCode

```python
from commpy import HammingCode
import numpy as np

code = HammingCode(m=3)                          # Hamming(7, 4)
codeword = code.encode(np.array([1, 0, 1, 1]))
corrupted = codeword.copy(); corrupted[2] ^= 1
message, corrected, error_pos = code.decode(corrupted)  # corrects any single-bit error
```

### CyclicCode

Generic cyclic code from a generator polynomial (over any `FiniteField`); the base BCH and
Reed-Solomon build on.

```python
from commpy import CyclicCode, PrimeField

code = CyclicCode(n=7, generator=[1, 1, 0, 1], field=PrimeField(2))  # (7,4) Hamming as a cyclic code
codeword = code.encode([1, 0, 1, 1])
code.is_codeword(codeword)          # True
code.syndrome(codeword)             # zero polynomial
```

### BCHCode

Binary BCH code correcting up to `t` errors, via Berlekamp-Massey + Chien search.

```python
from commpy import BCHCode

code = BCHCode(m=4, t=2)   # BCH(15, 7, t=2) over GF(16)
codeword = code.encode(message)
message, corrected, n_errors = code.decode(received)
```

### ReedSolomonCode

RS code over GF(2^m); error decoding (unknown positions, up to `t = (n-k)//2`) and a separate
erasures-only decoder (known positions, up to `n-k` — twice the error-only capability).

```python
from commpy import ReedSolomonCode

code = ReedSolomonCode(m=8, k=223)  # RS(255, 223), the classic CCSDS/DVB code
codeword = code.encode(message)
message, corrected, n_errors = code.decode(received)
message, corrected = code.decode_erasures(received, erasure_positions)
```

---

## Channel Coding: Convolutional Codes

`Trellis` + `ConvolutionalEncoder` + `viterbi_decode` (hard- and soft-decision).

```python
from commpy import Trellis, ConvolutionalEncoder, viterbi_decode

trellis = Trellis(constraint_length=7, generators=(0o171, 0o133))  # the Voyager/NASA code
encoder = ConvolutionalEncoder(trellis)

codeword, final_state = encoder.encode(message, terminate=True)  # zero-tail terminated
decoded = viterbi_decode(trellis, codeword, mode='hard', terminated=True)

# Soft-decision: feed LLRs from Modulator.soft_demodulate directly.
decoded = viterbi_decode(trellis, llrs, mode='soft', terminated=True)
```

- `Trellis(constraint_length, generators)` — `generators` is one tap-pattern integer per output
  bit (MSB = current input bit), giving a rate-`1/len(generators)` code.
- Viterbi's add-compare-select trellis traversal is JIT-accelerated via the optional `numba`
  extra (`pip install commpy[fast]`); without it, a pure-Python fallback gives identical results.

---

## Channel Coding: Interleaving

```python
from commpy import BlockInterleaver, ConvolutionalInterleaver

# Block: write rows, read columns. Frame-based (fixed block size).
il = BlockInterleaver(rows=8, cols=8)
interleaved = il.interleave(data)       # length rows*cols
original = il.deinterleave(interleaved)

# Convolutional (Ramsey type-II): streaming, round-robin shift-register lanes.
tx_il = ConvolutionalInterleaver(n_lanes=4, delay_increment=3)
rx_il = ConvolutionalInterleaver(n_lanes=4, delay_increment=3, is_deinterleaver=True)
interleaved = tx_il.process(data)
recovered = rx_il.process(interleaved)  # matches data after tx_il.total_delay samples
```

The convolutional interleaver is stateful and streaming: it introduces a fixed end-to-end
latency of `n_lanes * (n_lanes - 1) * delay_increment` samples (see `total_delay`), after which
`recovered[i] == data[i - total_delay]`.

---

## OFDM

```python
from commpy import OFDMModulator, OFDMDemodulator, MQAMModulator, papr_db

mod = OFDMModulator(n_fft=64, cp_len=16, active_subcarriers=range(4, 60))  # null DC/edges
demod = OFDMDemodulator(n_fft=64, cp_len=16, active_subcarriers=range(4, 60))

qam = MQAMModulator(16)
symbols = qam.modulate(bits)
tx = mod.modulate(symbols)          # (n_fft + cp_len) samples per OFDM symbol
print(papr_db(tx[:64]))             # PAPR of the first OFDM symbol, in dB
rx_symbols = demod.demodulate(tx)   # exact inverse (noiseless round trip)
```

**Signatures:**
```python
OFDMModulator(n_fft: int, cp_len: int, active_subcarriers: ArrayLike | None = None)
OFDMDemodulator(n_fft: int, cp_len: int, active_subcarriers: ArrayLike | None = None)
papr(signal) -> float        # linear peak-to-average power ratio
papr_db(signal) -> float
papr_ccdf(ofdm_symbols, thresholds_db) -> ndarray  # P(PAPR > threshold) per threshold
```

`active_subcarriers` defaults to all `n_fft` subcarriers; pass a subset to null DC/guard bands.
The theoretical worst case (all subcarriers equal) produces a single time-domain impulse with
`papr == n_fft`.

---

## Information Theory

### shannon_entropy()

Calculate Shannon entropy of a probability distribution.

**Signature:**
```python
shannon_entropy(probabilities: list[float]) -> float
```

**Parameters:**
- `probabilities` - List of probability values. Values ≤ 0 are skipped.

**Returns:**
- Shannon entropy in bits: $H = -\sum_{i} p_i \log_2(p_i)$

**Example:**
```python
from commpy import shannon_entropy

# Uniform distribution
H_uniform = shannon_entropy([0.25, 0.25, 0.25, 0.25])
print(f"Uniform: {H_uniform} bits")  # Output: 2.0

# Skewed distribution
H_skewed = shannon_entropy([0.5, 0.25, 0.125, 0.125])
print(f"Skewed: {H_skewed:.3f} bits")  # Output: ~1.75

# Zero probabilities are ignored
H_two = shannon_entropy([1.0, 0.0, 0.0])
print(f"Deterministic: {H_two} bits")  # Output: 0.0
```

---

### Mutual information & channel capacity

```python
from commpy import (
    binary_entropy, mutual_information, channel_capacity_bsc,
    channel_capacity_awgn, channel_capacity_dmc,
)
import numpy as np

binary_entropy(0.1)                    # H_b(p), the entropy of a Bernoulli(p) source
mutual_information(joint_prob_matrix)  # I(X;Y) from a joint distribution P[x,y]
channel_capacity_bsc(p=0.1)            # closed form: 1 - H_b(p)
channel_capacity_awgn(snr_linear=10)   # closed form (Shannon-Hartley): log2(1 + snr)

# General discrete memoryless channel, via the Blahut-Arimoto algorithm.
Q = np.array([[1.0, 0.0], [0.3, 0.7]])  # Z-channel transition matrix Q[x,y] = p(y|x)
capacity, optimal_input_dist = channel_capacity_dmc(Q)
```

`channel_capacity_dmc` iteratively finds the capacity-achieving input distribution; cross-checked
in the test suite against `channel_capacity_bsc` (symmetric channels) and a brute-force grid
search (asymmetric channels).

---

## Source Coding

Huffman coding and arithmetic coding (`_informationTheory/source_coding.py`).

```python
from commpy import huffman_codes, huffman_encode, huffman_decode

codes = huffman_codes({'a': 0.5, 'b': 0.25, 'c': 0.25})   # symbol -> bitstring
encoded = huffman_encode(['a', 'a', 'b', 'c'], codes)
decoded = huffman_decode(encoded, codes)                  # ['a', 'a', 'b', 'c']
```

```python
from commpy import arithmetic_encode, arithmetic_decode

probs = {'a': 0.5, 'b': 0.3, 'c': 0.2}
symbols = ['a', 'b', 'c', 'a', 'a']
code, n_bits = arithmetic_encode(symbols, probs)
decoded = arithmetic_decode(code, n_bits, len(symbols), probs)  # same probs, same length
```

The arithmetic coder uses exact `fractions.Fraction` interval arithmetic rather than the classic
fixed-precision-register formulation with carry propagation — simpler to verify correct, at the
cost of speed on very long sequences (appropriate for a supporting feature, not a bulk compressor).

### rate_distortion_binary()

```python
from commpy import rate_distortion_binary

rate_distortion_binary(p=0.5, distortion=0.1)  # R(D) = H_b(p) - H_b(D) for D <= min(p, 1-p)
```

Closed-form rate-distortion function for a Bernoulli(`p`) source under Hamming distortion.

---

## Queuing Theory

Closed-form M/M/1-family performance models (`_networking/queuing.py`), no simulation.

```python
from commpy import MM1Queue, MM1KQueue, MMcQueue

q = MM1Queue(arrival_rate=3.0, service_rate=5.0)
q.utilization, q.mean_number_in_system, q.mean_wait_in_queue  # rho, L, Wq

qk = MM1KQueue(arrival_rate=3.0, service_rate=2.0, capacity=10)  # finite capacity: always stable
qk.blocking_probability, qk.effective_arrival_rate

qc = MMcQueue(arrival_rate=8.0, service_rate=3.0, n_servers=4)  # Erlang-C
qc.erlang_c  # probability an arriving customer must wait
```

All three expose `mean_number_in_system`/`mean_number_in_queue`/`mean_wait_in_system`/
`mean_wait_in_queue`, consistent with Little's Law (`L = lambda * W`); `MM1KQueue` uses the
*effective* (admitted) arrival rate since blocked arrivals don't enter the system.

---

## Waveforms

### IQWaveform

Class for generating and manipulating IQ modulated waveforms.

**Signature:**
```python
IQWaveform(
    I: Sequence[float],
    Q: Sequence[float],
    T: float,
    fs: float,
    f0: float = 0.0,
    pulse_shape: Callable[[ndarray], ndarray] | None = None,
    span: int = 4,
) -> None
```

**Parameters:**
- `I`, `Q` - Arrays of I/Q symbol values (baseband symbols)
- `T` - Symbol period in seconds
- `fs` - Sample rate in Hz (samples per second)
- `f0` - Carrier frequency in Hz; use 0 for baseband-only output
- `pulse_shape` - Callable `g_s(tau)` defining the pulse shape. Default: rectangular pulse
- `span` - Pulse window duration in multiples of T (affects computation length)

**Attributes:**
- `.t` - Time vector (ndarray)
- `.s` - Transmitted signal:
  - If `f0 > 0`: Real-valued bandpass signal
  - If `f0 = 0`: Complex-valued baseband signal
- `.s_I` - I component (baseband)
- `.s_Q` - Q component (baseband)
- `.I`, `.Q` - Input symbol arrays
- `.T`, `.fs`, `.f0` - Parameters

**Methods:**

- `plot_waveform()`
  - Visualizes the transmitted signal `s(t)` using Matplotlib
  - Shows time-domain waveform with proper labels

**Example: Baseband**
```python
from commpy import IQWaveform
import numpy as np

# Simple constellation: alternating I and Q
I = np.array([1, 0, -1, 0])
Q = np.array([0, 1, 0, -1])

wf = IQWaveform(I=I, Q=Q, T=1e-3, fs=1e6)

# Baseband signal
print(f"Baseband signal shape: {wf.s.shape}")
print(f"Time vector: {wf.t[:10]}")
```

**Example: Bandpass with Custom Pulse**
```python
from commpy import IQWaveform
import numpy as np

def raised_cosine(tau, T, alpha=0.5):
    """Raised cosine pulse shape."""
    # Simplified: you may want a complete implementation
    return np.sinc(tau/T) 

I = np.array([1, 1, -1, -1])
Q = np.array([0, 0, 0, 0])

wf = IQWaveform(
    I=I, Q=Q,
    T=100e-6,           # 100 µs symbol period
    fs=10e6,            # 10 MHz sampling
    f0=1e6,             # 1 MHz carrier
    pulse_shape=raised_cosine,
    span=4
)

print(f"Bandpass signal (real): {wf.s[:100]}")
wf.plot_waveform()
```

---

## SDR Interoperability

Read/write complex baseband IQ recordings in two on-disk formats: a raw,
headerless binary stream (compatible with GNU Radio's
`blocks.file_sink`/`blocks.file_source`) and [SigMF](https://github.com/sigmf/SigMF)
recordings (a `.sigmf-data` sample file plus a `.sigmf-meta` JSON sidecar).

### write_iq() / read_iq()

**Signature:**
```python
write_iq(path: str | Path, samples: ArrayLike, dtype: DTypeLike = np.complex64) -> None
read_iq(path: str | Path, dtype: DTypeLike = np.complex64) -> ndarray
```

**Parameters:**
- `path` - Output/input file path
- `samples` - Complex baseband samples (write only)
- `dtype` - On-disk sample dtype; `np.complex64` (GNU Radio's default `gr_complex`) or `np.complex128`

**Raises:** `ValueError` if `dtype` is not `complex64`/`complex128` (`write_iq` only).

```python
from commpy import MQAMModulator, read_iq, write_iq

mod = MQAMModulator(16)
symbols = mod.modulate([0, 1] * 500)
write_iq('recording.cf32', symbols)
recovered = read_iq('recording.cf32')
```

### write_sigmf() / read_sigmf()

**Signature:**
```python
write_sigmf(
    path: str | Path,
    samples: ArrayLike,
    sample_rate: float,
    center_freq: float = 0.0,
    *,
    dtype: DTypeLike = np.complex64,
    description: str = '',
    author: str = '',
) -> None
read_sigmf(path: str | Path) -> tuple[ndarray, dict]
```

**Parameters:**
- `path` - Base path (extension-less); writes/reads `<path>.sigmf-data` and `<path>.sigmf-meta`.
  `read_sigmf` also accepts a path ending in `.sigmf-data` or `.sigmf-meta`.
- `sample_rate`, `center_freq` - Recorded in the `.sigmf-meta` sidecar's `global`/`captures` sections
- `description`, `author` - Free-text metadata fields

**Raises:** `ValueError` if `dtype` (write) or the recording's `core:datatype` (read) is not a supported complex format.

```python
from commpy import write_sigmf, read_sigmf

write_sigmf('capture', symbols, sample_rate=1e6, center_freq=915e6, description='16-QAM test capture')
recovered, meta = read_sigmf('capture')
print(meta['global']['core:sample_rate'])
```

---

## Link-Level Simulation

Monte-Carlo error-rate simulation with early stopping and confidence
intervals, plus waterfall-curve plotting.

### simulate_error_rate()

Generic core: drives an arbitrary "run N trials at this SNR, count errors"
function, so it works for bit-, frame-, or symbol-error rates alike.

**Signature:**
```python
simulate_error_rate(
    trial_fn: Callable[[float, Generator, int], tuple[int, int]],
    snr_db_range: ArrayLike,
    *,
    target_errors: int = 100,
    max_trials: int = 10_000_000,
    trials_per_batch: int = 10_000,
    confidence: float = 0.95,
    rng: Generator | None = None,
) -> SimulationResult
```

For each SNR point, `trial_fn` runs in batches until either `target_errors`
errors have been observed or `max_trials` trials have run — early stopping
avoids spending a large, fixed trial budget at SNR points that converge
almost immediately.

### simulate_ber()

Convenience wrapper around `simulate_error_rate` for the common
modulate → channel → demodulate → count-bit-errors case.

**Signature:**
```python
simulate_ber(
    modulator: Modulator,
    channel_fn: Callable[[ndarray, float, Generator], ndarray],
    snr_db_range: ArrayLike,
    *,
    bits_per_batch: int = 10_000,
    target_errors: int = 100,
    max_trials: int = 10_000_000,
    confidence: float = 0.95,
    rng: Generator | None = None,
) -> SimulationResult
```

`channel_fn` is called positionally as `channel_fn(symbols, snr_db, rng)`,
so `Channels.awgn`/`Channels.rayleigh` work directly; channels with extra
parameters (e.g. `Channels.rician`) need a small wrapper.

### SimulationResult

Frozen dataclass with one entry per SNR point: `snr_db`, `error_rate`,
`ci_lower`, `ci_upper` (Wilson score interval), `n_trials`, `n_errors`.

### plot_waterfall()

**Signature:**
```python
plot_waterfall(
    result: SimulationResult,
    theoretical: Callable[[ndarray], ndarray] | None = None,
    ax: Axes | None = None,
) -> Axes
```

Plots the BER/FER-vs-SNR curve (log scale) with confidence-interval error
bars, optionally overlaid with a closed-form `theoretical` reference curve.

```python
from commpy import Channels, MQAMModulator, plot_waterfall, simulate_ber

mod = MQAMModulator(16)
result = simulate_ber(mod, Channels.awgn, snr_db_range=[6, 8, 10, 12, 14], target_errors=200)
ax = plot_waterfall(result)
ax.figure.show()
```

---

## Utilities

### is_prime()

Check if a number is prime.

**Signature:**
```python
is_prime(n: int) -> bool
```

**Parameters:**
- `n` - Integer to test

**Returns:**
- `True` if `n` is prime, `False` otherwise

**Example:**
```python
from commpy import is_prime

print(is_prime(7))   # True
print(is_prime(10))  # False
```

---

### modinv()

Compute modular multiplicative inverse.

Find $a^{-1} \pmod{m}$ such that $a \cdot a^{-1} \equiv 1 \pmod{m}$.

**Signature:**
```python
modinv(a: int, m: int) -> int
```

**Parameters:**
- `a` - Integer to invert
- `m` - Modulus (should be coprime with `a`)

**Returns:**
- Modular inverse of `a` modulo `m`

**Raises:**
- `ValueError` if inverse doesn't exist (gcd(a, m) ≠ 1)

**Example:**
```python
from commpy import modinv

inv = modinv(3, 7)
print(inv)  # 5, since 3*5 = 15 ≡ 1 (mod 7)
print((3 * inv) % 7)  # 1
```

---

### PrimeField

See [Finite Fields](#finite-fields) above — `PrimeField` (GF(p)) now shares a common
`FiniteField` interface with `GF2m` (GF(2^m)). Note the divide-by method is `divide(a, b)`
(`gf7.divide(1, 3)` for the inverse of 3), not `inverse(a)`.

---

## Type Hints

All functions use Python type hints for better IDE support and type checking.

**Common types used:**
- `array-like` - Lists, tuples, or numpy arrays
- `ndarray` - NumPy array (from `numpy`)
- `Callable` - Function type
- `Sequence` - Ordered collection
- `float | int` - Union type (Python 3.10+)
- `... | None` - Optional type

---

## Error Handling

Most functions handle invalid inputs gracefully:
- **Array conversion**: Auto-converts array-like inputs to numpy arrays
- **Type preservation**: Output dtype/shape match input when possible
- **RNG seeding**: Use `np.random.default_rng(seed=N)` for reproducible results

---

## References & Further Reading

- Digital Communications theory
- Information Theory fundamentals
- IQ modulation and waveform generation

