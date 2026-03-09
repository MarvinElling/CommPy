# CommPy API Reference

Complete API documentation for the CommPy communication engineering library.

## Table of Contents

1. [Modulation](#modulation)
2. [Channels](#channels)
3. [Information Theory](#information-theory)
4. [Waveforms](#waveforms)
5. [Utilities](#utilities)

---

## Modulation

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

Finite field arithmetic over a prime modulus.

**Signature:**
```python
PrimeField(p: int)
```

**Parameters:**
- `p` - Prime modulus

**Methods:**

- `__init__(p: int)`
  - Initialize prime field GF(p)

- `add(a: int, b: int) -> int`
  - Returns `(a + b) mod p`

- `subtract(a: int, b: int) -> int`
  - Returns `(a - b) mod p`

- `multiply(a: int, b: int) -> int`
  - Returns `(a * b) mod p`

- `inverse(a: int) -> int`
  - Returns modular inverse of `a` in GF(p)

- `power(a: int, k: int) -> int`
  - Returns `a^k mod p`

**Example:**
```python
from commpy import PrimeField

gf7 = PrimeField(7)

result = gf7.add(5, 3)
print(result)  # 1, since (5+3) mod 7 = 1

inv = gf7.inverse(3)
print(inv)  # 5, since 3*5 ≡ 1 (mod 7)
```

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

