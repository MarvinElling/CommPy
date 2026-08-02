# CommPy User Guide

A comprehensive guide to understanding and using CommPy's features.

## Table of Contents

1. [Introduction](#introduction)
2. [Modulation & Demodulation](#modulation-demodulation)
3. [Channel Models](#channel-models)
4. [Information Theory](#information-theory)
5. [Waveform Generation](#waveform-generation)
6. [SDR Interoperability & Link Simulation](#sdr-interoperability-link-simulation)
7. [Practical Applications](#practical-applications)
8. [Best Practices](#best-practices)

---

## Introduction

### What is CommPy?

CommPy (Communication Python) is a toolkit for digital communications simulation and analysis. It provides:

- **Modulation**: Convert digital data to analog signals
- **Channels**: Model wireless transmission impairments
- **Analysis**: Calculate information-theoretic quantities
- **Waveforms**: Generate realistic RF signals

### Key Use Cases

- **Education**: Learn digital communications theory
- **Research**: Simulate communication systems
- **Development**: Prototype communication algorithms
- **Testing**: Validate communication designs

### Installation Quick Start

```bash
pip install commpy
python -c "from commpy import *; print('Successfully installed!')"
```

---

## Modulation & Demodulation

### What is Modulation?

Modulation maps digital data (bits or symbols) to analog signals suitable for transmission.

**Key Reasons for Modulation:**

1. **Frequency shifting**: Move signal to transmission frequency
2. **Bandwidth efficiency**: Pack more information in available spectrum
3. **Robustness**: Different schemes have different noise tolerance

### Basic Structure

```
Transmitter                Channel              Receiver
───────────              ────────             ──────────
Bits → [Modulate] → Signal → [Noise] → [Demodulate] → Bits
                                            ↓
                                      Symbol → Bit mapping
```

### BPSK (Binary Phase Shift Keying)

**Overview:** Maps each bit to a phase-shifted complex symbol.

**Constellation:**
```
        ↑ Q
        |
  ─1 ← ─┼─ +1 →
        |
        ↓
```

**Mapping:**
- Bit 0 → symbol -1
- Bit 1 → symbol +1

**Code:**
```python
from commpy import BPSK_Modulator
import numpy as np

# Modulation
bits = [0, 1, 0, 1]
symbols = BPSK_Modulator.modulate(bits)
# Result: [-1+0j, 1+0j, -1+0j, 1+0j]

# Demodulation
recovered = BPSK_Modulator.demodulate(symbols)
# Result: [0, 1, 0, 1]
```

**Advantages:**
- Simple implementation
- Robust to noise
- Good for low SNR

**Disadvantages:**
- Only 1 bit per symbol
- Lower spectral efficiency

**When to use:** When noise/interference is severe or simplicity is important.

---

### QPSK (Quadrature Phase Shift Keying)

**Overview:** Uses 4 symbols arranged in quadrature, encoding 2 bits per symbol.

**Constellation:**
```
     Q
     ↑     3 (11)
     |  2 (10) | 0 (00)
──────┼────────┼────── I
     |  1 (01) |
     |
```

**Mapping:**
- 00 → +1+1j (upper right)
- 01 → -1+1j (upper left)
- 10 → -1-1j (lower left)
- 11 → +1-1j (lower right)

**Code:**
```python
from commpy import QPSK_Modulator

# Input: 2-bit values [0, 1, 2, 3] = [00, 01, 10, 11]
symbols = QPSK_Modulator.modulate([0, 1, 2, 3])
# Result: Complex symbols at 4 positions

recovered = QPSK_Modulator.demodulate(symbols)
# Result: [0, 1, 2, 3]
```

**Advantages:**
- 2 bits per symbol (2× BPSK efficiency)
- Same power as BPSK per bit
- Widely used in practice

**Disadvantages:**
- Slightly more receiver complexity
- Requires I/Q demodulation

**When to use:** Most practical wireless systems (WiFi, cellular, satellite).

---

### ASK (Amplitude Shift Keying)

**Overview:** Encodes information in signal amplitude.

**Types:**
- **ASK-2:** 2 amplitude levels (binary)
- **ASK-4:** 4 amplitude levels (2 bits per symbol)

**ASK-2 Constellation:**
```
→ I axis
0 V
████ (Bit 0)
████████ (Bit 1)
```

**Code:**
```python
from commpy import ASK_2_Modulator, ASK_4_Modulator

# ASK-2: 2 levels
symbols_2 = ASK_2_Modulator.modulate([0, 1, 0], amplitudes=[-1, 1])

# ASK-4: 4 levels
symbols_4 = ASK_4_Modulator.modulate([0, 1, 2, 3])
```

**Advantages:**
- Simple hardware (amplitude control)
- Good for optical/intensity-modulated systems

**Disadvantages:**
- Sensitive to amplitude variations
- Poor performance with nonlinear channels

**When to use:** Optical communication, intensity-modulated systems, simple transmitters.

---

### PSK-8 (8-ary Phase Shift Keying)

**Overview:** 8 symbols at 45° intervals, encoding 3 bits per symbol.

**Constellation:**
```
        000
        ↓
   100     010
     \   /
      \ /
  ─────●───── I
      / \
     /   \
   101    001
        ↑
       011
     (and 111)
```

**Code:**
```python
from commpy import PSK_8_Modulator

# Input: 3-bit values
symbols = PSK_8_Modulator.modulate([0, 1, 2, 3, 4, 5, 6, 7])

recovered = PSK_8_Modulator.demodulate(symbols)
```

**Advantages:**
- 3 bits per symbol (high efficiency)
- Constant envelope (efficient power amplifier)

**Disadvantages:**
- More susceptible to noise than QPSK
- Requires more precise synchronization

**When to use:** High-speed links with good SNR, satellite communication.

---

### OOK (On-Off Keying)

**Overview:** Simply turns signal on/off for bits 1/0.

**Constellation:**
```
        ↑
    1 | ●
      |
  ────┼──── I
      |
    0 | ●
        No transmission
```

**Code:**
```python
from commpy import OOK_Modulator

# Simple binary modulation
symbols = OOK_Modulator.modulate([0, 1, 0, 1], amplitude=1.0)

recovered = OOK_Modulator.demodulate(symbols, threshold=0.5)
```

**Advantages:**
- Extremely simple
- Works with simple LEDs, lasers (optical)

**Disadvantages:**
- Very inefficient (1 bit per symbol)
- Poor noise performance

**When to use:** Optical communication, simple intensity modulation.

---

## Channel Models

### What are Channels?

Channel models simulate how signals degrade during transmission over real media.

### BSC (Binary Symmetric Channel)

**What it does:** Randomly flips bits with probability p.

**Model:**
```
Input  → [Bit flip?] → Output
         (prob p)

✓ Bit unchanged    (prob 1-p)
✗ Bit flipped      (prob p)
```

**Parameters:**
- `p`: Bit flip probability (0 to 1)

**Code:**
```python
from commpy import Channels
import numpy as np

bits = np.array([1, 0, 1, 1, 0])
corrupted = Channels.bsc(bits, p=0.1)  # 10% error rate

# For reproducibility
rng = np.random.default_rng(seed=42)
corrupted = Channels.bsc(bits, p=0.1, rng=rng)
```

**Use Cases:**
- Error-correcting code testing
- Digital channel simulation
- Information theory calculations

**Parameters to vary:**
- Very low p (0.01): Clean channel
- Moderate p (0.1): Realistic channel
- High p (0.5): Severely degraded

---

### BEC (Binary Erasure Channel)

**What it does:** Erases symbols (marks as unknown) with probability p.

**Model:**
```
Input → [Erasure?] → Output
        (prob p)

✓ Symbol unchanged   (prob 1-p)
✗ Symbol erased      (prob p) → erasure_value (-1)
```

**Parameters:**
- `p`: Erasure probability (0 to 1)
- `erasure_value`: Value marking erasure (default: -1)

**Code:**
```python
from commpy import Channels

bits = np.array([1, 0, 1, 1, 0])

# Standard BEC
erased = Channels.bec(bits, p=0.2)
# Result might be: [1, -1, 1, 1, 0]

# Custom erasure marker
erased = Channels.bec(bits, p=0.2, erasure_value=None)
```

**Use Cases:**
- Packet loss simulation
- Burst communication channels
- Fountain codes

**Key Difference from BSC:**
- BEC: Unknown where error is (marked as erasure)
- BSC: Error unknown to receiver (hard to correct)

---

### AWGN (Additive White Gaussian Noise)

**What it does:** Adds Gaussian noise to signal to achieve target SNR.

**Model:**
```
Transmitted → Add Gaussian → Received
Signal        Noise         Signal
s(t)          n(t)          r(t) = s(t) + n(t)
```

**Parameters:**
- `snr_db`: Target SNR in decibels
- Noise variance calculated from signal power

**SNR Relationship:**
$$\text{SNR}_{\text{linear}} = 10^{\text{SNR}_{\text{dB}}/10}$$

**Code:**
```python
from commpy import Channels
import numpy as np

# Create signal
signal = np.array([1+1j, -1-1j, 1+1j])

# Different SNR values
noisy_clean = Channels.awgn(signal, snr_db=20)  # Very clean
noisy_moderate = Channels.awgn(signal, snr_db=10)  # Moderate
noisy_bad = Channels.awgn(signal, snr_db=0)  # Severe noise
```

**SNR Guidelines:**
- 0 dB: Signal power = Noise power (very noisy)
- 10 dB: 10× higher signal power (moderate)
- 20 dB: 100× higher signal power (clean)
- 30+ dB: Very clean channel

**For Complex Signals:**
- Noise split equally between I and Q
- Power measured as mean(|s|²)

**Real-world SNR Examples:**
- Deep space communication: -5 to +5 dB
- WiFi: +10 to +30 dB
- Landline phone: +30+ dB

---

## Information Theory

### Shannon Entropy

**What it is:** Measure of average information content (uncertainty) in a distribution.

**Intuition:**
- Uniform distribution → maximum entropy
- Deterministic (one outcome) → entropy = 0
- Measures "surprise" or "information content"

**Formula:**
$$H = -\sum_{i} p_i \log_2(p_i)$$

where $p_i$ are probabilities.

**Code:**
```python
from commpy import shannon_entropy

# Uniform: All equally likely
H_uniform = shannon_entropy([0.25, 0.25, 0.25, 0.25])
print(H_uniform)  # 2.0 bits

# Skewed: One dominant outcome
H_skewed = shannon_entropy([0.8, 0.1, 0.1])
print(H_skewed)  # ~0.92 bits

# Deterministic: Only one outcome
H_determine = shannon_entropy([1.0, 0.0])
print(H_determine)  # 0.0 bits
```

**Use Cases:**
- Information content analysis
- Capacity calculations
- Compression bounds

**Examples:**
```python
# Fair coin flip
H_coin = shannon_entropy([0.5, 0.5])  # 1.0 bit

# Biased coin (70/30)
H_biased = shannon_entropy([0.7, 0.3])  # 0.881 bits

# Six-sided die
H_die = shannon_entropy([1/6]*6)  # 2.585 bits

# Communication channel output
p_0 = [0.9, 0.05, 0.05]  # 90% 0, 5% 1, 5% 2
H_chan = shannon_entropy(p_0)  # ~0.47 bits (mostly 0)
```

---

## Waveform Generation

### IQWaveform Class

**Purpose:** Generate realistic IQ-modulated RF signals for transmission.

**Concept:**
- Baseband: Complex I/Q symbols
- RF: Modulate to carrier frequency + add pulse shaping

**Components:**
```
I(t) ──┐
       ├─→ [Pulse shape] → [Carrier] → s(t) (RF signal)
Q(t) ──┘
```

### Basic Usage

```python
from commpy import IQWaveform
import numpy as np

# Define I/Q symbols
I = np.array([1, 0, -1, 0])      # I quadrature
Q = np.array([0, 1, 0, -1])      # Q quadrature

# Create waveform
waveform = IQWaveform(
    I=I,
    Q=Q,
    T=1e-4,           # Symbol period: 100 µs
    fs=1e6,           # Sample rate: 1 MHz
    f0=1e5,           # Carrier: 100 kHz
    span=4            # Span: 4 symbols
)

# Access results
print(f"Time points: {len(waveform.t)}")
print(f"Signal shape: {waveform.s.shape}")

# Plot
waveform.plot_waveform()
```

### Parameters

**I, Q Arrays:**
- Symbol values (typically from modulation output)
- Real numbers (I/Q components of complex symbols)

**T (Symbol Period):**
- Duration of one symbol
- Units: seconds
- Example: 1 kHz symbol rate → T = 1e-3 s

**fs (Sample Rate):**
- Samples per second
- Must be > 2× maximum frequency
- Example: 1 MHz provides good resolution at 100 kHz carrier

**f0 (Carrier Frequency):**
- RF frequency
- 0 for baseband only (complex output)
- > 0 for bandpass (real output)

**pulse_shape (Callable):**
- Function defining pulse envelope
- Default: rectangular (instantaneous transitions)
- Alternative: raised cosine, Gaussian, etc.

**span:**
- Pulse duration in symbol periods
- Lower value = shorter computation
- Higher value = more realistic pulse tail

### Output Signals

**Baseband (f0 = 0):**
```python
# Complex baseband signal
waveform.s  # Complex array
waveform.s_I, waveform.s_Q  # Separate I/Q
```

**Bandpass (f0 > 0):**
```python
# Real RF signal (modulated to f0)
waveform.s  # Real array
waveform.s_I, waveform.s_Q  # Baseband I/Q before modulation
```

### Example: QPSK Modulation over RF

```python
from commpy import QPSK_Modulator, IQWaveform
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Modulate data
bits = [0, 1, 2, 3, 0, 1, 2, 3]  # 2 bits each
symbols = QPSK_Modulator.modulate(bits)

# Step 2: Extract I/Q
I = symbols.real
Q = symbols.imag

# Step 3: Generate waveform
wf = IQWaveform(
    I=I, Q=Q,
    T=1e-5,              # 10 µs symbols
    fs=1e7,              # 10 MHz sampling
    f0=1e6,              # 1 MHz RF carrier
    span=4
)

# Step 4: Analyze
print(f"Duration: {wf.t[-1]*1e6:.0f} µs")
print(f"Samples: {len(wf.s)}")

# Step 5: Visualize
plt.figure(figsize=(12, 4))
plt.plot(wf.t[:500]*1e6, wf.s[:500])  # First 500 samples
plt.xlabel('Time (µs)')
plt.ylabel('Amplitude')
plt.title('QPSK RF Signal')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## SDR Interoperability & Link Simulation

### Why these two go together

Once you can generate waveforms (previous section), two things become useful: exchanging IQ
samples with real hardware/tools, and rigorously measuring how a link performs across SNR. Both
are covered here.

### Monte-Carlo BER simulation

The pattern from earlier sections — sweep SNR, count bit errors, compare against a theoretical
curve — is common enough to warrant a dedicated, statistically sound implementation:
`simulate_ber` runs each SNR point until a target error count is reached (bounding the
estimator's variance) instead of a fixed, one-size-fits-all trial count, and reports a Wilson
score confidence interval alongside the point estimate.

```python
from commpy import Channels, MQAMModulator, plot_waterfall, simulate_ber

mod = MQAMModulator(16)
result = simulate_ber(mod, Channels.awgn, snr_db_range=[6, 8, 10, 12, 14, 16], target_errors=200)

# result.error_rate, result.ci_lower, result.ci_upper, result.n_trials are all arrays,
# one entry per SNR point.
ax = plot_waterfall(result)
ax.figure.show()
```

For frame/block error rates (or any other error metric), use the lower-level
`simulate_error_rate(trial_fn, snr_db_range, ...)`, where `trial_fn(snr_db, rng, n_trials)`
returns `(n_errors, n_trials_run)` for whatever unit of trial you define (frames, codewords, ...).

### SDR file interoperability

`write_iq`/`read_iq` write/read a headerless raw complex-sample stream, directly compatible with
GNU Radio's `blocks.file_sink`/`blocks.file_source` blocks. `write_sigmf`/`read_sigmf` add a
[SigMF](https://github.com/sigmf/SigMF) `.sigmf-meta` JSON sidecar recording sample rate, center
frequency, and free-text description alongside the `.sigmf-data` samples — useful when a
recording needs to be self-describing (e.g. archiving a capture, or handing it to a teammate).

```python
from commpy import write_iq, read_iq, write_sigmf, read_sigmf

symbols = mod.modulate(bits)

# Raw binary: GNU Radio compatible, no metadata.
write_iq('recording.cf32', symbols)
recovered = read_iq('recording.cf32')

# SigMF: adds sample rate / center frequency / description metadata.
write_sigmf('capture', symbols, sample_rate=1e6, center_freq=915e6, description='16-QAM test')
recovered, meta = read_sigmf('capture')
print(meta['global']['core:sample_rate'], meta['captures'][0]['core:frequency'])
```

---

## Practical Applications

### Application 1: System Performance Evaluation

**Goal:** Measure how system performs at different SNR levels.

```python
import numpy as np
import matplotlib.pyplot as plt
from commpy import BPSK_Modulator, Channels

# SNR range to test
snr_range = np.arange(-5, 16, 1)  # -5 to 15 dB
num_bits = 10000
ber_list = []

for snr in snr_range:
    # Generate random bits
    bits = np.random.randint(0, 2, num_bits)
    
    # Modulate
    symbols = BPSK_Modulator.modulate(bits)
    
    # Add AWGN
    received = Channels.awgn(symbols, snr_db=snr)
    
    # Demodulate
    recovered = BPSK_Modulator.demodulate(received)
    
    # Calculate BER
    errors = np.sum(recovered != bits)
    ber = errors / num_bits
    ber_list.append(ber)
    print(f"SNR {snr:+3.0f} dB: BER = {ber:.4f}")

# Plot results
plt.semilogy(snr_range, ber_list, 'bo-')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate')
plt.title('Performance Curve')
plt.grid(True, alpha=0.3, which='both')
plt.show()
```

### Application 2: Channel Comparison

**Goal:** Compare performance over different channel types.

```python
from commpy import BPSK_Modulator, Channels

bits = np.random.randint(0, 2, 1000)
symbols = BPSK_Modulator.modulate(bits)

# AWGN Channel
noisy_awgn = Channels.awgn(symbols, snr_db=10)
recovered_awgn = BPSK_Modulator.demodulate(noisy_awgn)
ber_awgn = np.mean(recovered_awgn != bits)

# BSC Channel (same SNR approximately)
# Convert to bits for BSC
noisy_bsc = Channels.bsc(bits, p=0.05)
ber_bsc = np.mean(noisy_bsc != bits)

# BEC Channel
noisy_bec = Channels.bec(bits, p=0.05)
erasures = np.sum(noisy_bec == -1)

print(f"AWGN: BER = {ber_awgn:.4f}")
print(f"BSC:  BER = {ber_bsc:.4f}")
print(f"BEC:  {erasures} erasures out of {len(bits)}")
```

### Application 3: Error Correcting Code Testing

**Goal:** Test how codes perform over noisy channels.

```python
from commpy import Channels

# Simulate coded bits over BSC
#original_bits → [Encoder] → coded_bits → [BSC] → received → [Decoder] → recovered

def simple_repetition_code(bits, repetitions=3):
    """Repeat each bit N times."""
    return np.repeat(bits, repetitions)

def simple_decoder(received, repetitions=3):
    """Majority voting for received bits."""
    return np.array([
        np.mean(received[i*repetitions:(i+1)*repetitions]) > 0.5
        for i in range(len(received)//repetitions)
    ]).astype(int)

# Test
bits = np.array([1, 0, 1, 1, 0])
coded = simple_repetition_code(bits, repetitions=3)
noisy = Channels.bsc(coded, p=0.1)
recovered = simple_decoder(noisy, repetitions=3)

print(f"Original:  {bits}")
print(f"Recovered: {recovered}")
print(f"Errors: {np.sum(recovered != bits)}")
```

---

## Best Practices

### 1. Reproducibility

Always use seeded RNG for reproducible results:

```python
# Good: Reproducible
rng1 = np.random.default_rng(seed=42)
rng2 = np.random.default_rng(seed=42)
result1 = Channels.awgn(signal, snr_db=10, rng=rng1)
result2 = Channels.awgn(signal, snr_db=10, rng=rng2)
assert np.allclose(result1, result2)  # True

# Bad: Non-reproducible
result1 = Channels.awgn(signal, snr_db=10)
result2 = Channels.awgn(signal, snr_db=10)
# Different every time
```

### 2. Large-scale Simulations

For extensive simulations, organize code:

```python
class ModulationSim:
    """Reusable simulation structure."""
    
    def __init__(self, modulator_class, channel_model):
        self.modulator = modulator_class
        self.channel = channel_model
    
    def run_ber_test(self, snr_list, num_trials=100):
        results = []
        for snr in snr_list:
            ber_values = []
            for trial in range(num_trials):
                bits = self.generate_bits()
                symbols = self.modulator.modulate(bits)
                noisy = self.channel.apply(symbols, snr_db=snr)
                recovered = self.modulator.demodulate(noisy)
                ber = np.mean(recovered != bits)
                ber_values.append(ber)
            results.append(np.mean(ber_values))
        return results
```

### 3. Memory Management

For large signals, consider:

```python
# Problematic: Stores all in memory
all_symbols = []
for i in range(1000000):
    symbols = BPSK_Modulator.modulate([...])
    all_symbols.append(symbols)

# Better: Process in batches
chunk_size = 10000
for i in range(0, 1000000, chunk_size):
    symbols = BPSK_Modulator.modulate(bits[i:i+chunk_size])
    process(symbols)  # Don't store
```

### 4. Visualization

Always visualize results:

```python
import matplotlib.pyplot as plt

# Constellation plot
plt.scatter(symbols.real, symbols.imag, alpha=0.5)
plt.xlabel('I')
plt.ylabel('Q')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

# Time-domain signal
plt.plot(waveform.t, waveform.s)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# BER curve
plt.semilogy(snr_list, ber_list)
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.grid(True, alpha=0.3, which='both')
plt.show()
```

### 5. Documentation

Document your code:

```python
def ber_simulation(modulator_cls, snr_db, num_bits=10000):
    """Simulate BER for given modulation and SNR.
    
    Parameters:
    - modulator_cls: Modulation class (e.g., BPSK_Modulator)
    - snr_db: Target SNR in dB
    - num_bits: Number of bits to simulate
    
    Returns:
    - BER (float): Estimated bit error rate
    """
```

