# Getting Started with CommPy

A beginner's guide to using the CommPy communication engineering library.

## Installation

### From PyPI

```bash
pip install commpy
```

### From Source

```bash
git clone <repository-url>
cd CommPy
pip install -e .
```

### Verify Installation

```python
import commpy
print(commpy.__version__)

# Test a simple modulation
from commpy import BPSK_Modulator
bits = [0, 1, 0, 1]
symbols = BPSK_Modulator.modulate(bits)
print(symbols)
```

---

## Basic Concepts

### What is Digital Modulation?

Digital modulation converts binary data into analog signals suitable for transmission over wireless or wired channels.

CommPy provides several modulation schemes:

| Scheme | Bits/Symbol | Constellation | Use Case |
|--------|-------------|---------------|----------|
| BPSK | 1 | 2 points | Simple, robust |
| QPSK | 2 | 4 points | Bandwidth efficient |
| ASK-2 | 1 | 2 amplitudes | Amplitude-based |
| ASK-4 | 2 | 4 amplitudes | Bandwidth efficient |
| PSK-8 | 3 | 8 phases | High spectral efficiency |
| OOK | 1 | On/Off | Simple, optical |

### What are Channels?

Channels model real-world transmission impairments:

**BSC (Binary Symmetric Channel)**
- Random bit flips with probability `p`
- Useful for error-correcting code testing

**BEC (Binary Erasure Channel)**
- Symbols are erased (lost) with probability `p`
- Models packet loss scenarios

**AWGN (Additive White Gaussian Noise)**
- Gaussian noise added to signal
- Most common wireless channel model

---

## Tutorial 1: Simple Modulation & Demodulation

Let's modulate some bits, simulate transmission, and recover them.

```python
import numpy as np
from commpy import BPSK_Modulator, Channels

# Step 1: Create test data
bits = np.array([1, 0, 1, 1, 0, 1, 0, 0])
print(f"Original bits: {bits}")

# Step 2: Modulate using BPSK
symbols = BPSK_Modulator.modulate(bits)
print(f"Modulated symbols: {symbols}")

# Step 3: Transmit through AWGN channel
received = Channels.awgn(symbols, snr_db=5.0)
print(f"Received (noisy): {received}")

# Step 4: Demodulate
recovered = BPSK_Modulator.demodulate(received)
print(f"Recovered bits: {recovered}")

# Step 5: Check errors
errors = np.sum(recovered != bits)
print(f"Bit errors: {errors}/{len(bits)}")
```

**Output:**
```
Original bits: [1 0 1 1 0 1 0 0]
Modulated symbols: [ 1.+0.j -1.+0.j  1.+0.j  1.+0.j -1.+0.j  1.+0.j -1.+0.j -1.+0.j]
Received (noisy): [ 0.85+0.2j -1.1-0.15j ...]
Recovered bits: [1 0 1 1 0 1 0 0]
Bit errors: 0/8
```

---

## Tutorial 2: Bit Error Rate (BER) Simulation

Create a BER curve showing performance vs SNR.

```python
import numpy as np
import matplotlib.pyplot as plt
from commpy import BPSK_Modulator, Channels

# Parameters
snr_values = np.arange(0, 11, 2)  # 0, 2, 4, ..., 10 dB
num_bits = 10000
ber_values = []

# Loop over SNR values
for snr in snr_values:
    # Generate random bits
    bits = np.random.randint(0, 2, num_bits)
    
    # Modulate
    symbols = BPSK_Modulator.modulate(bits)
    
    # Add noise
    received = Channels.awgn(symbols, snr_db=snr)
    
    # Demodulate
    recovered = BPSK_Modulator.demodulate(received)
    
    # Calculate BER
    errors = np.sum(recovered != bits)
    ber = errors / num_bits
    ber_values.append(ber)
    print(f"SNR={snr:2d} dB: BER={ber:.4f}")

# Plot results
plt.figure(figsize=(8, 5))
plt.semilogy(snr_values, ber_values, 'bo-', label='BPSK')
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate')
plt.title('BER vs SNR for BPSK')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

---

## Tutorial 3: Channel Comparison

Compare different channel models.

```python
import numpy as np
from commpy import Channels

bits = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])

print("Original bits:", bits)
print()

# BSC with 20% error rate
bsc_out = Channels.bsc(bits, p=0.2)
print("BSC (20% error):", bsc_out)
print()

# BEC with 20% erasure rate
bec_out = Channels.bec(bits, p=0.2, erasure_value=-1)
print("BEC (20% erasure):", bec_out)
print()

# Count effects
bsc_errors = np.sum(bsc_out != bits)
bec_erasures = np.sum(bec_out == -1)
print(f"BSC: {bsc_errors} bit flips")
print(f"BEC: {bec_erasures} erasures")
```

---

## Tutorial 4: Different Modulation Schemes

Compare BPSK, QPSK, and ASK.

```python
import numpy as np
from commpy import BPSK_Modulator, QPSK_Modulator, ASK_4_Modulator

# For BPSK: 1 bit per symbol
bits_bpsk = [0, 1, 0, 1, 1, 0]
symbols_bpsk = BPSK_Modulator.modulate(bits_bpsk)
print("BPSK symbols (1 bit each):")
print(f"  {bits_bpsk} → {symbols_bpsk}")
print()

# For QPSK: 2 bits per symbol (grouped as 2-bit values)
bits_qpsk = [0, 1, 2, 3]  # 00, 01, 10, 11 in binary
symbols_qpsk = QPSK_Modulator.modulate(bits_qpsk)
print("QPSK symbols (2 bits each):")
print(f"  {bits_qpsk} → {symbols_qpsk}")
print()

# For ASK-4: 2 bits per symbol (4 amplitude levels)
bits_ask = [0, 1, 2, 3]
symbols_ask = ASK_4_Modulator.modulate(bits_ask)
print("ASK-4 symbols (2 bits each):")
print(f"  {bits_ask} → {symbols_ask}")
```

---

## Tutorial 5: IQ Waveform Generation

Generate an IQ modulated RF signal.

```python
import numpy as np
import matplotlib.pyplot as plt
from commpy import IQWaveform, BPSK_Modulator

# Step 1: Create bit sequence and modulate to BPSK
bits = [0, 1, 0, 1, 1, 0]
bpsk_symbols = BPSK_Modulator.modulate(bits)

# Step 2: Extract I and Q components
I = bpsk_symbols.real
Q = bpsk_symbols.imag

print(f"I symbols: {I}")
print(f"Q symbols: {Q}")

# Step 3: Create IQ waveform
waveform = IQWaveform(
    I=I,
    Q=Q,
    T=1e-4,           # 100 µs symbol period
    fs=1e6,           # 1 MHz sampling rate
    f0=100e3,         # 100 kHz carrier frequency
    span=4
)

# Step 4: Analyze the waveform
print(f"\nWaveform Statistics:")
print(f"  Duration: {waveform.t[-1]*1e6:.1f} µs")
print(f"  Samples: {len(waveform.t)}")
print(f"  Signal RMS: {np.sqrt(np.mean(waveform.s**2)):.3f}")

# Step 5: Plot
waveform.plot_waveform()
plt.show()

# Optional: Save signal data
np.savetxt('waveform_signal.txt', waveform.s)
np.savetxt('waveform_time.txt', waveform.t)
```

---

## Tutorial 6: using Reproducible Results with RNG

Use seeded random number generators for reproducible simulations.

```python
import numpy as np
from commpy import BPSK_Modulator, Channels

# Create a seeded RNG
rng = np.random.default_rng(seed=42)

# Use the same seed for consistent results
bits = np.array([0, 1, 0, 1, 1, 0])
symbols = BPSK_Modulator.modulate(bits)

# First transmission with seed 42
received_1 = Channels.awgn(symbols, snr_db=5, rng=np.random.default_rng(42))

# Second transmission with same seed
received_2 = Channels.awgn(symbols, snr_db=5, rng=np.random.default_rng(42))

# They should be identical
print(np.allclose(received_1, received_2))  # True
```

---

## Common Patterns

### Pattern 1: Monte Carlo Simulation

```python
def monte_carlo_ber(snr_db, num_trials=1000):
    """Estimate BER for given SNR."""
    ber_list = []
    for _ in range(num_trials):
        bits = np.random.randint(0, 2, 1000)
        symbols = BPSK_Modulator.modulate(bits)
        received = Channels.awgn(symbols, snr_db=snr_db)
        recovered = BPSK_Modulator.demodulate(received)
        ber = np.sum(recovered != bits) / len(bits)
        ber_list.append(ber)
    return np.mean(ber_list)

ber = monte_carlo_ber(snr_db=10)
```

### Pattern 2: Batch Processing

```python
def batch_modulate(bitstream, modulator_class):
    """Modulate multiple frames."""
    frames = [bitstream[i:i+100] for i in range(0, len(bitstream), 100)]
    return [modulator_class.modulate(frame) for frame in frames]

bits = np.random.randint(0, 2, 1000)
symbols = batch_modulate(bits, BPSK_Modulator)
```

### Pattern 3: Channel Cascade

```python
def apply_channels(signal):
    """Apply multiple channel effects."""
    # Add noise
    noisy = Channels.awgn(signal, snr_db=10)
    # Convert symbols to bits, apply BSC, convert back
    # ... (depends on application)
    return noisy
```

---

## Troubleshooting

### Issue: ImportError when importing commpy

**Solution:** Ensure package is installed:
```bash
pip install -e .  # From repo root
```

### Issue: Shape mismatch errors

**Solution:** Check input shapes match expectations:
```python
bits = np.array([0, 1])  # Must be 1D
symbols = BPSK_Modulator.modulate(bits)
```

### Issue: Unexpected BER values

**Solution:** Verify SNR calculation and ensure:
- Units are in dB
- SNR is reasonable (negative = very noisy, >20 = very clean)
- Signal and noise powers are correct

### Issue: Plotting not showing

**Solution:** Add `plt.show()` or enable interactive mode:
```python
%matplotlib inline  # In Jupyter
plt.show()          # In scripts
```

---

## Next Steps

1. **Explore Examples**: Check the `examples/` directory
2. **Read API Docs**: See [API Reference](API.md)
3. **Run Tests**: `pytest tests/`
4. **Experiment**: Modify examples for your use case
5. **Contribute**: Submit issues or pull requests

---

## Resources

- **Documentation**: [API Reference](API.md)
- **GitHub**: [Repository](https://github.com/...)
- **Issues**: Report bugs or request features
- **Discussions**: Ask questions or share ideas

---

## Quick Reference Cheat Sheet

```python
from commpy import *

# Modulation
symbols = BPSK_Modulator.modulate(bits)
bits = BPSK_Modulator.demodulate(symbols)

# Channels
noisy = Channels.awgn(signal, snr_db=10)
degraded = Channels.bsc(bits, p=0.1)
erased = Channels.bec(bits, p=0.1)

# Information Theory
H = shannon_entropy([0.5, 0.25, 0.25])

# Waveforms
wf = IQWaveform(I=I, Q=Q, T=1e-3, fs=1e6, f0=0)

# Utilities
is_prime(7)
modinv(3, 7)
```

