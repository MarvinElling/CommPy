# CommPy Frequently Asked Questions (FAQ)

Common questions about CommPy and their answers.

## Installation & Setup

### Q: How do I install CommPy?

**A:** Install via pip:

```bash
pip install commpy
```

Or from source:

```bash
git clone <repo-url>
cd CommPy
pip install -e .
```

See [GETTING_STARTED.md](GETTING_STARTED.md#installation) for detailed instructions.

---

### Q: What are the system requirements?

**A:** CommPy requires:

- Python 3.10 or higher
- NumPy
- Matplotlib (for plotting)

Operating System: Linux, macOS, Windows

---

### Q: How do I verify the installation?

**A:** Run this Python code:

```python
import commpy
from commpy import BPSK_Modulator

bits = [0, 1, 0, 1]
symbols = BPSK_Modulator.modulate(bits)
print(f"Success! Modulated: {symbols}")
```

If no errors appear, installation is successful.

---

### Q: I got an ImportError. What should I do?

**A:** Try these steps:

1. **Verify installation:**
   ```bash
   pip list | grep commpy
   ```

2. **Check Python path:**
   ```python
   import sys
   print(sys.path)
   ```

3. **Reinstall:**
   ```bash
   pip install --upgrade --force-reinstall commpy
   ```

See [GETTING_STARTED.md#troubleshooting](GETTING_STARTED.md#troubleshooting) for more.

---

## Modulation & Demodulation

### Q: Which modulation should I use?

**A:** Choose based on your needs:

| Need | Recommendation |
|------|---|
| Simple, robust | BPSK |
| Balance efficiency & robustness | QPSK |
| Very high efficiency | PSK-8 |
| Amplitude-based | ASK-2 or ASK-4 |
| Optical communication | OOK |

See [USER_GUIDE.md#modulation--demodulation](USER_GUIDE.md#modulation--demodulation) for detailed comparisons.

---

### Q: How do I modulate multiple bits?

**A:** Modulate arrays directly:

```python
from commpy import BPSK_Modulator

# Single array - any length
bits = [0, 1, 0, 1, 1, 0, 1, 1, 0]
symbols = BPSK_Modulator.modulate(bits)

# Or numpy array
import numpy as np
bits = np.array([0, 1, 0, 1])
symbols = BPSK_Modulator.modulate(bits)
```

Works with lists, tuples, or numpy arrays.

---

### Q: What's the difference between QPSK and ASK-4?

**A:** Both encode 2 bits per symbol, but differently:

| Aspect | QPSK | ASK-4 |
|--------|------|-------|
| Constellation | 4 phases | 4 amplitudes |
| Robustness | Better | Worse (sensitive to amplitude) |
| Practical use | WiFi, cellular | Some specialized systems |
| Recommendation | Generally better | For specific applications |

See [USER_GUIDE.md](USER_GUIDE.md) for details.

---

### Q: Can I create my own modulation scheme?

**A:** Currently, CommPy provides built-in classes. To create custom:

1. Implement modulate/demodulate methods
2. Return complex numpy array from modulate
3. See [CONTRIBUTING.md](CONTRIBUTING.md) for adding to CommPy

Example structure:

```python
class CustomModulator:
    @staticmethod
    def modulate(bits):
        # Map bits to symbols
        return np.array([...])
    
    @staticmethod
    def demodulate(symbols):
        # Map symbols back to bits
        return np.array([...])
```

---

## Channels & Noise

### Q: What SNR should I use for testing?

**A:** Depends on your scenario:

```python
# Very clean (lab conditions)
Channels.awgn(signal, snr_db=20)

# Typical wireless
Channels.awgn(signal, snr_db=10)

# Poor conditions
Channels.awgn(signal, snr_db=5)

# Extreme noise (research/testing)
Channels.awgn(signal, snr_db=0)
```

See [USER_GUIDE.md#awgn](USER_GUIDE.md#awgn---additive-white-gaussian-noise) for guidance.

---

### Q: What's the difference between BSC, BEC, and AWGN?

**A:**

| Channel | What Happens | Use Case |
|---------|---|---|
| **BSC** | Random bit flips | Digital channels, error-correcting codes |
| **BEC** | Symbols erasure (lost) | Packet loss, burst errors |
| **AWGN** | Gaussian noise added | Analog/wireless channels |

See [USER_GUIDE.md#channel-models](USER_GUIDE.md#channel-models).

---

### Q: How do I make simulations reproducible?

**A:** Use seeded random number generators:

```python
import numpy as np
from commpy import Channels

# Create seeded RNG
rng = np.random.default_rng(seed=42)

# Use everywhere that needs randomness
noisy1 = Channels.awgn(signal, snr_db=10, rng=rng)

# New RNG with same seed = identical results
rng = np.random.default_rng(seed=42)
noisy2 = Channels.awgn(signal, snr_db=10, rng=rng)

print(np.allclose(noisy1, noisy2))  # True
```

See [GETTING_STARTED.md#tutorial-6](GETTING_STARTED.md#tutorial-6-reproducible-results-with-rng).

---

### Q: How do I simulate realistic channels?

**A:** Combine multiple effects:

```python
from commpy import BPSK_Modulator, Channels

bits = np.random.randint(0, 2, 1000)
symbols = BPSK_Modulator.modulate(bits)

# 1. Add noise
noisy = Channels.awgn(symbols, snr_db=10)

# 2. Simulate some bit errors (simplified fading)
received_bits = BPSK_Modulator.demodulate(noisy)
faded = Channels.bsc(received_bits, p=0.02)

# 3. Demodulate final result
recovered = BPSK_Modulator.modulate(faded)
```

See [USER_GUIDE.md#practical-applications](USER_GUIDE.md#practical-applications).

---

## Information Theory

### Q: What is Shannon entropy?

**A:** It measures average information content (uncertainty) in a probability distribution.

- **High entropy**: Uncertain, lots of information
- **Low entropy**: Predictable, little information

```python
from commpy import shannon_entropy

# Uniform (most uncertain)
H_uniform = shannon_entropy([0.25, 0.25, 0.25, 0.25])
# → 2.0 bits

# Skewed (less uncertain)
H_skewed = shannon_entropy([0.8, 0.1, 0.1])
# → 0.92 bits

# Deterministic (no uncertainty)
H_det = shannon_entropy([1.0, 0.0])
# → 0.0 bits
```

---

### Q: What are the bounds on entropy?

**A:** For N outcomes:

- **Minimum**: 0 bits (deterministic)
- **Maximum**: log₂(N) bits (uniform distribution)

Example for 4 outcomes:
- Minimum possible: 0 bits
- Maximum possible: 2 bits
- Maximum is when all equally likely

---

## Waveforms

### Q: How do I generate a waveform?

**A:**

```python
from commpy import IQWaveform
import numpy as np

# 1. Create I/Q symbols
I = np.array([1, 0, -1, 0])
Q = np.array([0, 1, 0, -1])

# 2. Create waveform
wf = IQWaveform(
    I=I, Q=Q,
    T=1e-4,        # Symbol period: 100 µs
    fs=1e6,        # Sample rate: 1 MHz
    f0=1e5         # Carrier: 100 kHz
)

# 3. Access results
print(f"Signal: {wf.s[:100]}")
print(f"Time: {wf.t[:100]}")

# 4. Plot
wf.plot_waveform()
```

See [GETTING_STARTED.md#tutorial-5](GETTING_STARTED.md#tutorial-5-iq-waveform-generation).

---

### Q: What do I and Q mean?

**A:** I/Q (In-phase/Quadrature) components:

- **I**: Real part (cosine component)
- **Q**: Imaginary part (sine component)

Together they represent a complex signal:
```
Complex symbol = I + j*Q
```

Example:
```python
symbols = QPSK_Modulator.modulate([0, 1, 2, 3])
I_values = symbols.real  # I components
Q_values = symbols.imag  # Q components
```

---

### Q: What does `f0` (carrier frequency) do?

**A:**

- `f0=0`: Baseband output (complex signal)
- `f0>0`: Bandpass RF signal (real signal modulated to `f0`)

```python
# Baseband (complex)
wf_bb = IQWaveform(I=I, Q=Q, T=T, fs=fs, f0=0)
print(wf_bb.s.dtype)  # complex128

# RF (real)
wf_rf = IQWaveform(I=I, Q=Q, T=T, fs=fs, f0=1e6)
print(wf_rf.s.dtype)  # float64
```

---

### Q: How do I choose `fs` (sample rate)?

**A:** Must satisfy Nyquist: `fs > 2 * f0`

```python
f0 = 1e6  # 1 MHz carrier

# Must be > 2 MHz
fs_good = 5e6    # 5 MHz ✓
fs_bad = 1e6     # 1 MHz ✗ Too low

# Typical: 4× to 10× carrier frequency
fs_typical = 4 * f0  # 4 MHz
```

---

## Simulations & Analysis

### Q: How do I measure Bit Error Rate (BER)?

**A:**

```python
import numpy as np
from commpy import BPSK_Modulator, Channels

# 1. Generate data
bits = np.random.randint(0, 2, 10000)

# 2. Transmit
symbols = BPSK_Modulator.modulate(bits)
received = Channels.awgn(symbols, snr_db=5)

# 3. Receive
recovered = BPSK_Modulator.demodulate(received)

# 4. Calculate BER
errors = np.sum(recovered != bits)
ber = errors / len(bits)
print(f"BER = {ber:.4f}")
```

See [GETTING_STARTED.md#tutorial-2](GETTING_STARTED.md#tutorial-2-bit-error-rate-ber-simulation).

---

### Q: How do I create a BER vs SNR curve?

**A:** Loop over SNR values:

```python
import matplotlib.pyplot as plt

snr_range = np.arange(0, 11, 1)
ber_list = []

for snr in snr_range:
    bits = np.random.randint(0, 2, 10000)
    symbols = BPSK_Modulator.modulate(bits)
    noisy = Channels.awgn(symbols, snr_db=snr)
    recovered = BPSK_Modulator.demodulate(noisy)
    ber = np.mean(recovered != bits)
    ber_list.append(ber)

# Plot
plt.semilogy(snr_range, ber_list, 'bo-')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.grid(True, alpha=0.3, which='both')
plt.show()
```

See [GETTING_STARTED.md#tutorial-2](GETTING_STARTED.md#tutorial-2-bit-error-rate-ber-simulation).

---

## Performance & Optimization

### Q: How do I speed up large simulations?

**A:** Use batch processing:

```python
# Inefficient: Many small operations
for i in range(1000000):
    bit = get_bit(i)
    symbol = modulate(bit)
    process(symbol)

# Efficient: Batch operations
for i in range(0, 1000000, 10000):
    bits = [get_bit(j) for j in range(i, i+10000)]
    symbols = modulate(bits)  # Batch
    process(symbols)  # Batch
```

---

### Q: What array types work?

**A:** Most CommPy functions accept:

- Python lists: `[0, 1, 0, 1]`
- NumPy arrays: `np.array([...])`
- Tuples: `(0, 1, 0, 1)`

Output is always NumPy arrays.

```python
# All work the same way
symbols1 = BPSK_Modulator.modulate([0, 1])
symbols2 = BPSK_Modulator.modulate((0, 1))
symbols3 = BPSK_Modulator.modulate(np.array([0, 1]))

np.allclose(symbols1, symbols2)  # True
np.allclose(symbols2, symbols3)  # True
```

---

## Troubleshooting

### Q: I'm getting shape mismatch errors

**A:** Ensure input is correct shape:

```python
# ✓ Correct
bits = np.array([0, 1, 0, 1])  # 1D array
symbols = BPSK_Modulator.modulate(bits)

# ✗ Wrong
bits = np.array([[0, 1], [0, 1]])  # 2D array
symbols = BPSK_Modulator.modulate(bits)  # Error!
```

Use `reshape` if needed:

```python
bits = bits.reshape(-1)  # Flatten to 1D
```

---

### Q: My demodulation results are wrong

**A:** Check:

1. **Input shape**: Should match modulation output
2. **Data type**: Complex for modulation output
3. **Noise level**: Very noisy → higher error rate

```python
# Debug
print(f"Modulated shape: {symbols.shape}")
print(f"Modulated dtype: {symbols.dtype}")
print(f"Modulated sample: {symbols[0]}")

# Test without noise first
recovered = BPSK_Modulator.demodulate(symbols)
assert np.allclose(recovered, original_bits)
```

---

### Q: Plotting isn't showing

**A:** Add `plt.show()` or use Jupyter:

```python
# In script
import matplotlib.pyplot as plt
wf.plot_waveform()
plt.show()  # Added this line!

# In Jupyter
%matplotlib inline
wf.plot_waveform()  # Shows automatically
```

---

## Contributing & Development

### Q: How do I report a bug?

**A:** See [CONTRIBUTING.md#reporting-issues](../CONTRIBUTING.md#reporting-issues).

Include:
1. **Title**: Clear description
2. **Environment**: Python version, CommPy version
3. **Minimal code**: Reproduces the issue
4. **Expected vs Actual**: What should happen vs what happens

---

### Q: How do I contribute code?

**A:** See [CONTRIBUTING.md#getting-started](../CONTRIBUTING.md#getting-started).

Steps:
1. Fork repository
2. Create feature branch
3. Add tests
4. Submit pull request

---

### Q: What's the code style?

**A:** Follow PEP 8 with:

- 4 spaces for indentation
- 100 character line length
- Type hints on all functions
- Comprehensive docstrings

See [CONTRIBUTING.md#code-style](../CONTRIBUTING.md#code-style).

---

## More Help

- **Tutorials**: [GETTING_STARTED.md](GETTING_STARTED.md)
- **API Reference**: [API.md](API.md)
- **User Guide**: [USER_GUIDE.md](USER_GUIDE.md)
- **Quick Reference**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Documentation Index**: [INDEX.md](INDEX.md)

Can't find answer? Open an issue on GitHub!

---

*Last updated: March 2026*

