# CommPy Quick Reference Card

A one-page reference for the most common CommPy functions.

## Installation
```bash
pip install commpy
```

## Import Statement
```python
from commpy import *
```

---

## Modulation Classes

All modulation classes use `.modulate()` and `.demodulate()` static methods.

### BPSK (Binary Phase Shift Keying)
```python
from commpy import BPSK_Modulator

# Modulate
symbols = BPSK_Modulator.modulate([0, 1, 0, 1])
# → [-1+0j, 1+0j, -1+0j, 1+0j]

# Demodulate  
bits = BPSK_Modulator.demodulate(symbols)
# → [0, 1, 0, 1]
```

### QPSK (Quadrature Phase Shift Keying)
```python
from commpy import QPSK_Modulator

# 2 bits per symbol (values 0-3)
symbols = QPSK_Modulator.modulate([0, 1, 2, 3])
bits = QPSK_Modulator.demodulate(symbols)
```

### ASK (Amplitude Shift Keying)
```python
from commpy import ASK_2_Modulator, ASK_4_Modulator

# 2 levels
symbols = ASK_2_Modulator.modulate([0, 1, 0, 1])

# 4 levels (2 bits per symbol)
symbols = ASK_4_Modulator.modulate([0, 1, 2, 3])
```

### Other Schemes
```python
from commpy import PSK_8_Modulator, OOK_Modulator

# 8-PSK (3 bits per symbol)
symbols = PSK_8_Modulator.modulate([0, 1, 2, 3, 4, 5, 6, 7])

# OOK (On-Off Keying)
symbols = OOK_Modulator.modulate([0, 1, 0, 1])
```

---

## Channel Models

```python
from commpy import Channels
import numpy as np
```

### Binary Symmetric Channel (BSC)
```python
# Bit flip probability
corrupted = Channels.bsc(bits, p=0.1)

# With reproducible RNG
rng = np.random.default_rng(seed=42)
corrupted = Channels.bsc(bits, p=0.1, rng=rng)
```

### Binary Erasure Channel (BEC)
```python
# Erasure probability, mark with -1
erased = Channels.bec(bits, p=0.1, erasure_value=-1)
```

### AWGN (Additive White Gaussian Noise)
```python
# Add noise to achieve SNR in dB
noisy = Channels.awgn(signal, snr_db=10)

# With reproducible RNG
rng = np.random.default_rng(seed=42)
noisy = Channels.awgn(signal, snr_db=10, rng=rng)
```

---

## Information Theory

```python
from commpy import shannon_entropy

# Calculate Shannon entropy
H = shannon_entropy([0.25, 0.25, 0.5])
# → 1.5 bits
```

---

## Waveform Generation

```python
from commpy import IQWaveform
import numpy as np

# Create IQ symbols
I = np.array([1, 0, -1, 0])
Q = np.array([0, 1, 0, -1])

# Generate waveform
wf = IQWaveform(
    I=I, Q=Q,
    T=1e-4,        # Symbol period (s)
    fs=1e6,        # Sample rate (Hz)
    f0=1e5,        # Carrier freq (Hz), 0=baseband
    span=4
)

# Access results
print(wf.t)        # Time vector
print(wf.s)        # RF signal
print(wf.s_I)      # I component
print(wf.s_Q)      # Q component

# Plot
wf.plot_waveform()
```

---

## Utility Functions

```python
from commpy import is_prime, modinv, PrimeField

# Prime checking
is_prime(7)        # → True

# Modular inverse
modinv(3, 7)       # → 5 (since 3*5 ≡ 1 mod 7)

# Finite field arithmetic
gf = PrimeField(7)
gf.add(5, 3)       # → 1
gf.multiply(3, 2)  # → 6
gf.inverse(3)      # → 5
gf.power(2, 3)     # → 1 (2^3 mod 7)
```

---

## Visualization

Every function takes an optional `ax` and returns it. None call `plt.show()`.

```python
from commpy import (
    plot_constellation, plot_eye_diagram, plot_psd, plot_spectrogram,
    plot_iq_time, plot_filter_response,
)

plot_constellation(mod, received=rx, labels=True, regions=True)
plot_eye_diagram(waveform, sps=8)          # complex input draws I and Q
plot_psd(waveform, fs=8.0)                 # two-sided, centered on DC
plot_spectrogram(waveform, fs=8.0)
plot_iq_time(waveform, fs=8.0)
plot_filter_response(taps, fs=8.0)         # impulse + magnitude panels
```

```python
from commpy import (
    plot_parity_check, plot_tanner_graph, plot_trellis,
    plot_polar_reliabilities, plot_frozen_bits, plot_interleaver,
)

plot_parity_check(ldpc)                    # sparsity + density
plot_tanner_graph(ldpc)                    # small codes only
plot_trellis(trellis, n_stages=4, labels=True)
plot_polar_reliabilities(256)              # ranks, not raw scores
plot_frozen_bits(polar)
plot_interleaver(turbo)
```

```python
from commpy import (
    plot_llr_histogram, plot_decoder_convergence, plot_exit_chart,
    plot_viterbi_paths, plot_scl_paths,
)

plot_llr_histogram(llr, bits=sent)         # bit 0 must land positive
plot_decoder_convergence(ldpc, llr, max_iter=20)
plot_exit_chart(rsc_trellis, snr_db=-4.0)  # snr_db is Es/N0
plot_viterbi_paths(trellis, received)
plot_scl_paths(llr, polar.frozen, list_size=8)
```

```python
from commpy import (
    plot_error_rate_comparison, plot_channel_response, plot_equalizer_response,
    plot_ofdm_grid, plot_papr_ccdf, plot_mimo_capacity_cdf, plot_capacity_curves,
)

plot_error_rate_comparison({'uncoded': a, 'LDPC': b})   # keeps the CIs
plot_channel_response(taps)
plot_equalizer_response(taps, zf_equalizer(taps, 15))
plot_ofdm_grid(grid, active_subcarriers=mod.active_subcarriers)
plot_papr_ccdf(ofdm_symbols)
plot_mimo_capacity_cdf(4, 4, [0, 10, 20])
plot_capacity_curves()
```

```python
from commpy import animate_constellation, animate_decoding, animate_viterbi

anim = animate_constellation(mod, [20, 16, 12, 8])   # keep the reference!
anim.save('sweep.gif', writer='pillow', fps=3)
```

```python
from commpy import commpy_style, series_colors

with commpy_style():        # your figures, CommPy's look
    ...
series_colors(3)            # first 3 slots; stable across lengths
```

Optional interactive backend — `pip install "commpy[viz]"`:

```python
from commpy import plotly_constellation, plotly_eye_diagram, plotly_psd
from commpy import plotly_waterfall, plotly_tanner_graph
```

---

## Common Patterns

### Complete Simulation
```python
# Generate bits
bits = np.random.randint(0, 2, 100)

# Modulate
symbols = BPSK_Modulator.modulate(bits)

# Transmit through channel
noisy = Channels.awgn(symbols, snr_db=5)

# Demodulate
recovered = BPSK_Modulator.demodulate(noisy)

# Measure error
ber = np.mean(recovered != bits)
```

### BER Curve
```python
snr_range = np.arange(0, 11, 2)
ber_list = []

for snr in snr_range:
    bits = np.random.randint(0, 2, 1000)
    symbols = BPSK_Modulator.modulate(bits)
    noisy = Channels.awgn(symbols, snr_db=snr)
    recovered = BPSK_Modulator.demodulate(noisy)
    ber = np.mean(recovered != bits)
    ber_list.append(ber)
```

### Reproducible Results
```python
# Same seed → same results
rng = np.random.default_rng(seed=42)
result1 = Channels.awgn(signal, snr_db=10, rng=rng)

rng = np.random.default_rng(seed=42)
result2 = Channels.awgn(signal, snr_db=10, rng=rng)

np.allclose(result1, result2)  # → True
```

---

## Parameter Cheat Sheet

| Modulation | Bits/Symbol | Range | Use Case |
|------------|-------------|-------|----------|
| BPSK | 1 | Binary | Simple, robust |
| QPSK | 2 | 4 values | Common wireless |
| ASK-2 | 1 | 2 levels | Amplitude only |
| ASK-4 | 2 | 4 levels | Bandwidth efficient |
| PSK-8 | 3 | 8 values | High efficiency |
| OOK | 1 | On/Off | Optical |

| Channel | Parameter | Typical Range | Meaning |
|---------|-----------|---------------|---------|
| BSC | p | 0 to 1 | Bit flip probability |
| BEC | p | 0 to 1 | Erasure probability |
| AWGN | snr_db | -5 to +30 | Signal-to-noise ratio |

## SNR Guidelines

- **-5 to 0 dB**: Very noisy
- **5 to 10 dB**: Moderate noise
- **15 to 20 dB**: Clean
- **25+ dB**: Very clean

---

## Output Types

| Function | Input | Output |
|----------|-------|--------|
| `modulate()` | Array of bits/values | Complex ndarray |
| `demodulate()` | Complex ndarray | Integer ndarray |
| `awgn()` | Real or complex array | Same dtype |
| `bsc()` | Bit array | Same dtype |
| `bec()` | Array | Float array |
| `shannon_entropy()` | List of probabilities | Float (bits) |

---

## Debugging Tips

```python
# Check array shapes
print(symbols.shape)       # (100,)
print(symbols.dtype)       # complex128

# Verify modulation mapping
print(symbols[0])          # First symbol value

# Check channel effects
snr_clean = 20   # dB
snr_noisy = 5    # dB

# Reproducible testing
rng = np.random.default_rng(seed=123)
```

---

## New in 1.0.0

```python
# Generic modulation engine (preferred over the legacy classes above)
from commpy import MPSKModulator, MQAMModulator, MPAMModulator
mod = MQAMModulator(16)                          # Gray-coded, unit average energy
symbols = mod.modulate(bits)
llrs = mod.soft_demodulate(received, noise_var=0.1)

# Channel coding (FEC)
from commpy import CRC, HammingCode, CyclicCode, BCHCode, ReedSolomonCode
from commpy import Trellis, ConvolutionalEncoder, viterbi_decode
from commpy import BlockInterleaver, ConvolutionalInterleaver
CRC.crc32().compute(b'data')
codeword = HammingCode(m=3).encode(message)        # single-error-correcting
codeword = ReedSolomonCode(m=8, k=223).encode(message)  # burst/symbol-error-correcting

# PHY: pulse shaping, equalization, synchronization
from commpy import raised_cosine_filter, root_raised_cosine_filter
from commpy import zf_equalizer, mmse_equalizer
from commpy import gardner_timing_error, estimate_cfo_mth_power, costas_loop_bpsk

# OFDM
from commpy import OFDMModulator, OFDMDemodulator, papr, papr_db, papr_ccdf

# Finite fields (used internally by BCH/Reed-Solomon; usable directly too)
from commpy import PrimeField, GF2m

# Information theory
from commpy import (
    binary_entropy, mutual_information,
    channel_capacity_bsc, channel_capacity_awgn, channel_capacity_dmc,
    huffman_codes, huffman_encode, huffman_decode,
    arithmetic_encode, arithmetic_decode, rate_distortion_binary,
)

# Queuing theory
from commpy import MM1Queue, MM1KQueue, MMcQueue
```

See `docs/API.md` for full signatures and `examples/` for runnable end-to-end scripts.

---

## Imports Summary

```python
# Main classes
from commpy import (
    BPSK_Modulator, QPSK_Modulator,
    ASK_2_Modulator, ASK_4_Modulator,
    PSK_8_Modulator, OOK_Modulator,
    Channels, IQWaveform,
    shannon_entropy, is_prime, modinv,
    PrimeField
)

# Visualization (~30 plot_* / animate_* functions; see API.md#visualization)
from commpy import (
    plot_constellation, plot_eye_diagram, plot_psd,
    plot_tanner_graph, plot_decoder_convergence,
    plot_error_rate_comparison, commpy_style,
)

# NumPy (required for arrays)
import numpy as np

# SciPy (required: FFT for OFDM, solve_toeplitz for MMSE equalization, ...)
import scipy

# Plotting (matplotlib is a required dependency, not an optional one)
import matplotlib.pyplot as plt
```

---

## Resources

- **Full API**: `docs/API.md`
- **Getting Started**: `docs/GETTING_STARTED.md`
- **User Guide**: `docs/USER_GUIDE.md`
- **Contributing**: `CONTRIBUTING.md`

---

*Bookmark this page for quick reference!*

Last Updated: March 2026

