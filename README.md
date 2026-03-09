# CommPy

**CommPy** is a Python library for communication engineering. It provides tools for modulation, channel modeling, information theory calculations, and waveform generation.

## Overview

CommPy includes implementations of:
- **Modulation schemes**: BPSK, QPSK, ASK-2/4, PSK-8, OOK
- **Channel models**: Binary Symmetric Channel (BSC), Binary Erasure Channel (BEC), Additive White Gaussian Noise (AWGN)
- **Information theory**: Shannon entropy calculations
- **Waveform generation**: IQ modulated waveforms with customizable pulse shaping
- **Utility functions**: Prime field operations, mathematical utilities

## Features

✨ **Easy-to-use API** for modulation and demodulation
🔊 **Channel simulation** with controllable noise and error parameters  
📊 **Waveform generation** with flexible pulse shaping  
🧮 **Mathematical utilities** for communication theory  
🔒 **Full type hints** for better IDE support and code reliability

## Installation

```bash
pip install commpy
```

Or install from source:

```bash
git clone <repository-url>
cd CommPy
pip install -e .
```

## Quick Start

### Modulation

Modulate binary data using various schemes:

```python
from commpy import BPSK_Modulator, QPSK_Modulator

# BPSK modulation
bits = [0, 1, 0, 1, 1, 0]
symbols = BPSK_Modulator.modulate(bits)
demodulated = BPSK_Modulator.demodulate(symbols)

# QPSK modulation
symbols_qpsk = QPSK_Modulator.modulate(bits)
```

### Channel Simulation

Simulate signals over wireless channels:

```python
from commpy import Channels
import numpy as np

# Create a signal
signal = np.array([1+1j, -1-1j, 1+1j])

# Pass through AWGN channel at 10 dB SNR
noisy_signal = Channels.awgn(signal, snr_db=10)
```

### Information Theory

Calculate Shannon entropy:

```python
from commpy import shannon_entropy

probabilities = [0.25, 0.25, 0.5]
entropy = shannon_entropy(probabilities)
print(f"Shannon entropy: {entropy:.2f} bits")
```

### IQ Waveform Generation

Generate IQ modulated RF waveforms:

```python
from commpy import IQWaveform
import numpy as np

# Symbol sequences
I_symbols = np.array([1, 0, -1, 0])
Q_symbols = np.array([0, 1, 0, -1])

# Create waveform
waveform = IQWaveform(
    I=I_symbols,
    Q=Q_symbols,
    T=1e-3,           # Symbol period: 1 ms
    fs=1e6,           # Sample rate: 1 MHz
    f0=100e3          # Carrier frequency: 100 kHz
)

# Plot and analyze
waveform.plot_waveform()
```

## Module Structure

```
commpy/
├── _channelCoding/          # Modulation and channel models
│   ├── modulation_analogCarrier_digitalData.py
│   ├── channels.py
│   └── fields.py
├── _informationTheory/      # Information theory calculations
│   └── formulas.py
├── _waves/                  # Waveform generation
│   └── iq_wave.py
├── _utils/                  # Utility functions
│   └── maths.py
└── __init__.py             # Package entry point
```

## API Reference

### Modulation Classes

#### `BPSK_Modulator`
Binary Phase Shift Keying - maps 0→-1, 1→+1

```python
# Modulate
symbols = BPSK_Modulator.modulate(bits)

# Demodulate
bits = BPSK_Modulator.demodulate(symbols)
```

#### `QPSK_Modulator`
Quadrature Phase Shift Keying - 4-symbol constellation

#### `ASK_2_Modulator` & `ASK_4_Modulator`
Amplitude Shift Keying with 2 or 4 amplitude levels

#### `PSK_8_Modulator`
8-PSK modulation with 8-symbol constellation

#### `OOK_Modulator`
On-Off Keying - simple amplitude modulation

### Channel Models

#### `Channels.bsc(bits, p, rng=None)`
Binary Symmetric Channel - flips each bit with probability `p`

**Parameters:**
- `bits`: Array of bits (0/1 or boolean)
- `p`: Bit flip probability [0, 1]
- `rng`: Optional `np.random.Generator` for reproducibility

#### `Channels.bec(bits, p, erasure_value=-1, rng=None)`
Binary Erasure Channel - erases symbols with probability `p`

**Parameters:**
- `bits`: Array of symbols
- `p`: Erasure probability [0, 1]
- `erasure_value`: Symbol value to indicate erasure

#### `Channels.awgn(x, snr_db, rng=None)`
Additive White Gaussian Noise - adds noise to achieve target SNR

**Parameters:**
- `x`: Input signal (real or complex)
- `snr_db`: Target signal-to-noise ratio in dB
- `rng`: Optional `np.random.Generator` for reproducibility

### Information Theory

#### `shannon_entropy(probabilities)`
Calculate Shannon entropy of a probability distribution

**Parameters:**
- `probabilities`: List of probabilities (must sum to 1)

**Returns:** Shannon entropy in bits

### Waveform Generation

#### `IQWaveform(I, Q, T, fs, f0=0, pulse_shape=None, span=4)`
Generate IQ modulated waveforms

**Parameters:**
- `I, Q`: Arrays of I/Q symbols
- `T`: Symbol period (seconds)
- `fs`: Sample rate (Hz)
- `f0`: Carrier frequency (Hz), 0 for baseband
- `pulse_shape`: Callable defining pulse shape (default: rectangular)
- `span`: Pulse truncation window in multiples of T

**Attributes:**
- `.t`: Time vector
- `.s`: Baseband complex signal or bandpass real signal
- `.s_I`, `.s_Q`: I and Q components

**Methods:**
- `.plot_waveform()`: Visualize the generated waveform

### Utility Functions

#### `is_prime(n)`
Check if a number is prime

#### `modinv(a, m)`
Compute modular multiplicative inverse of `a` modulo `m`

#### `PrimeField`
Finite field arithmetic over prime modulus

## Examples

### Example 1: Modulate and Demodulate with AWGN

```python
import numpy as np
from commpy import BPSK_Modulator, Channels

# Generate random bits
bits = np.random.randint(0, 2, 100)

# Modulate
symbols = BPSK_Modulator.modulate(bits)

# Pass through AWGN channel
noisy = Channels.awgn(symbols, snr_db=5)

# Demodulate
recovered = BPSK_Modulator.demodulate(noisy)

# Check error rate
errors = np.sum(recovered != bits)
print(f"Bit errors: {errors}/100")
```

### Example 2: Multi-Symbol Modulation

```python
from commpy import QPSK_Modulator, ASK_4_Modulator

# QPSK has 4 symbols per 2 bits
pairs = [0, 1, 2, 3]  # 2-bit values
qpsk_symbols = QPSK_Modulator.modulate(pairs)

# ASK-4 also has 4 levels
ask_symbols = ASK_4_Modulator.modulate(pairs)
```

### Example 3: Channel Comparison

```python
from commpy import Channels
import numpy as np

bits = np.array([1, 0, 1, 1, 0])

# BSC with 10% error rate
bsc_out = Channels.bsc(bits, p=0.1)

# BEC with 10% erasure rate
bec_out = Channels.bec(bits, p=0.1)

print("Original:", bits)
print("BSC output:", bsc_out)
print("BEC output:", bec_out)
```

## Requirements

- Python ≥ 3.10
- NumPy
- Matplotlib (for plotting)

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) file for details.

## Author

Marvin Elling

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

