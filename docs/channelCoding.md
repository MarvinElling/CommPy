# channelCoding Module

This page documents the `channelCoding` utilities included in the `commpy` package.

## Channels

The `Channels` class provides simple, commonly used channel models for simulation.

Location: `src/CommPy/channelCoding/channels.py`

Public API
- `Channels.bsc(bits, p, rng=None)`
  - Binary Symmetric Channel: flips each bit independently with probability `p`.
  - Parameters:
    - `bits`: array-like of 0/1 or boolean values.
    - `p`: flip probability in [0, 1].
    - `rng`: optional `np.random.Generator` instance for reproducible randomness.
  - Returns: ndarray of same shape as `bits` with bits flipped where flips occurred.

- `Channels.bec(bits, p, erasure_value=-1, rng=None)`
  - Binary Erasure Channel: erases symbols with probability `p` and marks them with `erasure_value`.
  - Parameters:
    - `bits`: array-like of symbols (commonly 0/1).
    - `p`: erasure probability in [0, 1].
    - `erasure_value`: value to indicate an erasure (default `-1`).
    - `rng`: optional `np.random.Generator`.
  - Returns: ndarray (dtype float) with erased entries set to `erasure_value`.

- `Channels.awgn(x, snr_db, rng=None)`
  - Additive White Gaussian Noise: adds Gaussian noise to `x` to achieve specified SNR in dB.
  - Parameters:
    - `x`: input signal (array-like, real or complex).
    - `snr_db`: desired SNR in dB.
    - `rng`: optional `np.random.Generator`.
  - Returns: noisy signal ndarray with same shape and dtype as `x`.

Examples

```python
from CommPy.channelCoding.channels import Channels
import numpy as np

# BSC example
bits = np.array([0, 1, 1, 0])
noisy = Channels.bsc(bits, p=0.1, rng=np.random.default_rng(0))

# BEC example
erased = Channels.bec(bits, p=0.2, erasure_value=-1, rng=np.random.default_rng(1))

# AWGN example
signal = np.array([1.0, -1.0, 1.0])
noisy_signal = Channels.awgn(signal, snr_db=10.0, rng=np.random.default_rng(2))
```

Notes
- The AWGN function measures average signal power as mean(|x|^2). For zero-power inputs, a tiny floor is used to avoid division by zero.
- The BSC preserves boolean dtype when possible; numeric arrays are returned with the same numeric type.
