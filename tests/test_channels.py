import os
import sys
import numpy as np

# Ensure the package in `src/` is importable when running tests from the repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from CommPy.channelCoding.channels import Channels


def test_bsc():
    rng = np.random.default_rng(12345)
    bits = np.zeros(10000, dtype=int)
    out = Channels.bsc(bits, p=0.1, rng=rng)
    flips = np.sum(out != bits)
    frac = flips / len(bits)
    assert abs(frac - 0.1) < 0.01


def test_bec():
    rng = np.random.default_rng(12345)
    bits = np.ones(10000, dtype=int)
    out = Channels.bec(bits, p=0.2, erasure_value=-1, rng=rng)
    erasures = np.sum(out == -1)
    frac = erasures / len(bits)
    assert abs(frac - 0.2) < 0.02


def test_awgn():
    rng = np.random.default_rng(12345)
    x = np.ones(100000)
    snr_db = 10.0
    y = Channels.awgn(x, snr_db, rng=rng)
    noise = y - x
    signal_power = np.mean(x ** 2)
    noise_power = np.mean(noise ** 2)
    snr_est = 10 * np.log10(signal_power / noise_power)
    assert abs(snr_est - snr_db) < 0.25
