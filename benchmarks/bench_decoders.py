"""Micro-benchmarks for CommPy's hot decoding/modulation paths (pytest-benchmark).

Not part of the correctness suite (`testpaths = ["tests"]`); run explicitly with:

    pytest benchmarks/ --benchmark-only --no-cov

Each benchmark builds a representative received word once, then times the decode.
"""

import numpy as np
import pytest

from commpy import (
    Channels,
    ConvolutionalEncoder,
    LDPCCode,
    MPSKModulator,
    MQAMModulator,
    PolarCode,
    Trellis,
    TurboCode,
    viterbi_decode,
)


def _bpsk_llr(codeword, snr_db, rng):
    mod = MPSKModulator(2)
    received = Channels.awgn(mod.modulate(codeword), snr_db, rng=rng)
    return mod.soft_demodulate(received, 10.0 ** (-snr_db / 10.0))


def test_ldpc_belief_propagation(benchmark):
    rng = np.random.default_rng(0)
    code = LDPCCode.from_gallager(n=192, w_c=3, w_r=6, rng=rng)
    llr = _bpsk_llr(code.encode(rng.integers(0, 2, code.k).astype(np.uint8)), 2.0, rng)
    benchmark(lambda: code.decode(llr, method='min-sum'))


def test_polar_scl_decode(benchmark):
    rng = np.random.default_rng(0)
    code = PolarCode(block_length=256, k=128, design_snr_db=2.0)
    llr = _bpsk_llr(code.encode(rng.integers(0, 2, code.k).astype(np.uint8)), 2.0, rng)
    benchmark(lambda: code.decode(llr, list_size=8))


def test_turbo_iterative_decode(benchmark):
    rng = np.random.default_rng(0)
    code = TurboCode(k=256, rng=rng)
    llr = _bpsk_llr(code.encode(rng.integers(0, 2, code.k).astype(np.uint8)), 1.0, rng)
    benchmark(lambda: code.decode(llr, iterations=6))


def test_viterbi_decode(benchmark):
    rng = np.random.default_rng(0)
    trellis = Trellis(constraint_length=7, generators=(0o171, 0o133))
    encoder = ConvolutionalEncoder(trellis)
    codeword, _ = encoder.encode(rng.integers(0, 2, 400), terminate=True)
    benchmark(lambda: viterbi_decode(trellis, codeword, mode='hard', terminated=True))


@pytest.mark.parametrize('order', [16, 256])
def test_qam_modulate_demodulate(benchmark, order):
    rng = np.random.default_rng(0)
    mod = MQAMModulator(order)
    bits = rng.integers(0, 2, mod.bits_per_symbol * 10_000)
    benchmark(lambda: mod.demodulate(mod.modulate(bits)))
