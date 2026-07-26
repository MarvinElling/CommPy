"""Tests for BCJR component decoding and the full TurboCode.

The substantive checks are statistical, matching the BER-vs-theory style of the
suite: a single BCJR pass over one rate-1/2 RSC must already beat the uncoded
error rate, and the full iterative turbo decoder must beat it by more. Clean
round-trips, extrinsic-LLR behavior, reproducibility of the interleaver, and
argument validation are checked deterministically.
"""

import numpy as np
import pytest

from commpy import Channels, MPSKModulator, TurboCode, simulate_coded_ber
from commpy._channelCoding.turbo.bcjr import bcjr_decode
from commpy._channelCoding.turbo.rsc import RSCTrellis, rsc_encode


def test_bcjr_requires_equal_length_inputs():
    trellis = RSCTrellis(3, 0o7, 0o5)
    with pytest.raises(ValueError, match='equal length'):
        bcjr_decode(trellis, np.zeros(4), np.zeros(4), np.zeros(3))


def test_single_bcjr_pass_beats_uncoded(rng):
    trellis = RSCTrellis(3, 0o7, 0o5)
    mod = MPSKModulator(2)
    k, snr_db = 200, 2.0
    noise_var = 10.0 ** (-snr_db / 10.0)

    coded_errors = uncoded_errors = total = 0
    for _ in range(30):
        message = rng.integers(0, 2, k).astype(np.uint8)
        codeword = np.concatenate([message, rsc_encode(trellis, message)])  # rate-1/2 systematic
        received = Channels.awgn(mod.modulate(codeword), snr_db, rng=rng)
        llr = mod.soft_demodulate(received, noise_var)
        extrinsic = bcjr_decode(trellis, np.zeros(k), llr[:k], llr[k:])
        message_hat = ((llr[:k] + extrinsic) < 0).astype(np.uint8)
        coded_errors += int(np.sum(message_hat != message))

        received_uncoded = Channels.awgn(mod.modulate(message), snr_db, rng=rng)
        uncoded_errors += int(np.sum(mod.demodulate(received_uncoded) != message))
        total += k

    assert coded_errors / total < uncoded_errors / total


def test_turbo_clean_round_trip(rng):
    code = TurboCode(k=128, rng=np.random.default_rng(1))
    assert code.n == 3 * code.k
    for _ in range(5):
        message = rng.integers(0, 2, code.k).astype(np.uint8)
        codeword = code.encode(message)
        assert codeword.shape == (code.n,)
        llr = np.where(codeword == 0, 8.0, -8.0)
        message_hat, _, _ = code.decode(llr, iterations=4)
        np.testing.assert_array_equal(message_hat, message)


def test_turbo_beats_uncoded(rng):
    code = TurboCode(k=128, rng=np.random.default_rng(1))
    mod = MPSKModulator(2)
    snr_db = 1.0
    noise_var = 10.0 ** (-snr_db / 10.0)

    coded_errors = uncoded_errors = total = 0
    for _ in range(20):
        message = rng.integers(0, 2, code.k).astype(np.uint8)
        received = Channels.awgn(mod.modulate(code.encode(message)), snr_db, rng=rng)
        llr = mod.soft_demodulate(received, noise_var)
        coded_errors += int(np.sum(code.decode(llr, iterations=6)[0] != message))

        received_uncoded = Channels.awgn(mod.modulate(message), snr_db, rng=rng)
        uncoded_errors += int(np.sum(mod.demodulate(received_uncoded) != message))
        total += code.k

    assert coded_errors / total < uncoded_errors / total


def test_explicit_interleaver_is_used_and_reproducible():
    perm = np.roll(np.arange(64), 7)
    code = TurboCode(k=64, interleaver=perm)
    np.testing.assert_array_equal(code.interleaver, perm)
    np.testing.assert_array_equal(code.deinterleaver, np.argsort(perm))
    # Two codes with the same explicit interleaver encode identically.
    other = TurboCode(k=64, interleaver=perm)
    message = np.arange(64, dtype=np.uint8) % 2
    np.testing.assert_array_equal(code.encode(message), other.encode(message))


def test_argument_validation():
    code = TurboCode(k=32, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match='message must have length'):
        code.encode(np.zeros(code.k + 1, dtype=np.uint8))
    with pytest.raises(ValueError, match='llr must have length'):
        code.decode(np.zeros(code.n + 1))
    with pytest.raises(ValueError, match='iterations'):
        code.decode(np.zeros(code.n), iterations=0)
    with pytest.raises(ValueError, match='k must be'):
        TurboCode(k=0)
    with pytest.raises(ValueError, match='permutation'):
        TurboCode(k=8, interleaver=[0, 1, 2])


def test_simulate_coded_ber_runs_and_decreases(rng):
    code = TurboCode(k=128, rng=np.random.default_rng(1))
    result = simulate_coded_ber(
        code, MPSKModulator(2), Channels.awgn, [-6.0, -2.0],
        blocks_per_batch=8, target_errors=20, max_trials=3_000, rng=rng,
    )
    assert result.error_rate.shape == (2,)
    assert np.all(np.isfinite(result.error_rate))
    assert result.error_rate[1] < result.error_rate[0]
