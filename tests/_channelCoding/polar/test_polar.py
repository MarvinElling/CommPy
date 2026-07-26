"""Tests for PolarCode: SC / SCL / CRC-aided SCL decoding and coded BER.

The centerpiece is a brute-force maximum-likelihood cross-validation: with a
list large enough that pruning never discards the winner, SCL must return the
exact ML codeword. Because the decoder accumulates the *exact* log-likelihood
path metric, this is an equality check against exhaustive search over all
messages -- the same gold-standard style used for the Viterbi and algebraic
decoders elsewhere in the suite. Round-trip, CRC-selection, argument-validation,
and statistical coding-gain checks round it out.
"""

import numpy as np
import pytest

from commpy import CRC, Channels, MPSKModulator, PolarCode, simulate_coded_ber
from commpy._channelCoding.polar.decoder import polar_transform


def _ml_message(rx_llr, code):
    """Exhaustive ML message under the exact LLR path metric (min negative log-likelihood)."""
    best_message, best_metric = None, None
    for message_int in range(1 << code.k):
        message = np.array(
            [(message_int >> i) & 1 for i in range(code.k - 1, -1, -1)], dtype=np.uint8,
        )
        u = np.zeros(code.n, dtype=np.uint8)
        u[code.info_positions] = message
        codeword = polar_transform(u)
        metric = float(np.sum(np.logaddexp(0.0, -(1 - 2 * codeword.astype(float)) * rx_llr)))
        if best_metric is None or metric < best_metric:
            best_metric, best_message = metric, message
    return best_message


@pytest.mark.parametrize('list_size', [1, 4])
def test_clean_round_trip(list_size, rng):
    code = PolarCode(block_length=64, k=32, design_snr_db=2.0)
    for _ in range(10):
        message = rng.integers(0, 2, code.k).astype(np.uint8)
        codeword = code.encode(message)
        assert codeword.shape == (code.n,)
        llr = np.where(codeword == 0, 10.0, -10.0)
        message_hat, codeword_hat, _ = code.decode(llr, list_size=list_size)
        np.testing.assert_array_equal(message_hat, message)
        np.testing.assert_array_equal(codeword_hat, codeword)


def test_full_list_scl_matches_brute_force_ml(rng):
    code = PolarCode(block_length=8, k=3, construction='bhattacharyya', design_snr_db=0.0)
    mod = MPSKModulator(2)
    for _ in range(40):
        message = rng.integers(0, 2, code.k).astype(np.uint8)
        received = Channels.awgn(mod.modulate(code.encode(message)), 1.0, rng=rng)
        llr = mod.soft_demodulate(received, 10.0 ** (-1.0 / 10.0))
        full_list = code.decode(llr, list_size=1 << code.k)[0]  # no pruning -> ML
        np.testing.assert_array_equal(full_list, _ml_message(llr, code))


def test_crc_aided_scl_round_trips_and_selects_by_crc(rng):
    crc = CRC.crc8()
    code = PolarCode(block_length=128, k=32, crc=crc, design_snr_db=2.0)
    assert code.info_positions.size == code.k + crc.config.width
    for _ in range(10):
        message = rng.integers(0, 2, code.k).astype(np.uint8)
        codeword = code.encode(message)
        llr = np.where(codeword == 0, 10.0, -10.0)
        message_hat, _, rank = code.decode(llr, list_size=8)
        np.testing.assert_array_equal(message_hat, message)
        assert rank == 0  # the correct path also has the best metric here


def test_argument_validation():
    code = PolarCode(block_length=16, k=8)
    with pytest.raises(ValueError, match='message must have length'):
        code.encode(np.zeros(code.k + 1, dtype=np.uint8))
    with pytest.raises(ValueError, match='length'):
        code.decode(np.zeros(code.n + 1))
    with pytest.raises(ValueError, match='list_size'):
        code.decode(np.zeros(code.n), list_size=0)
    with pytest.raises(ValueError, match='power of two'):
        PolarCode(block_length=12, k=6)
    with pytest.raises(ValueError, match='exceeds block length'):
        PolarCode(block_length=16, k=12, crc=CRC.crc8())  # 12 + 8 > 16
    with pytest.raises(ValueError, match='k must be'):
        PolarCode(block_length=16, k=0)


@pytest.mark.parametrize('list_size', [1, 4])
def test_coded_ber_beats_uncoded(list_size, rng):
    code = PolarCode(block_length=64, k=32, design_snr_db=2.5)
    mod = MPSKModulator(2)
    snr_db = 2.5
    noise_var = 10.0 ** (-snr_db / 10.0)

    coded_errors = uncoded_errors = total_bits = 0
    for _ in range(40):
        message = rng.integers(0, 2, code.k).astype(np.uint8)
        received = Channels.awgn(mod.modulate(code.encode(message)), snr_db, rng=rng)
        llr = mod.soft_demodulate(received, noise_var)
        coded_errors += int(np.sum(code.decode(llr, list_size=list_size)[0] != message))

        received_uncoded = Channels.awgn(mod.modulate(message), snr_db, rng=rng)
        uncoded_errors += int(np.sum(mod.demodulate(received_uncoded) != message))
        total_bits += code.k

    assert coded_errors / total_bits < uncoded_errors / total_bits


def test_simulate_coded_ber_runs_and_decreases(rng):
    code = PolarCode(block_length=64, k=32, design_snr_db=2.0)
    result = simulate_coded_ber(
        code, MPSKModulator(2), Channels.awgn, [-3.0, 3.0],
        blocks_per_batch=10, target_errors=20, max_trials=6_000, rng=rng,
    )
    assert result.error_rate.shape == (2,)
    assert np.all(np.isfinite(result.error_rate))
    assert result.error_rate[1] < result.error_rate[0]
