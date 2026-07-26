"""Tests for the LDPCCode class, belief-propagation decoding, and coded BER.

Beyond structural round-trip checks (encode is systematic, a clean/confident
LLR vector decodes back to the message in one iteration), the substantive check
is statistical: over an AWGN channel the LDPC-coded bit-error rate must beat the
uncoded rate at the same SNR (coding gain), for both the sum-product and
min-sum decoders. This mirrors the BER-vs-theory style used elsewhere in the
suite rather than asserting exact per-block corrections BP does not guarantee.
"""

import numpy as np
import pytest

from commpy import Channels, LDPCCode, MPSKModulator, simulate_coded_ber
from commpy._channelCoding.ldpc.standards import rate_one_half_ldpc


@pytest.fixture
def code(rng):
    return LDPCCode.from_gallager(n=48, w_c=3, w_r=6, rng=rng)


def test_encode_produces_valid_codeword_and_is_systematic(code, rng):
    for _ in range(10):
        message = rng.integers(0, 2, code.k).astype(np.uint8)
        codeword = code.encode(message)
        assert codeword.shape == (code.n,)
        assert np.all((code.H @ codeword) % 2 == 0)  # it is a codeword
        np.testing.assert_array_equal(code.extract_message(codeword), message)  # systematic


def test_encode_wrong_length_raises(code):
    with pytest.raises(ValueError, match='length'):
        code.encode(np.zeros(code.k + 1, dtype=np.uint8))


@pytest.mark.parametrize('method', ['sum-product', 'min-sum'])
def test_decode_clean_confident_llr_recovers_message(code, method, rng):
    for _ in range(10):
        message = rng.integers(0, 2, code.k).astype(np.uint8)
        codeword = code.encode(message)
        llr = np.where(codeword == 0, 12.0, -12.0)  # confident + correct
        message_hat, codeword_hat, iterations = code.decode(llr, method=method)
        np.testing.assert_array_equal(message_hat, message)
        np.testing.assert_array_equal(codeword_hat, codeword)
        assert iterations == 1  # already a valid codeword


def test_decode_rejects_bad_arguments(code):
    with pytest.raises(ValueError, match='length'):
        code.decode(np.zeros(code.n + 1))
    with pytest.raises(ValueError, match='method'):
        code.decode(np.zeros(code.n), method='bogus')
    with pytest.raises(ValueError, match='max_iter'):
        code.decode(np.zeros(code.n), max_iter=0)


@pytest.mark.parametrize('method', ['sum-product', 'min-sum'])
def test_coded_ber_beats_uncoded(code, method, rng):
    mod = MPSKModulator(2)
    snr_db = 3.0
    noise_var = 10.0 ** (-snr_db / 10.0)

    coded_errors = uncoded_errors = total_bits = 0
    for _ in range(60):
        message = rng.integers(0, 2, code.k).astype(np.uint8)
        codeword = code.encode(message)

        received = Channels.awgn(mod.modulate(codeword), snr_db, rng=rng)
        llr = mod.soft_demodulate(received, noise_var)
        message_hat, _, _ = code.decode(llr, method=method)
        coded_errors += int(np.sum(message_hat != message))

        received_uncoded = Channels.awgn(mod.modulate(message), snr_db, rng=rng)
        uncoded_errors += int(np.sum(mod.demodulate(received_uncoded) != message))
        total_bits += code.k

    assert coded_errors / total_bits < uncoded_errors / total_bits


def test_high_snr_is_error_free(code, rng):
    mod = MPSKModulator(2)
    snr_db = 7.0
    noise_var = 10.0 ** (-snr_db / 10.0)
    errors = 0
    for _ in range(40):
        message = rng.integers(0, 2, code.k).astype(np.uint8)
        received = Channels.awgn(mod.modulate(code.encode(message)), snr_db, rng=rng)
        llr = mod.soft_demodulate(received, noise_var)
        errors += int(np.sum(code.decode(llr)[0] != message))
    assert errors == 0


def test_qc_standard_code_round_trips(rng):
    qc = rate_one_half_ldpc(z=8)
    assert qc.n == 48
    assert 0 < qc.rate < 1
    message = rng.integers(0, 2, qc.k).astype(np.uint8)
    codeword = qc.encode(message)
    assert np.all((qc.H @ codeword) % 2 == 0)
    llr = np.where(codeword == 0, 12.0, -12.0)
    np.testing.assert_array_equal(qc.decode(llr)[0], message)


def test_rejects_non_2d_matrix():
    with pytest.raises(ValueError, match='2-D'):
        LDPCCode(np.zeros(5, dtype=np.uint8))


def test_simulate_coded_ber_runs_and_decreases(code, rng):
    result = simulate_coded_ber(
        code, MPSKModulator(2), Channels.awgn, [1.0, 5.0],
        blocks_per_batch=20, target_errors=20, max_trials=40_000, rng=rng,
    )
    assert result.error_rate.shape == (2,)
    assert np.all(np.isfinite(result.error_rate))
    # Error rate must fall as SNR rises.
    assert result.error_rate[1] < result.error_rate[0]
