"""Tests for commpy.viterbi_decode.

Correctness leans on cross-validating Viterbi's output against brute-force
maximum-likelihood search over every possible message on small trellises --
the gold-standard check for trellis decoders -- rather than hand-verifying
the add-compare-select recursion by inspection.
"""

import numpy as np
import pytest

from commpy import Channels, ConvolutionalEncoder, MPSKModulator, Trellis, viterbi_decode


def _brute_force_hard(encoder, received, n_msg_bits):
    best_msg, best_dist = None, None
    for msg_int in range(1 << n_msg_bits):
        msg = [(msg_int >> i) & 1 for i in range(n_msg_bits - 1, -1, -1)]
        codeword, _ = encoder.encode(msg, terminate=True)
        dist = int(np.sum(codeword != received))
        if best_dist is None or dist < best_dist:
            best_dist, best_msg = dist, msg
    return np.array(best_msg)


def _brute_force_soft(encoder, llrs, n_msg_bits):
    best_msg, best_metric = None, None
    for msg_int in range(1 << n_msg_bits):
        msg = [(msg_int >> i) & 1 for i in range(n_msg_bits - 1, -1, -1)]
        codeword, _ = encoder.encode(msg, terminate=True)
        metric = float(np.sum((2 * codeword - 1) * llrs))
        if best_metric is None or metric < best_metric:
            best_metric, best_msg = metric, msg
    return np.array(best_msg)


@pytest.mark.parametrize(
    ('constraint_length', 'generators'), [(3, (0b111, 0b101)), (3, (0b110, 0b101, 0b111))],
)
def test_hard_decode_matches_brute_force_ml_no_noise(constraint_length, generators, rng):
    trellis = Trellis(constraint_length, generators)
    encoder = ConvolutionalEncoder(trellis)
    for _ in range(10):
        msg = rng.integers(0, 2, 8)
        codeword, _ = encoder.encode(msg, terminate=True)
        decoded = viterbi_decode(trellis, codeword, mode='hard', terminated=True)
        np.testing.assert_array_equal(decoded, msg)


@pytest.mark.parametrize(
    ('constraint_length', 'generators'), [(3, (0b111, 0b101)), (4, (0b1101, 0b1011))],
)
def test_hard_decode_matches_brute_force_ml_with_errors(constraint_length, generators, rng):
    trellis = Trellis(constraint_length, generators)
    encoder = ConvolutionalEncoder(trellis)
    n_msg_bits = 6  # small enough for exhaustive 2**6 = 64-candidate brute force

    for _ in range(15):
        msg = rng.integers(0, 2, n_msg_bits)
        codeword, _ = encoder.encode(msg, terminate=True)
        received = codeword.copy()
        n_flips = int(rng.integers(0, 3))
        flip_positions = rng.choice(len(received), size=n_flips, replace=False)
        received[flip_positions] ^= 1

        viterbi_result = viterbi_decode(trellis, received, mode='hard', terminated=True)
        brute_force_result = _brute_force_hard(encoder, received, n_msg_bits)
        np.testing.assert_array_equal(viterbi_result, brute_force_result)


def test_soft_decode_matches_brute_force_ml(rng):
    trellis = Trellis(3, (0b111, 0b101))
    encoder = ConvolutionalEncoder(trellis)
    n_msg_bits = 6

    for _ in range(15):
        msg = rng.integers(0, 2, n_msg_bits)
        codeword, _ = encoder.encode(msg, terminate=True)
        # Noisy LLRs, correlated with the true bits but not deterministic:
        # bit=0 -> LLR tends positive, bit=1 -> LLR tends negative.
        ideal_sign = 1 - 2 * codeword
        llrs = ideal_sign * (0.5 + rng.random(len(codeword))) + rng.normal(
            scale=0.3, size=len(codeword),
        )

        viterbi_result = viterbi_decode(trellis, llrs, mode='soft', terminated=True)
        brute_force_result = _brute_force_soft(encoder, llrs, n_msg_bits)
        np.testing.assert_array_equal(viterbi_result, brute_force_result)


def test_unterminated_decoding_recovers_message(rng):
    trellis = Trellis(3, (0b111, 0b101))
    encoder = ConvolutionalEncoder(trellis)
    msg = rng.integers(0, 2, 10)
    codeword, _ = encoder.encode(msg, terminate=False)
    decoded = viterbi_decode(trellis, codeword, mode='hard', terminated=False)
    np.testing.assert_array_equal(decoded, msg)


def test_invalid_mode_raises():
    trellis = Trellis(3, (0b111, 0b101))
    with pytest.raises(ValueError, match='mode'):
        viterbi_decode(trellis, [0, 0, 0, 0], mode='bogus')


def test_received_length_not_multiple_of_n_outputs_raises():
    trellis = Trellis(3, (0b111, 0b101))
    with pytest.raises(ValueError, match='multiple'):
        viterbi_decode(trellis, [0, 0, 0], mode='hard')


def test_ber_improves_with_coding_over_uncoded_bpsk(rng):
    """Sanity check: coded BER at a given SNR should beat uncoded BER (coding gain)."""
    trellis = Trellis(7, (0o171, 0o133))
    encoder = ConvolutionalEncoder(trellis)
    mod = MPSKModulator(2)
    snr_db = 2.0
    n_blocks = 40
    msg_len = 50

    coded_errors = 0
    uncoded_errors = 0
    total_bits = 0
    for _ in range(n_blocks):
        msg = rng.integers(0, 2, msg_len)
        codeword, _ = encoder.encode(msg, terminate=True)

        coded_symbols = mod.modulate(codeword)
        received_coded = Channels.awgn(coded_symbols, snr_db, rng=rng)
        llrs = mod.soft_demodulate(received_coded, noise_var=1.0)
        decoded = viterbi_decode(trellis, llrs, mode='soft', terminated=True)
        coded_errors += int(np.sum(decoded != msg))

        uncoded_symbols = mod.modulate(msg)
        received_uncoded = Channels.awgn(uncoded_symbols, snr_db, rng=rng)
        uncoded_bits = mod.demodulate(received_uncoded)
        uncoded_errors += int(np.sum(uncoded_bits != msg))

        total_bits += msg_len

    coded_ber = coded_errors / total_bits
    uncoded_ber = uncoded_errors / total_bits
    assert coded_ber < uncoded_ber
