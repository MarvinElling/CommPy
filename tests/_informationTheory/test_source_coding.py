"""Tests for commpy source coding: Huffman coding and arithmetic coding."""

import pytest

from commpy import (
    arithmetic_decode,
    arithmetic_encode,
    huffman_codes,
    huffman_decode,
    huffman_encode,
    shannon_entropy,
)


def test_huffman_is_prefix_free():
    codes = huffman_codes({'a': 0.4, 'b': 0.3, 'c': 0.2, 'd': 0.1})
    values = list(codes.values())
    for i, c1 in enumerate(values):
        for c2 in values[i + 1:]:
            assert not c1.startswith(c2)
            assert not c2.startswith(c1)


def test_huffman_more_probable_symbols_get_shorter_or_equal_codes():
    codes = huffman_codes({'a': 0.5, 'b': 0.25, 'c': 0.125, 'd': 0.125})
    assert len(codes['a']) <= len(codes['b']) <= len(codes['c'])
    assert len(codes['c']) == len(codes['d'])


def test_huffman_single_symbol_alphabet():
    codes = huffman_codes({'only': 1.0})
    assert codes == {'only': '0'}
    encoded = huffman_encode(['only', 'only', 'only'], codes)
    assert huffman_decode(encoded, codes) == ['only', 'only', 'only']


def test_huffman_round_trip(rng):
    probs = {'a': 0.4, 'b': 0.3, 'c': 0.2, 'd': 0.1}
    codes = huffman_codes(probs)
    symbols = list(rng.choice(list(probs.keys()), size=200, p=list(probs.values())))
    encoded = huffman_encode(symbols, codes)
    decoded = huffman_decode(encoded, codes)
    assert decoded == symbols


def test_huffman_rejects_empty_probabilities():
    with pytest.raises(ValueError, match='empty'):
        huffman_codes({})


def test_huffman_known_distribution_expected_length_near_entropy():
    # A classic textbook example (probs a power-of-two friendly distribution
    # so Huffman achieves the entropy bound exactly).
    probs = {'a': 0.5, 'b': 0.25, 'c': 0.125, 'd': 0.125}
    codes = huffman_codes(probs)
    expected_len = sum(probs[s] * len(c) for s, c in codes.items())
    assert expected_len == pytest.approx(shannon_entropy(list(probs.values())))


def test_huffman_decode_empty_bitstring_is_empty_sequence():
    assert huffman_decode('', {'a': '0', 'b': '10'}) == []


def test_huffman_decode_rejects_invalid_bitstring():
    with pytest.raises(ValueError, match='invalid'):
        huffman_decode('11', {'a': '0', 'b': '10'})


def test_arithmetic_round_trip_basic():
    probs = {'a': 0.5, 'b': 0.3, 'c': 0.2}
    symbols = ['a', 'b', 'c', 'a', 'a', 'b', 'c', 'c', 'a', 'b']
    code, n_bits = arithmetic_encode(symbols, probs)
    decoded = arithmetic_decode(code, n_bits, len(symbols), probs)
    assert decoded == symbols


def test_arithmetic_round_trip_random(rng):
    probs = {'a': 0.4, 'b': 0.3, 'c': 0.2, 'd': 0.1}
    for _ in range(30):
        n = int(rng.integers(1, 80))
        symbols = list(rng.choice(list(probs.keys()), size=n, p=list(probs.values())))
        code, n_bits = arithmetic_encode(symbols, probs)
        decoded = arithmetic_decode(code, n_bits, len(symbols), probs)
        assert decoded == symbols


def test_arithmetic_highly_skewed_probabilities():
    probs = {'common': 0.999, 'rare': 0.001}
    symbols = ['common'] * 40 + ['rare'] + ['common'] * 10
    code, n_bits = arithmetic_encode(symbols, probs)
    decoded = arithmetic_decode(code, n_bits, len(symbols), probs)
    assert decoded == symbols


def test_arithmetic_single_symbol_alphabet():
    probs = {'only': 1.0}
    symbols = ['only', 'only', 'only']
    code, n_bits = arithmetic_encode(symbols, probs)
    decoded = arithmetic_decode(code, n_bits, len(symbols), probs)
    assert decoded == symbols


def test_arithmetic_empty_sequence():
    probs = {'a': 0.5, 'b': 0.5}
    code, n_bits = arithmetic_encode([], probs)
    decoded = arithmetic_decode(code, n_bits, 0, probs)
    assert decoded == []


def test_arithmetic_rejects_zero_probability_symbol():
    probs = {'a': 1.0, 'b': 0.0}
    with pytest.raises(ValueError, match='zero probability'):
        arithmetic_encode(['a', 'b'], probs)


def test_arithmetic_achieves_close_to_entropy_bound(rng):
    # Compression ratio (bits/symbol) should be close to the source entropy
    # for a long-enough sequence, confirming the coder isn't leaving a lot
    # of compression on the table.
    probs = {'a': 0.5, 'b': 0.25, 'c': 0.125, 'd': 0.125}
    n = 500
    symbols = list(rng.choice(list(probs.keys()), size=n, p=list(probs.values())))
    _code, n_bits = arithmetic_encode(symbols, probs)
    bits_per_symbol = n_bits / n
    entropy = shannon_entropy(list(probs.values()))
    assert bits_per_symbol < entropy + 0.1
