"""Lossless source coding: Huffman coding and arithmetic coding.

The arithmetic coder uses exact `fractions.Fraction` interval arithmetic
rather than the classic fixed-precision-register formulation with periodic
renormalization and carry propagation. This sacrifices some raw throughput
on very long sequences (Fraction sizes grow with sequence length) in
exchange for being straightforwardly, obviously correct -- appropriate for
a library where source coding is a supporting feature, not a bulk-data
compressor optimized for gigabyte-scale streams.
"""

import heapq
import math
from collections.abc import Hashable, Mapping, Sequence
from fractions import Fraction
from itertools import count
from typing import Any, TypeVar

# `Mapping`'s key type is invariant in typeshed, so a bare `Hashable` key type
# would reject e.g. `dict[str, float]` (not a subtype of `Mapping[Hashable, float]`
# under invariance). A bound TypeVar instead makes these functions generic over
# the caller's actual symbol type, which both fixes that and gives more precise
# return types (e.g. `dict[str, str]` in, `dict[str, str]` out).
Symbol = TypeVar('Symbol', bound=Hashable)


def huffman_codes(probabilities: Mapping[Symbol, float]) -> dict[Symbol, str]:
    """Build a Huffman prefix code (symbol -> bitstring) minimizing expected code length.

    Args:
        probabilities: Symbol probabilities (need not be pre-normalized to
            sum exactly to 1; only relative weights matter for the tree
            structure).

    Returns:
        Mapping from each symbol to its binary codeword.

    Raises:
        ValueError: If `probabilities` is empty.
    """
    if not probabilities:
        msg = 'probabilities must not be empty.'
        raise ValueError(msg)
    if len(probabilities) == 1:
        (only_symbol,) = probabilities.keys()
        return {only_symbol: '0'}

    tie_breaker = count()
    # Heap entries: [combined_weight, tie_breaker_id, [[symbol, code], ...]].
    # tie_breaker is a unique int so heapq never needs to compare symbols or
    # code-pair lists directly (which may not be orderable). Deliberately
    # untyped (Any): a heterogeneous (float, int, list) heap entry doesn't
    # benefit from a more elaborate structure for this internal detail.
    heap: list[list[Any]] = [
        [weight, next(tie_breaker), [[sym, '']]] for sym, weight in probabilities.items()
    ]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[2]:
            pair[1] = '0' + pair[1]
        for pair in hi[2]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0], next(tie_breaker), lo[2] + hi[2]])

    return dict(heap[0][2])


def huffman_encode(symbols: Sequence[Symbol], codes: Mapping[Symbol, str]) -> str:
    """Encode a symbol sequence into a bitstring using a prebuilt Huffman code."""
    return ''.join(codes[s] for s in symbols)


def huffman_decode(encoded: str, codes: Mapping[Symbol, str]) -> list[Symbol]:
    """Decode a Huffman-coded bitstring back into its symbol sequence.

    Raises:
        ValueError: If `encoded` is not a valid concatenation of codewords.
    """
    reverse = {code: sym for sym, code in codes.items()}
    max_len = max(len(c) for c in codes.values())
    decoded: list[Symbol] = []
    i = 0
    while i < len(encoded):
        for length in range(1, max_len + 1):
            candidate = encoded[i:i + length]
            if candidate in reverse:
                decoded.append(reverse[candidate])
                i += length
                break
        else:
            msg = f'invalid or incomplete code at bit position {i}.'
            raise ValueError(msg)
    return decoded


def _cumulative_ranges(
    probabilities: Mapping[Symbol, float],
) -> dict[Symbol, tuple[Fraction, Fraction]]:
    """Exact (sum-to-1-guaranteed) cumulative probability ranges for arithmetic coding."""
    symbols = list(probabilities.keys())
    weights = [Fraction(probabilities[s]).limit_denominator(10**9) for s in symbols]
    total = sum(weights)
    if total <= 0:
        msg = 'probabilities must have positive total weight.'
        raise ValueError(msg)
    weights = [w / total for w in weights]  # normalize to sum to exactly 1

    ranges: dict[Symbol, tuple[Fraction, Fraction]] = {}
    acc = Fraction(0)
    for sym, w in zip(symbols, weights, strict=True):
        ranges[sym] = (acc, acc + w)
        acc += w
    return ranges


def arithmetic_encode(
    symbols: Sequence[Symbol], probabilities: Mapping[Symbol, float],
) -> tuple[int, int]:
    """Arithmetic-encode a symbol sequence given a (static) probability model.

    Args:
        symbols: The sequence to encode. `len(symbols)` must be tracked by
            the caller and passed to `arithmetic_decode`.
        probabilities: Symbol probabilities (need not be pre-normalized).

    Returns:
        `(code, n_bits)`: an integer whose `n_bits`-bit binary expansion is
        the shortest binary fraction uniquely identifying the final coding
        interval.

    Raises:
        ValueError: If `probabilities` assigns zero weight to a symbol that
            appears in `symbols`.
    """
    ranges = _cumulative_ranges(probabilities)
    low, high = Fraction(0), Fraction(1)
    for sym in symbols:
        if sym not in ranges or ranges[sym][1] == ranges[sym][0]:
            msg = f'symbol {sym!r} has zero probability under the given model.'
            raise ValueError(msg)
        width = high - low
        c_lo, c_hi = ranges[sym]
        high = low + width * c_hi
        low = low + width * c_lo

    n_bits = 0
    while True:
        scale = 1 << n_bits
        code = math.ceil(low * scale)
        if Fraction(code, scale) < high:
            return code, n_bits
        n_bits += 1


def arithmetic_decode(
    code: int, n_bits: int, n_symbols: int, probabilities: Mapping[Symbol, float],
) -> list[Symbol]:
    """Invert `arithmetic_encode`.

    Args:
        code: The integer code returned by `arithmetic_encode`.
        n_bits: The bit-width returned by `arithmetic_encode`.
        n_symbols: The original sequence length (`len(symbols)`), needed to
            know when to stop decoding.
        probabilities: The same probability model used to encode.

    Returns:
        The decoded symbol sequence.

    Raises:
        ValueError: If decoding fails to match any symbol's range (e.g. a
            mismatched `probabilities` model).
    """
    ranges = _cumulative_ranges(probabilities)
    value = Fraction(code, 1 << n_bits)
    low, high = Fraction(0), Fraction(1)
    decoded: list[Symbol] = []

    for _ in range(n_symbols):
        width = high - low
        target = (value - low) / width
        for sym, (c_lo, c_hi) in ranges.items():
            if c_lo <= target < c_hi:
                decoded.append(sym)
                high = low + width * c_hi
                low = low + width * c_lo
                break
        else:
            msg = 'decoding failed: value does not fall within any symbol range.'
            raise ValueError(msg)

    return decoded


__all__ = [
    'arithmetic_decode',
    'arithmetic_encode',
    'huffman_codes',
    'huffman_decode',
    'huffman_encode',
]
