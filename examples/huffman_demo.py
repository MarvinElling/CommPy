"""Huffman coding: build a code, compress a message, verify against the entropy bound.

Demonstrates commpy.huffman_codes/huffman_encode/huffman_decode.
"""

from collections import Counter

from commpy import huffman_codes, huffman_decode, huffman_encode, shannon_entropy


def main() -> None:
    """Run the Huffman coding demo."""
    text = 'this is an example of a huffman tree'
    counts = Counter(text)
    total = sum(counts.values())
    probabilities = {ch: n / total for ch, n in counts.items()}

    codes = huffman_codes(probabilities)
    print('Symbol probabilities and codes:')
    for ch, code in sorted(codes.items(), key=lambda kv: len(kv[1])):
        label = repr(ch)
        print(f'  {label:>6}  p={probabilities[ch]:.3f}  code={code}')

    encoded = huffman_encode(list(text), codes)
    decoded = huffman_decode(encoded, codes)
    assert ''.join(decoded) == text
    print(f'\nOriginal: {len(text) * 8} bits (8 bits/char)')
    print(f'Huffman:  {len(encoded)} bits ({len(encoded) / len(text):.2f} bits/char)')
    print(f'Entropy bound: {shannon_entropy(list(probabilities.values())):.2f} bits/char')
    print('OK: decoded text matches the original exactly.')


if __name__ == '__main__':
    main()
