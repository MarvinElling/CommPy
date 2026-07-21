"""Hamming(7,4): encode a message, corrupt one bit, and correct it.

Demonstrates commpy.HammingCode's single-error-correction guarantee.
"""

import numpy as np

from commpy import HammingCode


def main() -> None:
    """Run the Hamming(7,4) encode/corrupt/correct demo."""
    code = HammingCode(m=3)  # Hamming(7, 4)
    print(f'Hamming({code.n}, {code.k}): corrects any single-bit error.')

    message = np.array([1, 0, 1, 1])
    codeword = code.encode(message)
    print(f'message  = {message}')
    print(f'codeword = {codeword}')

    corrupted = codeword.copy()
    flip_pos = 4
    corrupted[flip_pos] ^= 1
    print(f'corrupted (bit {flip_pos} flipped) = {corrupted}')

    decoded, corrected, error_pos = code.decode(corrupted)
    print(f'detected error at 1-indexed position {error_pos}')
    print(f'corrected codeword = {corrected}')
    print(f'decoded message    = {decoded}')
    assert np.array_equal(decoded, message)
    print('OK: original message recovered exactly.')


if __name__ == '__main__':
    main()
