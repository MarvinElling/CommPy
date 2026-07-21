"""Reed-Solomon: symbol-error correction, and erasure correction composed with commpy.Channels.bec.

Demonstrates both decode paths of commpy.ReedSolomonCode: `decode` (unknown
error positions, up to t errors) and `decode_erasures` (known erasure
positions, up to n-k erasures -- twice the error-only capability).
"""

import numpy as np

from commpy import Channels, ReedSolomonCode


def main() -> None:
    """Run the Reed-Solomon error- and erasure-correction demos."""
    rng = np.random.default_rng(0)

    # RS(255, 223) over GF(256): the classic CCSDS/DVB code, t = 16.
    code = ReedSolomonCode(m=8, k=223)
    print(f'RS({code.n}, {code.k}) over GF({code.field.order}): t={code.t} errors guaranteed.')

    message = rng.integers(0, code.field.order, code.k)
    codeword = code.encode(message)

    corrupted = codeword.copy()
    error_positions = rng.choice(code.n, size=code.t, replace=False)
    corrupted[error_positions] = rng.integers(1, code.field.order, code.t)
    decoded, _, n_errors = code.decode(corrupted)
    print(f'Injected {code.t} symbol errors at unknown positions -> corrected {n_errors}.')
    assert np.array_equal(decoded, message)
    print('OK: message recovered exactly via error-only decoding.')

    # Erasure correction composed with the BEC channel model: BEC marks
    # erased positions, decode_erasures uses that side information directly.
    # p is chosen so the expected erasure count sits safely under the code's
    # n-k capacity; re-draw if an unlucky sample still exceeds it (rare, but
    # possible for a Bernoulli process -- unlike the fixed-count error demo
    # above, capacity here isn't hardcoded into the draw itself).
    erasure_prob = 0.06
    for _attempt in range(10):
        received = Channels.bec(codeword.astype(float), p=erasure_prob, erasure_value=-1, rng=rng)
        erasure_positions = np.where(received == -1)[0]
        if len(erasure_positions) <= code.n_minus_k:
            break
    print(f'BEC erased {len(erasure_positions)}/{code.n} symbols (capacity: {code.n_minus_k}).')
    decoded_e, _ = code.decode_erasures(received.astype(np.int64), erasure_positions)
    assert np.array_equal(decoded_e, message)
    print('OK: message recovered exactly via erasure decoding.')


if __name__ == '__main__':
    main()
