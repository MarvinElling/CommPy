"""Convolutional coding + hard- and soft-decision Viterbi decoding over a noisy channel.

Demonstrates the classic (7, 1/2) Voyager/NASA code and compares coded vs.
uncoded BER at a fixed SNR to show the coding gain.
"""

import numpy as np

from commpy import Channels, ConvolutionalEncoder, MPSKModulator, Trellis, viterbi_decode


def main() -> None:
    """Run the convolutional-coding coding-gain demo."""
    rng = np.random.default_rng(0)
    trellis = Trellis(constraint_length=7, generators=(0o171, 0o133))
    encoder = ConvolutionalEncoder(trellis)
    mod = MPSKModulator(2)  # BPSK
    snr_db = 2.0
    msg_len = 200
    n_blocks = 30

    coded_errors = uncoded_errors = total_bits = 0
    for _ in range(n_blocks):
        message = rng.integers(0, 2, msg_len)
        codeword, _ = encoder.encode(message, terminate=True)

        coded_symbols = mod.modulate(codeword)
        received = Channels.awgn(coded_symbols, snr_db, rng=rng)
        llrs = mod.soft_demodulate(received, noise_var=1.0)
        decoded = viterbi_decode(trellis, llrs, mode='soft', terminated=True)
        coded_errors += int(np.sum(decoded != message))

        uncoded_symbols = mod.modulate(message)
        uncoded_received = Channels.awgn(uncoded_symbols, snr_db, rng=rng)
        uncoded_bits = mod.demodulate(uncoded_received)
        uncoded_errors += int(np.sum(uncoded_bits != message))

        total_bits += msg_len

    coded_ber = coded_errors / total_bits
    uncoded_ber = uncoded_errors / total_bits
    print(f'Rate-1/2 K=7 convolutional code, BPSK, {snr_db} dB SNR, {total_bits} bits:')
    print(f'  uncoded BER = {uncoded_ber:.4f}')
    print(f'  coded BER   = {coded_ber:.4f}')
    print(f'  coding gain: {"yes" if coded_ber < uncoded_ber else "no"} '
          f'({uncoded_ber / max(coded_ber, 1e-12):.1f}x fewer errors)')
    assert coded_ber < uncoded_ber


if __name__ == '__main__':
    main()
