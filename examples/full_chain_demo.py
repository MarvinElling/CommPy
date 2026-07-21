"""Full transmit/receive chain, composing most of CommPy's pieces end to end.

    text --Huffman--> compressed bits --convolutional code--> coded bits
         --BPSK modulate--> symbols --multipath channel + AWGN-->
         --ZF equalize--> --soft demodulate (LLRs)--> --Viterbi decode-->
    coded bits --Huffman decode--> text

This is the classic layered picture: source coding removes redundancy,
channel coding adds structured redundancy back for error protection
(the separation theorem), and the physical layer (modulation, channel,
equalization) sits in between.
"""

from collections import Counter

import numpy as np

from commpy import (
    Channels,
    ConvolutionalEncoder,
    MPSKModulator,
    Trellis,
    huffman_codes,
    huffman_decode,
    huffman_encode,
    viterbi_decode,
    zf_equalizer,
)

TEXT = 'the quick brown fox jumps over the lazy dog ' * 4
CHANNEL_TAPS = np.array([1.0, 0.3, -0.15])  # a short multipath channel
EQUALIZER_TAPS = 21
SNR_DB = 8.0


def main() -> None:
    """Run the full transmit/receive chain demo."""
    rng = np.random.default_rng(0)

    # 1. Source coding: Huffman-compress the text.
    counts = Counter(TEXT)
    total = sum(counts.values())
    probabilities = {ch: n / total for ch, n in counts.items()}
    codes = huffman_codes(probabilities)
    compressed = huffman_encode(list(TEXT), codes)
    bits = np.array([int(b) for b in compressed])
    print(f'1. Huffman: {len(TEXT) * 8} bits -> {len(bits)} bits')

    # 2. Channel coding: rate-1/2 convolutional code, zero-tail terminated.
    trellis = Trellis(constraint_length=7, generators=(0o171, 0o133))
    encoder = ConvolutionalEncoder(trellis)
    codeword, _ = encoder.encode(bits, terminate=True)
    print(f'2. Convolutional code: {len(bits)} bits -> {len(codeword)} coded bits')

    # 3. Modulation: BPSK.
    mod = MPSKModulator(2)
    symbols = mod.modulate(codeword)

    # 4. Channel: multipath (ISI) + AWGN.
    tx = np.convolve(symbols, CHANNEL_TAPS)
    rx = Channels.awgn(tx, SNR_DB, rng=rng)
    print(f'4. Multipath channel ({len(CHANNEL_TAPS)} taps) + AWGN at {SNR_DB} dB SNR')

    # 5. Equalization: zero-forcing FIR equalizer removes the ISI.
    equalizer = zf_equalizer(CHANNEL_TAPS, EQUALIZER_TAPS)
    equalized = np.convolve(rx, equalizer)
    delay = (len(CHANNEL_TAPS) + EQUALIZER_TAPS - 1) // 2
    recovered_symbols = equalized[delay:delay + len(symbols)]

    # 6. Soft demodulation (LLRs) + 7. soft-decision Viterbi decoding.
    llrs = mod.soft_demodulate(recovered_symbols, noise_var=1.0)
    decoded_bits = viterbi_decode(trellis, llrs, mode='soft', terminated=True)
    bit_errors = int(np.sum(decoded_bits != bits))
    print(f'6-7. Equalize + soft-demodulate + Viterbi-decode: {bit_errors}/{len(bits)} bit errors')

    # 8. Source decoding: Huffman-decompress back to text.
    recovered_text = ''.join(huffman_decode(''.join(str(b) for b in decoded_bits), codes))
    print(f'8. Huffman decode: {"OK, exact match" if recovered_text == TEXT else "MISMATCH"}')

    assert bit_errors == 0
    assert recovered_text == TEXT
    print('\nFull chain OK: original text recovered exactly after the full TX/RX pipeline.')


if __name__ == '__main__':
    main()
