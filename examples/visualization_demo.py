"""Visualization tour: one figure from each family of CommPy plots.

Builds a small 16-QAM link and an LDPC code, then draws a representative plot
from each of the four families -- signal/modulation, FEC structure, decoder
diagnostics, and system/channel -- into a single 2x3 figure. Every `plot_*`
function accepts an `ax` and returns it, which is exactly what makes this
composition possible.

Run it directly to see the figure; `--save DIR` writes the panels out
individually instead, which is how the documentation gallery images are made.
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from commpy import (
    Channels,
    LDPCCode,
    MPSKModulator,
    MQAMModulator,
    Trellis,
    commpy_style,
    plot_capacity_curves,
    plot_constellation,
    plot_decoder_convergence,
    plot_eye_diagram,
    plot_llr_histogram,
    plot_psd,
    plot_tanner_graph,
    plot_trellis,
    plot_viterbi_paths,
    root_raised_cosine_filter,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes

SPS = 8  # samples per symbol for the pulse-shaped waveform


def build_waveform(
    modulator: MQAMModulator, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (received symbols, pulse-shaped waveform) for a small 16-QAM link."""
    symbols = modulator.modulate(rng.integers(0, 2, 4000))
    received = Channels.awgn(symbols, 18.0, rng=rng)

    taps = root_raised_cosine_filter(1.0, 0.35, 6)(np.arange(0.0, 6.0, 1.0 / SPS))
    upsampled = np.zeros(symbols.size * SPS, dtype=np.complex128)
    upsampled[::SPS] = symbols
    return received, np.convolve(upsampled, taps, mode='same')


def build_coded_link(
    rng: np.random.Generator,
) -> tuple[LDPCCode, np.ndarray, np.ndarray]:
    """Return an LDPC code with one received block's codeword and channel LLRs."""
    code = LDPCCode.from_gallager(n=96, w_c=3, w_r=6, rng=rng)
    bpsk = MPSKModulator(2)
    codeword = code.encode(rng.integers(0, 2, code.k).astype(np.uint8))
    noisy = Channels.awgn(bpsk.modulate(codeword), 0.0, rng=rng)
    return code, codeword, bpsk.soft_demodulate(noisy, 1.0)


def main() -> None:
    """Draw one representative plot from each visualization family."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save', type=Path, default=None,
                        help='write each panel to this directory instead of showing them')
    args, _ = parser.parse_known_args()

    rng = np.random.default_rng(0)
    modulator = MQAMModulator(16)
    received, waveform = build_waveform(modulator, rng)
    code, codeword, llr = build_coded_link(rng)
    trellis = Trellis(3, [0o7, 0o5])
    hard_received = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1])

    panels: list[tuple[str, Callable[[Axes], Axes]]] = [
        ('constellation', lambda ax: plot_constellation(modulator, received=received, ax=ax)),
        ('eye_diagram', lambda ax: plot_eye_diagram(waveform, SPS, n_traces=120, ax=ax)),
        ('psd', lambda ax: plot_psd(waveform, fs=float(SPS), ax=ax)),
        ('tanner_graph', lambda ax: plot_tanner_graph(code, ax=ax)),
        ('llr_histogram', lambda ax: plot_llr_histogram(llr, bits=codeword, ax=ax)),
        ('decoder_convergence', lambda ax: plot_decoder_convergence(code, llr,
                                                                    max_iter=10, ax=ax)),
        ('trellis', lambda ax: plot_trellis(trellis, n_stages=4, ax=ax)),
        ('viterbi_paths', lambda ax: plot_viterbi_paths(trellis, hard_received,
                                                        max_stages=6, ax=ax)),
        ('capacity_curves', lambda ax: plot_capacity_curves(ax=ax)),
    ]

    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        with commpy_style():
            for name, draw in panels:
                figure, ax = plt.subplots(figsize=(6.5, 4.5))
                draw(ax)
                figure.savefig(args.save / f'{name}.png', dpi=140, bbox_inches='tight')
                plt.close(figure)
        print(f'Wrote {len(panels)} panels to {args.save}')
        return

    figure, axes = plt.subplots(3, 3, figsize=(17, 13))
    for (name, draw), ax in zip(panels, axes.ravel(), strict=True):
        draw(ax)
        print(f'drew {name}')
    figure.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
