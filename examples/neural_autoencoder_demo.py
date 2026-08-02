"""End-to-end learned modulation: a communication autoencoder (requires commpy[ml]).

Trains a small autoencoder to carry one of 16 messages over a single complex
channel use through AWGN, then plots the constellation it *learned* from scratch
-- the canonical "deep learning for the physical layer" demo. Requires the
optional PyTorch extra (`pip install "commpy[ml]"`); without it the demo prints
a note and exits cleanly.
"""

import matplotlib.pyplot as plt


def main() -> None:
    """Train a 16-message / 1-use autoencoder and plot its learned constellation."""
    try:
        import commpy.ml as ml  # noqa: PLC0415, PLR0402 -- optional extra, imported lazily so the demo degrades gracefully
    except ImportError:
        print("This demo needs the optional ML extra: pip install 'commpy[ml]'")
        return

    import torch  # noqa: PLC0415 -- only needed on the torch-available path above

    model = ml.Autoencoder(num_messages=16, num_channel_uses=1, hidden=64)
    snr_db = 15.0
    before = ml.block_error_rate(model, snr_db, num_messages=20_000, seed=1)
    ml.train_autoencoder(
        model, snr_db=snr_db, steps=1500, batch_size=512, learning_rate=1e-2, seed=0,
    )
    after = ml.block_error_rate(model, snr_db, num_messages=20_000, seed=1)
    print(f'16-message autoencoder over 1 complex use @ {snr_db:.0f} dB')
    print(f'  block-error rate: untrained={before:.4f}  trained={after:.4f}')

    constellation = model.transmit(torch.arange(model.num_messages)).detach().numpy().reshape(-1, 2)
    _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(constellation[:, 0], constellation[:, 1], s=80)
    for i, (re, im) in enumerate(constellation):
        ax.annotate(str(i), (re, im), textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('In-phase')
    ax.set_ylabel('Quadrature')
    ax.set_title('Learned constellation (autoencoder)')
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
