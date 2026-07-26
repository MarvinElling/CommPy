"""End-to-end learned communication as an autoencoder (PyTorch).

The transmitter (a small MLP + power-normalization) maps one of `M` messages to
`n` complex channel uses; a differentiable AWGN channel corrupts them; the
receiver (another MLP) recovers the message. Trained jointly to minimize
message error, the pair *learns a constellation and its detector* end to end --
the canonical "deep learning for the physical layer" demonstration, here in a
few dozen lines with no framework beyond PyTorch.
"""

import torch
from torch import nn

from commpy._ml.channels import awgn, normalize_power


class Autoencoder(nn.Module):
    """An `(n, M)` communication autoencoder: `M` messages over `n` complex uses.

    The transmitter maps a message index to `n` unit-average-energy complex
    symbols; the receiver maps the noisy symbols back to message logits.
    """

    def __init__(self, num_messages: int, num_channel_uses: int, hidden: int = 64) -> None:
        """Build the autoencoder.

        Args:
            num_messages: Alphabet size `M` (the code carries `log2(M)` bits over
                `num_channel_uses` symbols).
            num_channel_uses: Number of complex channel uses `n` per message.
            hidden: Hidden-layer width of the transmitter/receiver MLPs.
        """
        super().__init__()
        self.num_messages = num_messages
        self.num_channel_uses = num_channel_uses
        self.transmitter = nn.Sequential(
            nn.Linear(num_messages, hidden), nn.ReLU(), nn.Linear(hidden, 2 * num_channel_uses),
        )
        self.receiver = nn.Sequential(
            nn.Linear(2 * num_channel_uses, hidden), nn.ReLU(), nn.Linear(hidden, num_messages),
        )

    def transmit(self, messages: torch.Tensor) -> torch.Tensor:
        """Map message indices to power-normalized symbols, shape `(batch, n, 2)`."""
        one_hot = nn.functional.one_hot(messages, self.num_messages).float()
        symbols = self.transmitter(one_hot).view(-1, self.num_channel_uses, 2)
        return normalize_power(symbols)

    def forward(self, messages: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Transmit messages through the AWGN channel and return receiver logits."""
        received = awgn(self.transmit(messages), snr_db)
        return self.receiver(received.reshape(received.shape[0], -1))


def train_autoencoder(  # noqa: PLR0913 -- each parameter is a distinct training hyperparameter
    model: Autoencoder,
    snr_db: float,
    *,
    steps: int = 2000,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    seed: int | None = None,
) -> list[float]:
    """Train an `Autoencoder` at a fixed SNR with cross-entropy on the message.

    Args:
        model: The `Autoencoder` to train (updated in place).
        snr_db: Training SNR in dB.
        steps: Number of optimizer steps.
        batch_size: Messages per step.
        learning_rate: Adam learning rate.
        seed: Optional seed for reproducible message/noise sampling.

    Returns:
        The per-step training-loss history.
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    history: list[float] = []
    model.train()
    for _ in range(steps):
        messages = torch.randint(0, model.num_messages, (batch_size,), generator=generator)
        logits = model(messages, snr_db)
        loss = loss_fn(logits, messages)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
    return history


@torch.no_grad()
def block_error_rate(
    model: Autoencoder, snr_db: float, num_messages: int = 100_000, seed: int | None = None,
) -> float:
    """Estimate the block (message) error rate of a trained autoencoder at `snr_db`."""
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    model.eval()
    messages = torch.randint(0, model.num_messages, (num_messages,), generator=generator)
    decoded = model(messages, snr_db).argmax(dim=1)
    return float((decoded != messages).float().mean().item())


__all__ = ['Autoencoder', 'block_error_rate', 'train_autoencoder']
