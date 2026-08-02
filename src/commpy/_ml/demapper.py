"""A learned (neural) soft demapper (PyTorch).

A small MLP is trained to map a received complex symbol to per-bit
log-likelihood ratios for a given `Modulator`'s constellation -- a drop-in
learned replacement for `Modulator.soft_demodulate`. It reproduces the analytic
demapper on AWGN and, unlike the closed form, can be trained directly on samples
from a channel with no tractable likelihood.
"""

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from commpy._channels.channels import Channels
from commpy._modulation.base import Modulator


class NeuralDemapper(nn.Module):
    """MLP mapping a received symbol `(real, imag)` to `bits_per_symbol` bit LLRs."""

    def __init__(self, bits_per_symbol: int, hidden: int = 64) -> None:
        """Build the demapper.

        Args:
            bits_per_symbol: Number of bits per constellation symbol.
            hidden: Hidden-layer width.
        """
        super().__init__()
        self.bits_per_symbol = bits_per_symbol
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, bits_per_symbol),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return per-bit logits for received symbols, shape `(N, bits_per_symbol)`.

        A logit is the network's estimate of `log(P(bit=1) / P(bit=0))`, so the
        package-convention LLR (`log(P(0)/P(1))`) is its negation.
        """
        return self.net(features)

    @torch.no_grad()
    def soft_demodulate(self, received: NDArray[np.complex128]) -> NDArray[np.float64]:
        """Compute bit LLRs for received symbols, matching `Modulator.soft_demodulate`.

        Args:
            received: Received complex baseband symbols, shape `(N,)`.

        Returns:
            LLRs (`log(P(0)/P(1))`, positive favors bit 0), length
            `N * bits_per_symbol`, in the same bit ordering as `Modulator`.
        """
        features = torch.tensor(
            np.stack([received.real, received.imag], axis=1), dtype=torch.float32,
        )
        llrs = -self.forward(features)  # logit is log P(1)/P(0); LLR is its negation
        return np.asarray(llrs.numpy().reshape(-1), dtype=np.float64)


def train_demapper(  # noqa: PLR0913 -- each parameter is a distinct training hyperparameter
    model: NeuralDemapper,
    modulator: Modulator,
    snr_db: float,
    *,
    steps: int = 2000,
    symbols_per_step: int = 2000,
    learning_rate: float = 1e-3,
    seed: int | None = None,
) -> list[float]:
    """Train a `NeuralDemapper` on AWGN samples of a `Modulator`'s constellation.

    Args:
        model: The `NeuralDemapper` to train (updated in place).
        modulator: The reference `Modulator` supplying the constellation.
        snr_db: Training SNR in dB.
        steps: Number of optimizer steps.
        symbols_per_step: Symbols drawn per step.
        learning_rate: Adam learning rate.
        seed: Optional seed for reproducible sampling.

    Returns:
        The per-step training-loss history.
    """
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    bits_per_symbol = modulator.bits_per_symbol

    history: list[float] = []
    model.train()
    for _ in range(steps):
        bits = rng.integers(0, 2, symbols_per_step * bits_per_symbol)
        received = Channels.awgn(modulator.modulate(bits), snr_db, rng=rng)
        features = torch.tensor(
            np.stack([received.real, received.imag], axis=1), dtype=torch.float32,
        )
        target = torch.tensor(bits.reshape(-1, bits_per_symbol), dtype=torch.float32)
        loss = loss_fn(model(features), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
    return history


__all__ = ['NeuralDemapper', 'train_demapper']
