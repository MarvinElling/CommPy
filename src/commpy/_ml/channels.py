"""Differentiable channel models for end-to-end learning (PyTorch).

Complex baseband signals are represented as real tensors with a trailing size-2
dimension `(..., 2)` holding the in-phase/quadrature components -- the
autograd-friendly representation used throughout `commpy.ml`, so gradients flow
back through the channel to a learned transmitter.
"""

import torch


def awgn(symbols: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add complex AWGN at a given SNR to a batch of symbols.

    The noise variance is set from the batch's mean per-symbol energy so the
    resulting signal-to-noise ratio is `snr_db` (matching `Channels.awgn`'s
    average-power convention); the operation is differentiable in `symbols`.

    Args:
        symbols: Real tensor of shape `(..., 2)` (in-phase/quadrature).
        snr_db: Desired SNR in dB.

    Returns:
        The noisy symbols, same shape as `symbols`.

    Raises:
        ValueError: If the last dimension of `symbols` is not 2.
    """
    if symbols.shape[-1] != 2:
        msg = f'symbols must have a trailing dimension of size 2, got {symbols.shape[-1]}.'
        raise ValueError(msg)
    snr_linear = 10.0 ** (snr_db / 10.0)
    symbol_energy = symbols.pow(2).sum(dim=-1).mean()
    noise_std = torch.sqrt(symbol_energy / (2.0 * snr_linear))
    return symbols + noise_std * torch.randn_like(symbols)


def normalize_power(symbols: torch.Tensor) -> torch.Tensor:
    """Scale symbols to unit average per-symbol energy (the transmit power constraint).

    Args:
        symbols: Real tensor of shape `(..., 2)`.

    Returns:
        The scaled symbols with `mean(||symbol||^2) == 1`.
    """
    mean_energy = symbols.pow(2).sum(dim=-1).mean()
    return symbols / torch.sqrt(mean_energy.clamp_min(1e-12))


__all__ = ['awgn', 'normalize_power']
