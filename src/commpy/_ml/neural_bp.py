"""Neural (weighted) min-sum decoding of an LDPC code (PyTorch).

Unrolls the min-sum belief-propagation iterations of an existing `LDPCCode` into
a differentiable network with a learnable weight per iteration (Nachmani et al.'s
"neural belief propagation"). With all weights at 1 it is ordinary min-sum
decoding; the weights can be trained to claw back some of the loss min-sum
suffers versus sum-product. It reuses the code's precomputed Tanner-graph
adjacency directly, so it ties the learned layer to the classical decoder.
"""

import numpy as np
import torch
from torch import nn

from commpy._channelCoding.ldpc.code import LDPCCode


class NeuralMinSumDecoder(nn.Module):
    """Differentiable weighted-min-sum LDPC decoder over a code's Tanner graph."""

    # Declared so type-checkers see concrete tensor types for the registered
    # parameter/buffers (nn.Module.__getattr__ otherwise widens them to Module).
    weights: nn.Parameter
    edge_var: torch.Tensor
    check_edges: torch.Tensor
    check_mask: torch.Tensor

    def __init__(self, code: LDPCCode, num_iterations: int = 10) -> None:
        """Build the unrolled decoder.

        Args:
            code: The `LDPCCode` whose parity-check structure is decoded.
            num_iterations: Number of unrolled min-sum iterations (network depth).
        """
        super().__init__()
        graph = code._tanner  # noqa: SLF001 -- the decoder is built for this code's own graph
        if graph.check_edges.shape[1] < 2:
            msg = 'min-sum requires every check node to have degree >= 2.'
            raise ValueError(msg)
        self.n = code.n
        self.num_iterations = num_iterations
        self.weights = nn.Parameter(torch.ones(num_iterations))
        self.register_buffer('edge_var', torch.as_tensor(graph.edge_var, dtype=torch.long))
        self.register_buffer(
            'check_edges', torch.as_tensor(graph.check_edges, dtype=torch.long).clamp_min(0),
        )
        self.register_buffer('check_mask', torch.as_tensor(graph.check_mask, dtype=torch.bool))

    def _check_update(self, var_to_check: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Weighted min-sum check-node update over the padded adjacency grid."""
        magnitude = var_to_check.abs()[:, self.check_edges]  # (B, m, max_dc)
        sign = torch.sign(var_to_check)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)[:, self.check_edges]
        magnitude = magnitude.masked_fill(~self.check_mask, float('inf'))
        sign = sign.masked_fill(~self.check_mask, 1.0)

        two_smallest = torch.topk(magnitude, k=2, dim=2, largest=False)
        excl_min = two_smallest.values[..., 0:1].expand_as(magnitude).clone()
        excl_min.scatter_(2, two_smallest.indices[..., 0:1], two_smallest.values[..., 1:2])
        excl_sign = sign.prod(dim=2, keepdim=True) * sign  # divide out own sign (+/-1)

        grid = weight * excl_sign * excl_min
        check_to_var = torch.zeros_like(var_to_check)
        check_to_var[:, self.check_edges[self.check_mask]] = grid[:, self.check_mask]
        return check_to_var

    def forward(self, channel_llr: torch.Tensor) -> torch.Tensor:
        """Return per-bit posterior LLRs for a batch of channel LLRs, shape `(B, n)`."""
        edge_var = self.edge_var
        var_to_check = channel_llr[:, edge_var]
        posterior = channel_llr
        for iteration in range(self.num_iterations):
            check_to_var = self._check_update(var_to_check, self.weights[iteration])
            incoming = torch.zeros_like(channel_llr)
            incoming.index_add_(1, edge_var, check_to_var)
            posterior = channel_llr + incoming
            var_to_check = posterior[:, edge_var] - check_to_var
        return posterior

    @torch.no_grad()
    def decode(self, channel_llr: np.ndarray) -> np.ndarray:
        """Hard-decision decode a batch of channel LLRs into codeword bits.

        Args:
            channel_llr: LLRs of shape `(n,)` or `(batch, n)`
                (`log(P(0)/P(1))`, positive favors bit 0).

        Returns:
            Decoded codeword bits (`uint8`), same batch shape as the input.
        """
        llr = np.atleast_2d(np.asarray(channel_llr, dtype=np.float64))
        posterior = self.forward(torch.tensor(llr, dtype=torch.float32))
        bits = (posterior < 0).to(torch.uint8).numpy()
        return bits[0] if np.ndim(channel_llr) == 1 else bits


def train_neural_min_sum(  # noqa: PLR0913 -- each parameter is a distinct training hyperparameter
    decoder: NeuralMinSumDecoder,
    code: LDPCCode,
    snr_db: float,
    *,
    steps: int = 200,
    batch_size: int = 64,
    learning_rate: float = 1e-2,
    seed: int | None = None,
) -> list[float]:
    """Train the per-iteration weights on AWGN samples (BPSK, all-zero codewords).

    Uses the standard all-zero-codeword training setup: for a linear code the
    decoder's behavior is codeword-independent, so sampling the transmitted
    zero codeword suffices. The loss is the bitwise cross-entropy of the
    posterior against the true (all-zero) bits.

    Args:
        decoder: The `NeuralMinSumDecoder` to train (updated in place).
        code: The corresponding `LDPCCode` (for its length `n`).
        snr_db: Training SNR in dB.
        steps: Number of optimizer steps.
        batch_size: Codewords per step.
        learning_rate: Adam learning rate.
        seed: Optional seed for reproducible noise.

    Returns:
        The per-step training-loss history.
    """
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    noise_std = float(np.sqrt(10.0 ** (-snr_db / 10.0) / 2.0))

    history: list[float] = []
    decoder.train()
    for _ in range(steps):
        # All-zero codeword -> BPSK +1; received LLR mean is positive (favors 0).
        received = 1.0 + noise_std * rng.standard_normal((batch_size, code.n))
        channel_llr = torch.tensor(2.0 * received / (noise_std**2), dtype=torch.float32)
        posterior = decoder(channel_llr)
        # Target bit 0 everywhere; BCE expects P(bit=1) logit = -posterior.
        loss = loss_fn(-posterior, torch.zeros_like(posterior))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
    return history


__all__ = ['NeuralMinSumDecoder', 'train_neural_min_sum']
