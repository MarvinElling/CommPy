"""Successive-cancellation list (SCL) decoding of polar codes.

The decoder walks the polarization tree with the two exact log-domain kernels

- `_f` (check/upper node): `f(a, b) = ln((1 + e^(a+b)) / (e^a + e^b))`, and
- `_g` (variable/lower node): `g(a, b, u) = b + (1 - 2u) a`,

maintaining up to `list_size` candidate paths. Each path accumulates an exact
log-likelihood path metric, so with a list large enough that no pruning ever
discards the winner the decoder returns the maximum-likelihood codeword -- the
property the test-suite cross-validates against brute-force ML. Setting
`list_size = 1` recovers plain successive-cancellation (SC) decoding.

Paths only ever branch or get pruned at the leaves; the recursion therefore
carries, alongside each returned partial codeword, a "parent map" so an ancestor
node can realign its own per-path LLRs after a child has reshuffled the list.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

_FloatArray = NDArray[np.float64]
_ByteArray = NDArray[np.uint8]


def _f(a: _FloatArray, b: _FloatArray) -> _FloatArray:
    """Exact log-domain check-node (upper) LLR update."""
    return np.asarray(np.logaddexp(0.0, a + b) - np.logaddexp(a, b), dtype=np.float64)


def _g(a: _FloatArray, b: _FloatArray, u: _ByteArray) -> _FloatArray:
    """Exact log-domain variable-node (lower) LLR update given partial sums `u`."""
    return np.asarray(b + (1.0 - 2.0 * u.astype(np.float64)) * a, dtype=np.float64)


def _penalty(llr: float, bit: int) -> float:
    """Path-metric increment for deciding `bit` given decision LLR `llr` (>= 0)."""
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def polar_transform(u: ArrayLike) -> _ByteArray:
    """Apply the polar (Kronecker `F`) transform: encode `u` into a codeword.

    Computes `x = u * F^{kron n}` over GF(2) in place, where `F = [[1, 0], [1, 1]]`
    (the non-bit-reversed convention matched by the decoder).

    Args:
        u: Length-`N` bit vector (`N` a power of two), frozen positions zeroed.

    Returns:
        The length-`N` codeword (`uint8`).
    """
    x = np.asarray(u, dtype=np.uint8).copy()
    length = x.size
    step = 1
    while step < length:
        for start in range(0, length, 2 * step):
            x[start:start + step] ^= x[start + step:start + 2 * step]
        step *= 2
    return x


class _SCLDecoder:
    """Stateful successive-cancellation list decoder over the polarization tree."""

    def __init__(self, frozen: NDArray[np.bool_], list_size: int) -> None:
        self.frozen = frozen
        self.list_size = list_size
        self.block_length = int(frozen.size)
        self.metrics: list[float] = [0.0]
        self.decisions: list[_ByteArray] = [np.zeros(self.block_length, dtype=np.uint8)]

    def decode(self, channel_llr: ArrayLike) -> list[tuple[float, _ByteArray, _ByteArray]]:
        """Return surviving `(metric, u_hat, codeword)` paths, sorted best metric first."""
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=np.uint8)]
        llr = np.asarray(channel_llr, dtype=np.float64)
        codewords, _ = self._node([llr], 0, self.block_length)
        paths = list(zip(self.metrics, self.decisions, codewords, strict=True))
        return sorted(paths, key=lambda path: path[0])

    def _leaf(self, llrs: list[_FloatArray], index: int) -> tuple[list[_ByteArray], list[int]]:
        """Decode a single leaf, branching/pruning info bits; returns (betas, parent_map)."""
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=np.uint8) for _ in llrs], list(range(len(llrs)))

        candidates = [
            (self.metrics[path] + _penalty(float(llr[0]), bit), path, bit)
            for path, llr in enumerate(llrs)
            for bit in (0, 1)
        ]
        candidates.sort(key=lambda candidate: candidate[0])
        kept = candidates[:self.list_size]

        new_metrics: list[float] = []
        new_decisions: list[_ByteArray] = []
        betas: list[_ByteArray] = []
        parent_map: list[int] = []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            decision = self.decisions[path].copy()
            decision[index] = bit
            new_decisions.append(decision)
            betas.append(np.array([bit], dtype=np.uint8))
            parent_map.append(path)
        self.metrics = new_metrics
        self.decisions = new_decisions
        return betas, parent_map

    def _node(
        self, llrs: list[_FloatArray], base: int, length: int,
    ) -> tuple[list[_ByteArray], list[int]]:
        """Recursively decode a subtree; returns (partial codewords, parent_map)."""
        if length == 1:
            return self._leaf(llrs, base)

        half = length // 2
        upper = [_f(llr[:half], llr[half:]) for llr in llrs]
        beta_upper, map_upper = self._node(upper, base, half)

        a = [llrs[map_upper[p]][:half] for p in range(len(map_upper))]
        b = [llrs[map_upper[p]][half:] for p in range(len(map_upper))]
        lower = [_g(a[p], b[p], beta_upper[p]) for p in range(len(beta_upper))]
        beta_lower, map_lower = self._node(lower, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map


def scl_decode(
    channel_llr: ArrayLike, frozen: NDArray[np.bool_], list_size: int,
) -> list[tuple[float, _ByteArray, _ByteArray]]:
    """Successive-cancellation list decode; `list_size = 1` is plain SC.

    Args:
        channel_llr: Channel LLRs, length `N`, `L = log(P(0)/P(1))`.
        frozen: Length-`N` boolean frozen mask (`True` = frozen to 0).
        list_size: Number of survivor paths `L` (`>= 1`).

    Returns:
        Surviving `(path_metric, u_hat, codeword)` tuples, best metric first.

    Raises:
        ValueError: If `list_size < 1` or `len(channel_llr) != len(frozen)`.
    """
    if list_size < 1:
        msg = f'list_size must be >= 1, got {list_size}.'
        raise ValueError(msg)
    llr = np.asarray(channel_llr, dtype=np.float64)
    if llr.size != frozen.size:
        msg = f'channel_llr must have length {frozen.size}, got {llr.size}.'
        raise ValueError(msg)
    return _SCLDecoder(frozen, list_size).decode(llr)


__all__ = ['polar_transform', 'scl_decode']
