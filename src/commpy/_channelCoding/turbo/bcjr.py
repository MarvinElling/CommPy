"""BCJR (log-MAP) decoding of one RSC constituent code.

Computes exact a-posteriori bit LLRs on the RSC trellis via the forward/backward
recursion, entirely in the log domain (`max*` = `logaddexp`, i.e. true log-MAP,
not the max-log approximation). All state recursions are vectorized over the
trellis states; only the time axis is a Python loop.

The decoder returns *extrinsic* LLRs -- the a-posteriori LLR with the systematic
channel LLR and the incoming a-priori LLR removed -- which is exactly what the
partner decoder consumes as its a-priori input in turbo iteration. LLRs follow
the package convention `L = log(P(0)/P(1))` (positive favors bit 0).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from commpy._channelCoding.turbo.rsc import RSCTrellis

_NEG_INF = -1.0e30


def bcjr_decode(
    trellis: RSCTrellis,
    apriori_llr: ArrayLike,
    systematic_llr: ArrayLike,
    parity_llr: ArrayLike,
) -> NDArray[np.float64]:
    """Log-MAP decode one RSC block; return per-bit *extrinsic* LLRs.

    Args:
        trellis: The constituent `RSCTrellis`.
        apriori_llr: A-priori LLRs for the information bits, length `k`.
        systematic_llr: Channel LLRs for the systematic bits, length `k`.
        parity_llr: Channel LLRs for the parity bits, length `k`.

    Returns:
        Extrinsic LLRs for the information bits, length `k`.

    Raises:
        ValueError: If the three LLR inputs are not all the same length.
    """
    la = np.asarray(apriori_llr, dtype=np.float64)
    ls = np.asarray(systematic_llr, dtype=np.float64)
    lp = np.asarray(parity_llr, dtype=np.float64)
    if not la.size == ls.size == lp.size:
        msg = 'apriori_llr, systematic_llr, and parity_llr must have equal length.'
        raise ValueError(msg)

    k = la.size
    n_states = trellis.n_states
    next0 = trellis.next_state[:, 0]
    next1 = trellis.next_state[:, 1]
    # Branch metric gamma[t, s, u] = 0.5 * ((1-2u)(La+Ls) + (1-2*parity)(Lp)).
    sys_sign = 1.0 - 2.0 * trellis.parity.astype(np.float64)  # (n_states, 2)
    # gamma per input, shape (k, n_states):
    g0 = 0.5 * (la + ls)[:, None] + 0.5 * sys_sign[None, :, 0] * lp[:, None]
    g1 = -0.5 * (la + ls)[:, None] + 0.5 * sys_sign[None, :, 1] * lp[:, None]

    # Forward recursion. Each per-step contribution scatters by the (bijective)
    # state map; states are max-normalized per step to prevent overflow (only
    # LLR *differences* matter, so a per-step constant is irrelevant).
    alpha = np.full((k + 1, n_states), _NEG_INF)
    alpha[0, 0] = 0.0
    contrib0 = np.full(n_states, _NEG_INF)
    contrib1 = np.full(n_states, _NEG_INF)
    for t in range(k):
        prev = alpha[t]
        contrib0[next0] = prev + g0[t]
        contrib1[next1] = prev + g1[t]
        nxt = np.logaddexp(contrib0, contrib1)
        alpha[t + 1] = nxt - nxt.max()

    # Backward recursion (unterminated: all terminal states equally likely).
    beta = np.full((k + 1, n_states), _NEG_INF)
    beta[k] = 0.0
    for t in range(k - 1, -1, -1):
        nxt = beta[t + 1]
        b = np.logaddexp(nxt[next0] + g0[t], nxt[next1] + g1[t])
        beta[t] = b - b.max()

    # A-posteriori LLR per bit, vectorized over the whole block, then reduced to
    # extrinsic information (a-priori and systematic channel parts removed).
    term0 = alpha[:k] + g0 + beta[1:][:, next0]
    term1 = alpha[:k] + g1 + beta[1:][:, next1]
    apposteriori = np.logaddexp.reduce(term0, axis=1) - np.logaddexp.reduce(term1, axis=1)
    return np.asarray(apposteriori - ls - la, dtype=np.float64)


__all__ = ['bcjr_decode']
