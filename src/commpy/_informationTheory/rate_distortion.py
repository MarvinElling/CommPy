"""Rate-distortion theory: the binary source, closed-form case."""

from commpy._informationTheory.formulas import binary_entropy


def rate_distortion_binary(p: float, distortion: float) -> float:
    """Rate-distortion function `R(D)` for a Bernoulli(`p`) source under Hamming distortion.

    Closed-form result (Cover & Thomas): `R(D) = H_b(p) - H_b(D)` for
    `0 <= D <= min(p, 1-p)`, and `R(D) = 0` beyond that (a single bit
    suffices -- always output the majority symbol -- once the allowed
    distortion already exceeds what guessing the majority symbol costs).

    Args:
        p: Source Bernoulli parameter, in `[0, 1]`.
        distortion: Allowed average Hamming distortion `D >= 0`.

    Returns:
        Minimum achievable rate (bits/source symbol) at distortion `D`.

    Raises:
        ValueError: If `distortion < 0` or `p` is not in `[0, 1]`.
    """
    if not 0 <= p <= 1:
        msg = f'p must be in [0, 1], got {p}.'
        raise ValueError(msg)
    if distortion < 0:
        msg = 'distortion must be non-negative.'
        raise ValueError(msg)
    d_max = min(p, 1 - p)
    if distortion >= d_max:
        return 0.0
    return binary_entropy(p) - binary_entropy(distortion)


__all__ = ['rate_distortion_binary']
