"""commpy-mcp: a Model Context Protocol server exposing CommPy to AI agents.

After `pip install "commpy[mcp]"`, running `commpy-mcp` starts a stdio MCP
server that lets an assistant *drive CommPy simulations*: list what the library
can do, compute channel capacity, and run uncoded or coded (LDPC/polar/turbo)
bit-error-rate sweeps over AWGN.

The tool implementations below are plain functions of CommPy that return
JSON-serializable dictionaries, so they are fully testable without the `mcp`
package installed; `build_server` is the thin adapter that registers them with
`FastMCP`, imported lazily so importing this module never requires `mcp`.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from commpy import (
    Channels,
    LDPCCode,
    MPAMModulator,
    MPSKModulator,
    MQAMModulator,
    PolarCode,
    SimulationResult,
    TurboCode,
    channel_capacity_awgn,
    channel_capacity_bsc,
    simulate_ber,
    simulate_coded_ber,
)
from commpy._modulation.base import Modulator
from commpy._simulation.link_simulation import SoftInputCode

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

_MODULATORS: dict[str, type[Modulator]] = {
    'psk': MPSKModulator, 'qam': MQAMModulator, 'pam': MPAMModulator,
}


def _points(result: SimulationResult) -> list[dict[str, float]]:
    """Serialize a `SimulationResult` into a list of JSON-friendly per-SNR points."""
    return [
        {
            'snr_db': float(snr), 'error_rate': float(rate),
            'ci_lower': float(low), 'ci_upper': float(high), 'trials': int(trials),
        }
        for snr, rate, low, high, trials in zip(
            result.snr_db, result.error_rate, result.ci_lower, result.ci_upper,
            result.n_trials, strict=True,
        )
    ]


def list_capabilities() -> dict[str, Any]:
    """List the channel codes, modulations, and channels these tools can use."""
    return {
        'fec_codes': {
            'ldpc': 'Low-density parity-check; belief-propagation decoding',
            'polar': 'Polar; successive-cancellation / CRC-aided list decoding',
            'turbo': 'Rate-1/3 parallel-concatenated; iterative BCJR decoding',
        },
        'modulations': {
            'psk': 'M-PSK (order = M, a power of two)',
            'qam': 'Square M-QAM (order = M)',
            'pam': 'M-PAM (order = M)',
        },
        'channels': ['awgn', 'bsc'],
    }


def channel_capacity(
    channel: str, snr_db: float = 10.0, error_probability: float = 0.1,
) -> dict[str, Any]:
    """Compute the capacity (bits per channel use) of an AWGN or BSC channel.

    Args:
        channel: The channel model, `'awgn'` or `'bsc'`.
        snr_db: SNR in dB (used for `'awgn'`).
        error_probability: Crossover probability (used for `'bsc'`).
    """
    if channel == 'awgn':
        capacity = float(channel_capacity_awgn(10.0 ** (snr_db / 10.0)))
        return {'channel': 'awgn', 'snr_db': snr_db, 'capacity_bits_per_use': capacity}
    if channel == 'bsc':
        capacity = float(channel_capacity_bsc(error_probability))
        return {
            'channel': 'bsc', 'error_probability': error_probability,
            'capacity_bits_per_use': capacity,
        }
    msg = f"channel must be 'awgn' or 'bsc', got {channel!r}."
    raise ValueError(msg)


def run_ber_simulation(  # noqa: PLR0913 -- each parameter is a distinct simulation knob
    modulation: str,
    order: int,
    snr_db: list[float],
    target_errors: int = 100,
    max_bits: int = 200_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Run an uncoded Monte-Carlo BER sweep for a modulation over AWGN.

    Args:
        modulation: `'psk'`, `'qam'`, or `'pam'`.
        order: Constellation size `M` (a power of two).
        snr_db: SNR points (dB) to evaluate.
        target_errors: Early-stopping error target per SNR point.
        max_bits: Hard cap on simulated bits per SNR point.
        seed: RNG seed for reproducibility.
    """
    if modulation not in _MODULATORS:
        msg = f'modulation must be one of {sorted(_MODULATORS)}, got {modulation!r}.'
        raise ValueError(msg)
    modulator = _MODULATORS[modulation](order)
    result = simulate_ber(
        modulator, Channels.awgn, snr_db,
        target_errors=target_errors, max_trials=max_bits, rng=np.random.default_rng(seed),
    )
    return {'modulation': modulation, 'order': order, 'points': _points(result)}


def _build_code(code: str) -> SoftInputCode:
    """Construct a preset soft-input FEC code by name."""
    if code == 'ldpc':
        return LDPCCode.from_gallager(n=48, w_c=3, w_r=6, rng=np.random.default_rng(0))
    if code == 'polar':
        return PolarCode(block_length=64, k=32, design_snr_db=2.0)
    if code == 'turbo':
        return TurboCode(k=128, rng=np.random.default_rng(0))
    msg = f"code must be 'ldpc', 'polar', or 'turbo', got {code!r}."
    raise ValueError(msg)


def run_coded_ber_simulation(  # noqa: PLR0913 -- each parameter is a distinct simulation knob
    code: str,
    snr_db: list[float],
    modulation: str = 'psk',
    order: int = 2,
    target_errors: int = 50,
    max_blocks: int = 400,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a coded Monte-Carlo BER sweep for an LDPC/polar/turbo code over AWGN.

    Args:
        code: `'ldpc'`, `'polar'`, or `'turbo'` (a sensible preset of each).
        snr_db: SNR points (dB) to evaluate.
        modulation: `'psk'`, `'qam'`, or `'pam'`.
        order: Constellation size `M`.
        target_errors: Early-stopping error target per SNR point.
        max_blocks: Hard cap on simulated codeword blocks per SNR point.
        seed: RNG seed for reproducibility.
    """
    if modulation not in _MODULATORS:
        msg = f'modulation must be one of {sorted(_MODULATORS)}, got {modulation!r}.'
        raise ValueError(msg)
    block_code = _build_code(code)
    modulator = _MODULATORS[modulation](order)
    result = simulate_coded_ber(
        block_code, modulator, Channels.awgn, snr_db,
        target_errors=target_errors, max_trials=max_blocks * block_code.k,
        rng=np.random.default_rng(seed),
    )
    return {
        'code': code, 'n': block_code.n, 'k': block_code.k,
        'rate': block_code.k / block_code.n, 'points': _points(result),
    }


def build_server() -> 'FastMCP':
    """Build the `FastMCP` server with all CommPy tools registered.

    Raises:
        ImportError: If the `mcp` package (the `commpy[mcp]` extra) is not installed.
    """
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: PLC0415 -- optional extra, imported lazily
    except ImportError as exc:  # pragma: no cover -- only without the extra installed
        msg = "commpy-mcp requires the 'mcp' package. Install it with: pip install 'commpy[mcp]'"
        raise ImportError(msg) from exc

    server = FastMCP('commpy')
    for tool in (
        list_capabilities, channel_capacity, run_ber_simulation, run_coded_ber_simulation,
    ):
        server.tool()(tool)
    return server


def main() -> None:
    """Console entry point: run the CommPy MCP server over stdio."""
    build_server().run()


if __name__ == '__main__':
    main()
