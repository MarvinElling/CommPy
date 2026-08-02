"""AI-for-wireless layer for CommPy (optional; requires `pip install "commpy[ml]"`).

This module is deliberately **not** imported by `commpy/__init__.py`, so the
base install stays NumPy/SciPy-only and importing `commpy` never pulls in
PyTorch. Import it explicitly:

    import commpy.ml as ml

It provides differentiable, trainable physical-layer components built on
PyTorch: a differentiable AWGN channel, an end-to-end learned `Autoencoder`, a
`NeuralDemapper` (a learned `Modulator.soft_demodulate`), and a
`NeuralMinSumDecoder` that unrolls an `LDPCCode`'s belief propagation into a
trainable network.
"""

try:
    import torch as _torch  # noqa: F401
except ImportError as exc:  # pragma: no cover -- exercised only without the extra installed
    _msg = (
        "commpy.ml requires PyTorch, which is not installed. "
        "Install it with: pip install 'commpy[ml]'"
    )
    raise ImportError(_msg) from exc

from commpy._ml.autoencoder import Autoencoder, block_error_rate, train_autoencoder
from commpy._ml.channels import awgn, normalize_power
from commpy._ml.demapper import NeuralDemapper, train_demapper
from commpy._ml.neural_bp import NeuralMinSumDecoder, train_neural_min_sum

__all__ = [
    'Autoencoder',
    'NeuralDemapper',
    'NeuralMinSumDecoder',
    'awgn',
    'block_error_rate',
    'normalize_power',
    'train_autoencoder',
    'train_demapper',
    'train_neural_min_sum',
]
