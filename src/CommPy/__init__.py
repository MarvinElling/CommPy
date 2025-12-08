"""CommPy package init.

Expose commonly used symbols at package level.
"""

from ._utils.maths import is_prime, modinv
from .math import Formulas

# Keep a backwards-compatible name `formulas` pointing to the class `Formulas`.
formulas = Formulas

__all__ = [
    'Formulas',
    'formulas',
    'is_prime',
    'modinv',
]
