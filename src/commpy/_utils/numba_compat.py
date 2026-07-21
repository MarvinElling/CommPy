"""Optional JIT-acceleration shim.

Hot-loop modules (e.g. Viterbi decoding) import `njit` from here rather than
from `numba` directly, so the package works correctly -- just slower -- with
or without the optional `numba` extra (`pip install commpy[fast]`) installed.
"""

from collections.abc import Callable
from typing import Any, TypeVar

_F = TypeVar('_F', bound=Callable[..., Any])

try:
    from numba import njit as _njit

    NUMBA_AVAILABLE = True
    njit = _njit
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- mirrors numba.njit's own permissive signature
        """No-op fallback for `numba.njit` when numba isn't installed."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func: _F) -> _F:
            return func

        return decorator


__all__ = ['NUMBA_AVAILABLE', 'njit']
