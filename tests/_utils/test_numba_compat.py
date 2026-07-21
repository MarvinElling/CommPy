"""Tests for commpy._utils.numba_compat."""

from commpy._utils.numba_compat import NUMBA_AVAILABLE, njit


def test_njit_bare_decorator():
    @njit
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_njit_parametrized_decorator():
    @njit(cache=True)
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_numba_available_is_bool():
    assert isinstance(NUMBA_AVAILABLE, bool)
