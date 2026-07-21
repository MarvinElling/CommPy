"""Shared pytest fixtures and helpers for the CommPy test suite."""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducible statistical tests."""
    return np.random.default_rng(12345)


def assert_ber_close(measured: float, theoretical: float, tol: float) -> None:
    """Assert a measured bit-error rate is within `tol` of its theoretical value.

    Args:
        measured: Empirically observed error rate.
        theoretical: Closed-form/expected error rate.
        tol: Maximum allowed absolute deviation.
    """
    assert abs(measured - theoretical) < tol, (
        f'measured={measured!r} theoretical={theoretical!r} tol={tol!r}'
    )
