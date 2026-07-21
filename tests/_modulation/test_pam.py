"""Tests for commpy.MPAMModulator."""

import numpy as np
import pytest

from commpy import MPAMModulator


def test_2pam_is_plus_minus_one():
    mod = MPAMModulator(2)
    np.testing.assert_allclose(sorted(mod.constellation.real), [-1.0, 1.0])


def test_levels_are_equally_spaced_and_symmetric():
    mod = MPAMModulator(4)
    levels = sorted(mod.constellation.real)
    spacings = np.diff(levels)
    np.testing.assert_allclose(spacings, spacings[0])
    np.testing.assert_allclose(sum(levels), 0.0, atol=1e-9)


@pytest.mark.parametrize('m', [2, 4, 8, 16])
def test_unit_average_energy(m):
    mod = MPAMModulator(m)
    assert np.mean(np.abs(mod.constellation)**2) == pytest.approx(1.0)


def test_adjacent_levels_differ_by_one_bit():
    mod = MPAMModulator(8)
    order = np.argsort(mod.constellation.real)
    ordered_labels = mod.bit_labels[order]
    for i in range(len(ordered_labels) - 1):
        diff = ordered_labels[i] != ordered_labels[i + 1]
        assert diff.sum() == 1
