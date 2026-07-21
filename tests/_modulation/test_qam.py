"""Tests for commpy.MQAMModulator."""

import numpy as np
import pytest

from commpy import MQAMModulator


def test_rejects_non_power_of_four():
    for invalid_m in (2, 8, 32, 3, 5):
        with pytest.raises(ValueError, match='power of four'):
            MQAMModulator(invalid_m)


def test_16qam_has_16_distinct_symbols():
    mod = MQAMModulator(16)
    assert len(np.unique(mod.constellation)) == 16


def test_16qam_forms_a_symmetric_square_grid():
    mod = MQAMModulator(16)
    real_levels = sorted(set(np.round(mod.constellation.real, 8)))
    imag_levels = sorted(set(np.round(mod.constellation.imag, 8)))
    assert len(real_levels) == 4
    assert len(imag_levels) == 4
    # Symmetric about the origin.
    np.testing.assert_allclose(real_levels, [-x for x in reversed(real_levels)])


@pytest.mark.parametrize('m', [4, 16, 64, 256])
def test_unit_average_energy(m):
    mod = MQAMModulator(m)
    assert np.mean(np.abs(mod.constellation)**2) == pytest.approx(1.0)


def test_i_and_q_axes_are_independently_gray_coded():
    # Moving to a horizontally- or vertically-adjacent grid point should flip
    # exactly one bit, since I/Q are independently Gray-coded axes.
    mod = MQAMModulator(16)
    side = 4
    labels = mod.bit_labels.reshape(side, side, -1)
    for i in range(side - 1):
        for j in range(side):
            diff = labels[i, j] != labels[i + 1, j]
            assert diff.sum() == 1
    for i in range(side):
        for j in range(side - 1):
            diff = labels[i, j] != labels[i, j + 1]
            assert diff.sum() == 1
