"""Tests for polar-code frozen-set construction.

Checks the structural guarantees both reliability metrics must satisfy: the
selected free set has the requested size, polarization order is respected (the
last synthetic channel is the most reliable, the first the least), and invalid
parameters are rejected.
"""

import numpy as np
import pytest

from commpy._channelCoding.polar.construction import (
    bhattacharyya_reliabilities,
    frozen_mask,
    gaussian_approx_reliabilities,
)


@pytest.mark.parametrize('method', ['gaussian', 'bhattacharyya'])
@pytest.mark.parametrize('n_free', [1, 4, 7])
def test_frozen_mask_selects_requested_free_count(method, n_free):
    mask = frozen_mask(8, n_free, method=method, design_snr_db=1.0)
    assert mask.shape == (8,)
    assert int((~mask).sum()) == n_free  # exactly n_free free positions
    assert mask.dtype == np.bool_


@pytest.mark.parametrize(
    'reliabilities', [gaussian_approx_reliabilities, bhattacharyya_reliabilities],
)
def test_polarization_orders_channels(reliabilities):
    score = reliabilities(16, 1.0)
    assert score.shape == (16,)
    # In natural order the first synthetic channel is the worst, the last best.
    assert score[-1] == score.max()
    assert score[0] == score.min()


@pytest.mark.parametrize('method', ['gaussian', 'bhattacharyya'])
def test_single_free_channel_is_the_most_reliable(method):
    mask = frozen_mask(16, 1, method=method, design_snr_db=1.0)
    # The one free position must be the last (most reliable) synthetic channel.
    assert np.flatnonzero(~mask).tolist() == [15]


def test_frozen_mask_rejects_bad_arguments():
    with pytest.raises(ValueError, match='power of two'):
        frozen_mask(12, 4)
    with pytest.raises(ValueError, match='n_free'):
        frozen_mask(8, 0)
    with pytest.raises(ValueError, match='method'):
        frozen_mask(8, 4, method='bogus')
