"""Tests for commpy.commpy_style and commpy.series_colors (shared plot styling)."""

import matplotlib.pyplot as plt
import pytest

from commpy import commpy_style, series_colors


def test_style_applies_inside_its_context():
    with commpy_style():
        assert plt.rcParams['axes.grid'] is True
        assert plt.rcParams['legend.frameon'] is False


def test_default_colormap_resolves_inside_the_context():
    # rcParams stores image.cmap by name, so the colormap must be registered
    # before anything inside the context asks matplotlib to resolve it.
    with commpy_style():
        _, ax = plt.subplots()
        image = ax.imshow([[0.0, 1.0], [1.0, 0.0]])
        assert image.get_cmap().name == 'commpy_sequential'


def test_style_is_scoped_and_restores_the_previous_settings():
    before = plt.rcParams['axes.prop_cycle']
    with commpy_style():
        assert plt.rcParams['axes.prop_cycle'] != before
    assert plt.rcParams['axes.prop_cycle'] == before


def test_series_colors_are_stable_across_lengths():
    # Color tracks the entity, not its rank: asking for fewer series must not
    # renumber the ones that remain.
    assert series_colors(3) == series_colors(8)[:3]


def test_series_colors_are_distinct_within_the_palette():
    colors = series_colors(8)
    assert len(set(colors)) == 8


def test_series_colors_warns_once_hues_must_repeat():
    with pytest.warns(UserWarning, match='hues will repeat'):
        colors = series_colors(9)
    assert colors[8] == colors[0]
