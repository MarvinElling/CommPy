"""Tests for commpy's structural forward-error-correction plots."""

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from commpy import (
    BlockInterleaver,
    HammingCode,
    LDPCCode,
    PolarCode,
    Trellis,
    TurboCode,
    plot_frozen_bits,
    plot_interleaver,
    plot_parity_check,
    plot_polar_reliabilities,
    plot_tanner_graph,
    plot_trellis,
)
from commpy._channelCoding.turbo.rsc import RSCTrellis


@pytest.fixture
def ldpc(rng):
    return LDPCCode.from_gallager(48, 3, 6, rng=rng)


def line_collections(ax):
    return [c for c in ax.collections if isinstance(c, LineCollection)]


# matplotlib types these as optional; the tests only reach them once drawn.
def image_array(ax, index=0):
    array = ax.images[index].get_array()
    assert array is not None
    return np.asarray(array)


def legend_labels(ax):
    legend = ax.get_legend()
    assert legend is not None
    return [text.get_text() for text in legend.get_texts()]


class TestParityCheck:
    def test_draws_the_matrix_and_reports_its_density(self, ldpc):
        ax = plot_parity_check(ldpc)
        assert len(ax.images) == 1
        assert np.array_equal(image_array(ax), ldpc.H)
        # A regular (3, 6) Gallager matrix has exactly 3 ones per column.
        assert '12.5%' in ax.get_title()

    def test_accepts_a_bare_matrix(self, ldpc):
        assert isinstance(plot_parity_check(ldpc.H), Axes)

    def test_accepts_other_codes_carrying_an_h_matrix(self):
        ax = plot_parity_check(HammingCode(3))
        assert image_array(ax).shape == (3, 7)


class TestTannerGraph:
    def test_edge_count_matches_the_ones_of_h(self, ldpc):
        ax = plot_tanner_graph(ldpc)
        edges = line_collections(ax)[0].get_segments()
        assert len(edges) == int(ldpc.H.sum())

    def test_both_node_sets_are_drawn_and_named(self, ldpc):
        ax = plot_tanner_graph(ldpc)
        assert len(ax.collections) == 3  # edges + variable nodes + check nodes
        assert legend_labels(ax) == ['variable nodes', 'check nodes']

    def test_title_reports_the_graph_size(self, ldpc):
        title = plot_tanner_graph(ldpc).get_title()
        assert '48 variable nodes' in title
        assert '24 checks' in title


class TestTrellis:
    def test_one_branch_per_state_and_input_at_every_stage(self):
        trellis = Trellis(3, [0o7, 0o5])
        ax = plot_trellis(trellis, n_stages=4)
        drawn = sum(len(c.get_segments()) for c in line_collections(ax))
        assert drawn == 4 * trellis.n_states * 2

    def test_input_bits_differ_by_line_style_not_only_color(self):
        ax = plot_trellis(Trellis(3, [0o7, 0o5]), n_stages=2)
        # get_linestyle() returns dash specs holding lists, so compare reprs.
        styles = {str(c.get_linestyle()) for c in line_collections(ax)}
        assert len(styles) == 2

    def test_accepts_a_recursive_systematic_trellis(self):
        ax = plot_trellis(RSCTrellis(3, 0o7, 0o5), n_stages=3)
        assert sum(len(c.get_segments()) for c in line_collections(ax)) == 3 * 4 * 2

    def test_labels_annotate_every_branch(self):
        trellis = Trellis(3, [0o7, 0o5])
        ax = plot_trellis(trellis, n_stages=2, labels=True)
        assert len(ax.texts) == 2 * trellis.n_states * 2

    def test_rejects_a_non_positive_stage_count(self):
        with pytest.raises(ValueError, match='at least 1'):
            plot_trellis(Trellis(3, [0o7, 0o5]), n_stages=0)


class TestPolar:
    def test_both_constructions_are_drawn_as_ranks(self):
        ax = plot_polar_reliabilities(64)
        assert len(ax.lines) == 2
        for line in ax.lines:
            ranks = np.asarray(line.get_ydata())
            # A rank curve is a permutation of 0..N-1, whatever the raw scores.
            assert np.array_equal(np.sort(ranks), np.arange(64))

    def test_ranks_are_comparable_because_scales_are_not(self):
        # The Bhattacharyya score is in [-1, 0] and the Gaussian one is an
        # unbounded mean LLR; plotting raw scores together would be misleading,
        # so both series must share the rank axis.
        ax = plot_polar_reliabilities(64)
        lower, upper = ax.get_ylim()
        assert lower < 0
        assert upper > 63

    def test_frozen_map_marks_exactly_the_frozen_positions(self):
        code = PolarCode(64, 32)
        ax = plot_frozen_bits(code)
        grid = image_array(ax)
        assert grid.sum() == code.n - code.k
        assert np.array_equal(grid.ravel(), code.frozen)

    def test_frozen_map_is_folded_into_a_rectangle(self):
        assert image_array(plot_frozen_bits(PolarCode(256, 128))).shape == (16, 16)


class TestInterleaver:
    def test_plots_the_permutation_of_a_turbo_code(self, rng):
        code = TurboCode(64, rng=rng)
        ax = plot_interleaver(code)
        assert np.array_equal(np.asarray(ax.collections[0].get_offsets())[:, 1], code.interleaver)

    def test_accepts_a_block_interleaver(self):
        interleaver = BlockInterleaver(4, 8)
        ax = plot_interleaver(interleaver)
        drawn = np.asarray(ax.collections[0].get_offsets())[:, 1]
        assert np.array_equal(drawn, interleaver.interleave(np.arange(32)))

    def test_accepts_a_bare_permutation(self):
        perm = np.array([2, 0, 3, 1])
        ax = plot_interleaver(perm)
        assert np.array_equal(np.asarray(ax.collections[0].get_offsets())[:, 1], perm)

    def test_axes_are_square_so_clustering_is_not_distorted(self, rng):
        assert plot_interleaver(TurboCode(32, rng=rng)).get_aspect() == 1.0
