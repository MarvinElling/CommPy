import matplotlib as mpl

mpl.use('Agg')  # headless backend so the plot_* functions don't require a display

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test created.

    The plotting functions open a new figure whenever no axes is passed, and
    matplotlib warns once more than 20 stay open. Closing after each test keeps
    the suite quiet and its memory flat.
    """
    yield
    plt.close('all')
