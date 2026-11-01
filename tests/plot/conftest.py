"""matplotlib set-up for the plot tests."""

from collections.abc import Iterator

# Before pyplot is imported: the plot tests run headless however they are started (task,
# plain pytest, an IDE), so the backend is fixed here, not by an environment variable.
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Close every pyplot figure a test made; plot functions create them for ax=None."""
    yield
    plt.close("all")
