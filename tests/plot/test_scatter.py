"""plot.scatter: responses on the circumplex, asserted on artist data, never on pixels."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from soundscapy import plot
from soundscapy.datasets import isd
from soundscapy.distribution.coordinates import Coordinates, valid_mask
from soundscapy.projection.project import project

LABELS = ("pleasant", "annoying", "eventful", "uneventful")
QUADRANTS = ("vibrant", "chaotic", "monotonous", "calm")


@pytest.fixture(scope="module")
def coords() -> Coordinates:
    return project(isd.load(subset=True), angles="equal")


def _collections(ax) -> list[PathCollection]:
    return [c for c in ax.collections if isinstance(c, PathCollection)]


def _style(collection) -> tuple[tuple[float, ...], bytes]:
    """Return the (facecolor, marker path vertices) pair that tells series apart."""
    face = tuple(float(v) for v in collection.get_facecolor()[0])
    return face, collection.get_paths()[0].vertices.tobytes()


def test_offsets_equal_valid_xy(coords):
    ax = plot.scatter(coords)
    (collection,) = _collections(ax)
    # Exact: the valid rows are passed to matplotlib unchanged.
    np.testing.assert_array_equal(collection.get_offsets(), coords.xy[valid_mask(coords)])


def test_grouped_offsets(coords):
    ax = plot.scatter(coords, by="location")
    groups = coords.split("location")
    expected = np.concatenate([g.xy[valid_mask(g)] for g in groups.values()])
    offsets = np.concatenate([c.get_offsets() for c in _collections(ax)])
    # Exact: each group's valid rows are passed to matplotlib unchanged, in sorted key order.
    np.testing.assert_array_equal(offsets, expected)
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == list(groups)


def test_default_colours_are_petroff10(coords):
    ax = plot.scatter(coords, by="location")
    petroff = mpl.color_sequences["petroff10"]
    faces = [c.get_facecolor()[0] for c in _collections(ax)]
    for i, face in enumerate(faces):
        # Exact: to_rgba of the sequence entry is the colour matplotlib stores.
        np.testing.assert_array_equal(face, to_rgba(petroff[i % len(petroff)]))


def test_ungrouped_colour_is_the_first_petroff10(coords):
    ax = plot.scatter(coords)
    (collection,) = _collections(ax)
    # Exact: to_rgba of the sequence entry is the colour matplotlib stores.
    np.testing.assert_array_equal(
        collection.get_facecolor()[0], to_rgba(mpl.color_sequences["petroff10"][0])
    )


def test_more_than_ten_groups_are_distinct(coords):
    ax = plot.scatter(coords, by="location")
    styles = [_style(c) for c in _collections(ax)]
    assert len(styles) == 15
    assert len(set(styles)) == 15
    legend = ax.get_legend()
    assert legend is not None
    assert [_style(h) for h in legend.legend_handles] == styles


def test_group_with_no_valid_rows_keeps_an_empty_series(coords):
    included = coords.labels["ssp_included"].copy()
    first = min(coords.split("location"))
    location = coords.provenance["id_columns"]["location"]
    included[coords.labels[location] == first] = False
    hidden = Coordinates(
        xy=coords.xy,
        circumplex=coords.circumplex,
        labels={**coords.labels, "ssp_included": included},
        origin=coords.origin,
        settings=coords.settings,
        provenance=coords.provenance,
    )
    styles = [_style(c) for c in _collections(plot.scatter(coords, by="location"))]
    collections = _collections(plot.scatter(hidden, by="location"))
    assert len(collections) == 15
    assert np.asarray(collections[0].get_offsets()).shape == (0, 2)
    assert collections[0].get_label() == first
    assert [_style(c) for c in collections[1:]] == styles[1:]


def test_no_pyplot_figure_created(coords):
    plt.close("all")
    ax = plot.scatter(coords, by="location")
    assert plt.get_fignums() == []
    assert isinstance(ax.figure, Figure)


def test_returns_the_given_ax(coords):
    ax = Figure().add_subplot()
    assert plot.scatter(coords, ax=ax) is ax


def test_kwargs_reach_artist(coords):
    ax = plot.scatter(coords, by="location", color="k", marker="x")
    for collection in _collections(ax):
        # Exact: "k" is stored as pure black.
        np.testing.assert_array_equal(collection.get_facecolor()[0], to_rgba("k"))
    assert len({np.asarray(c.get_paths()[0].vertices).tobytes() for c in _collections(ax)}) == 1


def test_axis_and_quadrant_labels(coords):
    ax = plot.scatter(coords)
    texts = {t.get_text() for t in ax.texts}
    assert set(LABELS) | set(QUADRANTS) <= texts
    assert ax.get_xlabel() == "ISOPleasant"
    assert ax.get_ylabel() == "ISOEventful"
    # Exact: the limits are set to these values, not computed.
    assert ax.get_xlim() == (-1.0, 1.0)
    assert ax.get_ylim() == (-1.0, 1.0)
    # The S1 circumplex file ships no translations, so German falls back to English.
    german = plot.scatter(coords, label_language="deu")
    assert {t.get_text() for t in german.texts} == texts


@pytest.mark.parametrize("suffix", ["png", "svg", "pdf"])
def test_saves_png_svg_pdf(coords, tmp_path: Path, suffix):
    path = tmp_path / f"scatter.{suffix}"
    figure = plot.scatter(coords, by="location").figure
    assert isinstance(figure, Figure)
    figure.savefig(path)
    assert path.stat().st_size > 0
