"""plot.scatter: responses on the circumplex, asserted on artist data, never on pixels."""

import dataclasses
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from soundscapy import plot
from soundscapy.distribution.coordinates import Coordinates, valid_mask

LABELS = ("pleasant", "annoying", "eventful", "uneventful")
QUADRANTS = ("vibrant", "chaotic", "monotonous", "calm")


def _collections(ax) -> list[PathCollection]:
    return [c for c in ax.collections if isinstance(c, PathCollection)]


def _style(collection) -> tuple[tuple[float, ...], bytes]:
    """Return the (facecolor, marker path vertices) pair that tells series apart."""
    face = tuple(float(v) for v in collection.get_facecolor()[0])
    return face, collection.get_paths()[0].vertices.tobytes()


def test_offsets_equal_valid_xy(isd_coords):
    ax = plot.scatter(isd_coords)
    (collection,) = _collections(ax)
    # Exact: the valid rows are passed to matplotlib unchanged.
    np.testing.assert_array_equal(collection.get_offsets(), isd_coords.xy[valid_mask(isd_coords)])


def test_grouped_offsets(isd_coords):
    ax = plot.scatter(isd_coords, by="location")
    groups = isd_coords.split("location")
    expected = np.concatenate([g.xy[valid_mask(g)] for g in groups.values()])
    offsets = np.concatenate([c.get_offsets() for c in _collections(ax)])
    # Exact: each group's valid rows are passed to matplotlib unchanged, in sorted key order.
    np.testing.assert_array_equal(offsets, expected)
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == list(groups)


def test_default_colours_are_petroff10(isd_coords):
    ax = plot.scatter(isd_coords, by="location")
    petroff = mpl.color_sequences["petroff10"]
    faces = [c.get_facecolor()[0] for c in _collections(ax)]
    for i, face in enumerate(faces):
        # Exact: to_rgba of the sequence entry is the colour matplotlib stores.
        np.testing.assert_array_equal(face, to_rgba(petroff[i % len(petroff)]))


def test_ungrouped_colour_is_the_first_petroff10(isd_coords):
    ax = plot.scatter(isd_coords)
    (collection,) = _collections(ax)
    # Exact: to_rgba of the sequence entry is the colour matplotlib stores.
    np.testing.assert_array_equal(
        collection.get_facecolor()[0], to_rgba(mpl.color_sequences["petroff10"][0])
    )


def test_more_than_ten_groups_are_distinct(isd_coords):
    ax = plot.scatter(isd_coords, by="location")
    styles = [_style(c) for c in _collections(ax)]
    assert len(styles) == 15
    assert len(set(styles)) == 15
    legend = ax.get_legend()
    assert legend is not None
    assert [_style(h) for h in legend.legend_handles] == styles


def test_group_with_no_valid_rows_keeps_an_empty_series(isd_coords):
    included = isd_coords.labels["ssp_included"].copy()
    first = min(isd_coords.split("location"))
    location = isd_coords.provenance["id_columns"]["location"]
    included[isd_coords.labels[location] == first] = False
    hidden = Coordinates(
        xy=isd_coords.xy,
        circumplex=isd_coords.circumplex,
        labels={**isd_coords.labels, "ssp_included": included},
        origin=isd_coords.origin,
        settings=isd_coords.settings,
        provenance=isd_coords.provenance,
    )
    styles = [_style(c) for c in _collections(plot.scatter(isd_coords, by="location"))]
    collections = _collections(plot.scatter(hidden, by="location"))
    assert len(collections) == 15
    assert np.asarray(collections[0].get_offsets()).shape == (0, 2)
    assert collections[0].get_label() == first
    assert [_style(c) for c in collections[1:]] == styles[1:]


def test_no_pyplot_figure_created(isd_coords):
    plt.close("all")
    ax = plot.scatter(isd_coords, by="location")
    assert plt.get_fignums() == []
    assert isinstance(ax.figure, Figure)


def test_returns_the_given_ax(isd_coords):
    ax = Figure().add_subplot()
    assert plot.scatter(isd_coords, ax=ax) is ax


def test_kwargs_reach_artist(isd_coords):
    ax = plot.scatter(isd_coords, by="location", color="k", marker="x")
    for collection in _collections(ax):
        # Exact: "k" is stored as pure black.
        np.testing.assert_array_equal(collection.get_facecolor()[0], to_rgba("k"))
    assert len({np.asarray(c.get_paths()[0].vertices).tobytes() for c in _collections(ax)}) == 1


def test_axis_and_quadrant_labels(isd_coords):
    ax = plot.scatter(isd_coords)
    texts = {t.get_text() for t in ax.texts}
    assert set(LABELS) | set(QUADRANTS) <= texts
    assert ax.get_xlabel() == "ISOPleasant"
    assert ax.get_ylabel() == "ISOEventful"
    # Exact: the limits are set to these values, not computed.
    assert ax.get_xlim() == (-1.0, 1.0)
    assert ax.get_ylim() == (-1.0, 1.0)
    # The S1 circumplex file ships no translations, so German falls back to English.
    german = plot.scatter(isd_coords, label_language="deu")
    assert {t.get_text() for t in german.texts} == texts


@pytest.mark.parametrize("suffix", ["png", "svg", "pdf"])
def test_saves_png_svg_pdf(isd_coords, tmp_path: Path, suffix):
    path = tmp_path / f"scatter.{suffix}"
    figure = plot.scatter(isd_coords, by="location").figure
    assert isinstance(figure, Figure)
    figure.savefig(path)
    assert path.stat().st_size > 0


def test_c_keyword_replaces_the_colour_cycle(isd_coords):
    ax = plot.scatter(isd_coords, by="location", c="k")
    collections = _collections(ax)
    assert len(collections) == 15
    for collection in collections:
        # Exact: "k" is stored as pure black.
        np.testing.assert_array_equal(collection.get_facecolor()[0], to_rgba("k"))


def _relabel(isd_coords, location: str, rows: slice) -> Coordinates:
    """Return ``isd_coords`` with the location label of ``rows`` set to ``location``."""
    column = isd_coords.provenance["id_columns"]["location"]
    values = isd_coords.labels[column].copy()
    values[rows] = location
    return dataclasses.replace(isd_coords, labels={**isd_coords.labels, column: values})


def _legend_texts(ax) -> list[str]:
    legend = ax.get_legend()
    assert legend is not None
    return [t.get_text() for t in legend.get_texts()]


def test_blank_group_key_is_labelled_blank(isd_coords):
    ax = plot.scatter(_relabel(isd_coords, "", slice(0, 5)), by="location")
    texts = _legend_texts(ax)
    assert texts[0] == "(blank)"
    assert len(texts) == len(_collections(ax)) == 16


def test_underscore_group_key_is_shown_with_repr(isd_coords):
    ax = plot.scatter(_relabel(isd_coords, "_hidden", slice(0, 5)), by="location")
    texts = _legend_texts(ax)
    assert repr("_hidden") in texts
    assert len(texts) == len(_collections(ax)) == 16


def test_by_ssp_reasons_labels_the_included_points(isd_coords):
    ax = plot.scatter(isd_coords, by="ssp_reasons")
    assert _legend_texts(ax) == ["(blank)", "incomplete_attributes"]
    # Exact: the blank-reason group is every included row, passed unchanged.
    np.testing.assert_array_equal(
        _collections(ax)[0].get_offsets(), isd_coords.xy[valid_mask(isd_coords)]
    )


def test_by_ssp_flags_draws_one_labelled_group(isd_coords):
    ax = plot.scatter(isd_coords, by="ssp_flags")
    assert _legend_texts(ax) == ["(blank)"]


def test_empty_coordinates_draw_without_legend_or_warning(isd_coords):
    empty = isd_coords.select(np.zeros(isd_coords.n(), dtype=bool))
    ax = plot.scatter(empty, by="location")
    assert _collections(ax) == []
    assert ax.get_legend() is None
