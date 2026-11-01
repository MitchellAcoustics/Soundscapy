"""plot.scatter: responses on the circumplex, asserted on artist data, never on pixels."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import pytest
import seaborn as sns
from cycler import cycler
from matplotlib.collections import PathCollection
from matplotlib.colors import to_hex, to_rgba
from matplotlib.figure import Figure

from soundscapy import plot
from soundscapy.distribution.coordinates import Coordinates, valid_mask
from soundscapy.errors import MappingError

if TYPE_CHECKING:
    from pathlib import Path

LABELS = ("pleasant", "annoying", "eventful", "uneventful")
QUADRANTS = ("vibrant", "chaotic", "monotonous", "calm")

# The default is seaborn's colour-blind palette itself (spec §6.15), so seaborn is the oracle.
COLORBLIND = [to_hex(rgb) for rgb in sns.color_palette("colorblind")]


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


def test_default_colours_are_seaborn_colorblind(isd_coords):
    ax = plot.scatter(isd_coords, by="location")
    faces = [c.get_facecolor()[0] for c in _collections(ax)]
    assert len(faces) == 15
    for i, face in enumerate(faces):
        # Exact: to_rgba of the palette entry is the colour matplotlib stores; the eleventh
        # group starts the palette again.
        np.testing.assert_array_equal(face, to_rgba(COLORBLIND[i % len(COLORBLIND)]))


def test_ungrouped_colour_is_the_first_default_colour(isd_coords):
    ax = plot.scatter(isd_coords)
    (collection,) = _collections(ax)
    # Exact: to_rgba of the palette entry is the colour matplotlib stores.
    np.testing.assert_array_equal(collection.get_facecolor()[0], to_rgba(COLORBLIND[0]))


def test_the_default_cycle_does_not_leak(isd_coords):
    # No hidden global state: the default palette is applied in an rc_context that restores
    # rcParams, so later plots by the user or by another library see matplotlib's own cycle.
    before = mpl.rcParams["axes.prop_cycle"]
    plot.scatter(isd_coords, by="location")
    assert mpl.rcParams["axes.prop_cycle"] == before == mpl.rcParamsDefault["axes.prop_cycle"]
    (line,) = plt.figure().add_subplot().plot([0.0, 0.1], [0.0, 0.1])
    assert to_hex(line.get_color()) == to_hex(before.by_key()["color"][0])


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


def test_ax_none_creates_one_pyplot_figure(isd_coords):
    # matplotlib's pattern for libraries: a notebook shows it, plt.savefig reaches it.
    plt.close("all")
    ax = plot.scatter(isd_coords, by="location")
    assert plt.get_fignums() == [plt.gcf().number]
    assert plt.gca() is ax


def test_created_axes_carries_the_default_cycle(isd_coords):
    # Lines the user adds afterwards follow the same colours.
    ax = plot.scatter(isd_coords)
    (line,) = ax.plot([0.0, 0.1], [0.0, 0.1])
    assert to_hex(line.get_color()) == COLORBLIND[0]


def test_user_style_sets_the_group_colours(isd_coords):
    with mpl.style.context("ggplot"):
        ggplot = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        ax = plot.scatter(isd_coords, by="location")
    faces = [c.get_facecolor()[0] for c in _collections(ax)]
    for i, face in enumerate(faces):
        # Exact: to_rgba of the cycle entry is the colour matplotlib stores.
        np.testing.assert_array_equal(face, to_rgba(ggplot[i % len(ggplot)]))


def test_a_short_user_cycle_keeps_groups_distinct(isd_coords):
    with mpl.rc_context({"axes.prop_cycle": cycler(color=["k", "r", "b", "g", "m"])}):
        ax = plot.scatter(isd_coords, by="location")
    styles = [_style(c) for c in _collections(ax)]
    assert len(styles) == len(set(styles)) == 15


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


def test_per_point_colours_follow_the_valid_rows(isd_coords):
    n = isd_coords.n()
    valid = valid_mask(isd_coords)
    assert not valid.all()
    values = np.arange(n, dtype=float)
    (collection,) = _collections(plot.scatter(isd_coords, c=values))
    # Exact: the valid rows' values are passed to matplotlib unchanged.
    np.testing.assert_array_equal(collection.get_array(), values[valid])
    faces = np.array([to_rgba("k"), to_rgba("r")])[np.arange(n) % 2]
    (collection,) = _collections(plot.scatter(isd_coords, facecolors=faces))
    # Exact: to_rgba values are the colours matplotlib stores.
    np.testing.assert_array_equal(collection.get_facecolor(), faces[valid])


def test_per_point_colours_with_by_raise(isd_coords):
    n = isd_coords.n()
    with pytest.raises(ValueError, match=r"c=.*one colour per response") as info:
        plot.scatter(isd_coords, by="location", c=np.zeros(n))
    assert "single colour" in str(info.value)
    with pytest.raises(ValueError, match=r"color=.*one colour per response"):
        plot.scatter(isd_coords, by="location", color=["k"] * n)


def test_single_colour_with_by_is_not_per_point(isd_coords):
    # An RGB tuple is one colour, whatever its length.
    ax = plot.scatter(isd_coords, by="location", color=(1.0, 0.0, 0.0))
    for collection in _collections(ax):
        # Exact: to_rgba of the tuple is the colour matplotlib stores.
        np.testing.assert_array_equal(collection.get_facecolor()[0], to_rgba((1.0, 0.0, 0.0)))


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


def test_a_user_cycle_without_colours_falls_back_to_the_default_colours(isd_coords):
    with mpl.rc_context({"axes.prop_cycle": cycler(linestyle=["-", "--"])}):
        ax = plot.scatter(isd_coords)
    (collection,) = _collections(ax)
    # Exact: to_rgba of the palette entry is the colour matplotlib stores.
    np.testing.assert_array_equal(collection.get_facecolor()[0], to_rgba(COLORBLIND[0]))


def test_a_user_cycle_of_rgb_tuples_is_drawn_as_given(isd_coords):
    rgb = [(1.0, 0.0, 0.0), (0.0, 0.5, 0.0)]
    with mpl.rc_context({"axes.prop_cycle": cycler(color=rgb)}):
        ax = plot.scatter(isd_coords, by="location")
    faces = [c.get_facecolor()[0] for c in _collections(ax)]
    for i, face in enumerate(faces):
        # Exact: to_rgba of the tuple is the colour matplotlib stores.
        np.testing.assert_array_equal(face, to_rgba(rgb[i % 2]))


def test_a_refused_call_leaves_no_figure(isd_coords):
    plt.close("all")
    with pytest.raises(MappingError):
        plot.scatter(isd_coords, by="no_such_column")
    assert plt.get_fignums() == []
