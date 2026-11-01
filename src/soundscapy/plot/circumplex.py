"""Plots of projected responses on the circumplex."""

from collections.abc import Mapping
from itertools import cycle
from typing import TYPE_CHECKING, Any

import numpy as np

from soundscapy._model import _UNSET, _Unset
from soundscapy.distribution.coordinates import valid_mask
from soundscapy.plot._common import circumplex_axes, group_colours, new_axes

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from soundscapy.distribution.coordinates import Coordinates

__all__ = ["scatter"]

# Markers taken in turn on each full pass through the group colours, so every
# (colour, marker) pair is unique for up to three times as many groups as colours.
_MARKERS = ("o", "s", "^")

# Keywords of Axes.scatter that set the face colour; any one of them replaces the cycle.
_COLOUR_KEYWORDS = frozenset({"c", "color", "facecolor", "facecolors", "fc"})


def _legend_label(key: str) -> str:
    """Return a label matplotlib keeps in the legend: it drops ``""`` and ``"_..."``."""
    if key == "":
        return "(blank)"
    return repr(key) if key.startswith("_") else key


def _per_point(value: object, n_rows: int) -> bool:
    """Return whether a colour keyword's value gives one colour (or value) per row."""
    import matplotlib.colors as mcolors

    if isinstance(value, str) or mcolors.is_color_like(value):
        return False
    return np.shape(value)[:1] == (n_rows,)


def _per_point_colours(
    kwargs: Mapping[str, Any], coords: "Coordinates", *, grouped: bool
) -> dict[str, Any]:
    """Return the per-point colour keywords of ``kwargs``, cut to the valid rows."""
    options: dict[str, Any] = {}
    for keyword in _COLOUR_KEYWORDS.intersection(kwargs):
        value = kwargs[keyword]
        if not _per_point(value, coords.n()):
            continue
        if grouped:
            msg = (
                f"{keyword}= gives one colour per response, which cannot be split across "
                "the groups of by=; pass a single colour, or select each group and plot "
                "it separately"
            )
            raise ValueError(msg)
        options[keyword] = np.asarray(value)[valid_mask(coords)]
    return options


def scatter(
    coords: "Coordinates",
    *,
    by: str | _Unset = _UNSET,
    ax: "Axes | None" = None,
    label_language: str | None = None,
    **kwargs: Any,  # forwarded unchanged to matplotlib's Axes.scatter
) -> "Axes":
    """Draw projected responses on the circumplex, one series per group.

    Only valid rows (see :func:`~soundscapy.distribution.coordinates.valid_mask`) are
    drawn. Groups are drawn in sorted key order, each labelled with its key in a legend:
    a blank key is labelled ``(blank)`` and a key starting with ``_`` is shown quoted
    (``'_x'``), as matplotlib would otherwise leave both out of the legend. Group
    colours follow matplotlib's colour cycle (``axes.prop_cycle``), so a style sheet,
    ``rc_context`` or ``style.context`` the user applies sets them; while the cycle is
    matplotlib's default, the colour-vision-deficiency-safe ``petroff10`` style is used.
    On each full pass through the colours the marker changes (``"o"``, ``"s"``, ``"^"``,
    in turn), so three times as many groups as colours stay distinguishable (30 with
    ``petroff10``). A ``c``, ``color`` or ``facecolor`` keyword replaces the colours, and a
    ``marker`` keyword the markers, for every group. A group with no valid rows still gets
    an empty labelled series, so each group keeps its colour and marker across plots of
    the same groups.

    Parameters
    ----------
    coords : Coordinates
        The projected responses; their circumplex gives the axes and labels.
    by : str, optional
        An id role (such as ``"location"``), ``"language"`` or a label column to group
        by; left unset or ``""``, all responses form one series.
    ax : matplotlib.axes.Axes, optional
        The Axes to draw on; by default the Axes of a new pyplot figure, which a notebook
        shows and ``plt.savefig`` saves.
    label_language : str, optional
        ISO 639-3 code of the axis and quadrant labels; by default the circumplex's
        reference language. A language with no translation falls back to it.
    **kwargs
        Passed to each ``Axes.scatter`` call.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on. ``show()`` is never called; a notebook displays the figure, and
        a script calls ``plt.show()`` or ``plt.savefig()``.

    Raises
    ------
    MappingError
        If ``by`` names no id role, label column or ``"language"``.
    ValueError
        If ``by`` is set and a colour keyword gives one colour per response.

    References
    ----------
    The circumplex scatter view: [@mitchell2022how].

    Examples
    --------
    >>> import soundscapy as sspy
    >>> coords = sspy.project(sspy.load_isd(subset=True), angles="equal")
    >>> ax = sspy.plot.scatter(coords, by="location")
    """
    group_by = "" if isinstance(by, _Unset) else by
    per_point = _per_point_colours(kwargs, coords, grouped=bool(group_by))
    groups = coords.split(group_by) if group_by else {"": coords}
    if ax is None:  # after the arguments are checked, so a refused call leaves no figure
        ax = new_axes()
    colours = group_colours()
    styles = ((colour, marker) for marker in cycle(_MARKERS) for colour in colours)
    for (key, group), (colour, marker) in zip(groups.items(), styles, strict=False):
        xy = group.xy[valid_mask(group)]
        options: dict[str, Any] = {"marker": marker, **kwargs, **per_point}
        if _COLOUR_KEYWORDS.isdisjoint(kwargs):
            options["color"] = colour
        if group_by:
            options["label"] = _legend_label(key)
        ax.scatter(xy[:, 0], xy[:, 1], **options)
    circumplex_axes(ax, coords.circumplex, label_language)
    if group_by and groups:
        ax.legend()
    return ax
