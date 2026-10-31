"""Plots of projected responses on the circumplex."""

from itertools import cycle
from typing import TYPE_CHECKING, Any

from soundscapy._model import _UNSET, _Unset
from soundscapy.distribution.coordinates import valid_mask
from soundscapy.plot._common import circumplex_axes, new_axes

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from soundscapy.distribution.coordinates import Coordinates

__all__ = ["scatter"]

# Markers taken in turn on each full pass through the colour sequence, so every
# (colour, marker) pair is unique for up to 30 groups.
_MARKERS = ("o", "s", "^")


def scatter(
    coords: "Coordinates",
    *,
    by: str | _Unset = _UNSET,
    ax: "Axes | None" = None,
    label_language: str | None = None,
    **kwargs: Any,  # noqa: ANN401  # forwarded unchanged to matplotlib's Axes.scatter
) -> "Axes":
    """Draw projected responses on the circumplex, one series per group.

    Only valid rows (see :func:`~soundscapy.distribution.coordinates.valid_mask`) are
    drawn. Groups are drawn in sorted key order, each labelled with its key in a legend.
    Group colours follow matplotlib's ``petroff10`` colour sequence; on each full pass
    through its ten colours the marker changes (``"o"``, ``"s"``, ``"^"``, in turn), so
    up to 30 groups stay distinguishable. A ``color`` or ``marker`` keyword overrides
    this cycle for every group. A group with no valid rows still gets an empty labelled
    series, so each group keeps its colour and marker across plots of the same groups.

    Parameters
    ----------
    coords : Coordinates
        The projected responses; their circumplex gives the axes and labels.
    by : str, optional
        An id role (such as ``"location"``), ``"language"`` or a label column to group
        by; left unset or ``""``, all responses form one series.
    ax : matplotlib.axes.Axes, optional
        The Axes to draw on; by default the Axes of a new pyplot-free figure.
    label_language : str, optional
        ISO 639-3 code of the axis and quadrant labels; by default the circumplex's
        reference language. A language with no translation falls back to it.
    **kwargs
        Passed to each ``Axes.scatter`` call.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes drawn on. ``show()`` is never called: in a notebook, end the cell with
        ``ax.figure`` to display it.

    Raises
    ------
    MappingError
        If ``by`` names no id role, label column or ``"language"``.

    References
    ----------
    The circumplex scatter view: [@mitchell2022how].

    Examples
    --------
    >>> import soundscapy as sspy
    >>> coords = sspy.project(sspy.load_isd(subset=True), angles="equal")
    >>> ax = sspy.plot.scatter(coords, by="location")
    """
    import matplotlib as mpl  # noqa: PLC0415  # matplotlib is imported lazily

    if ax is None:
        ax = new_axes()
    group_by = "" if isinstance(by, _Unset) else by
    groups = coords.split(group_by) if group_by else {"": coords}
    colours = mpl.color_sequences["petroff10"]
    styles = ((colour, marker) for marker in cycle(_MARKERS) for colour in colours)
    for (key, group), (colour, marker) in zip(groups.items(), styles, strict=False):
        xy = group.xy[valid_mask(group)]
        options: dict[str, Any] = {"color": colour, "marker": marker, **kwargs}
        if group_by:
            options["label"] = key
        ax.scatter(xy[:, 0], xy[:, 1], **options)
    circumplex_axes(ax, coords.circumplex, label_language)
    if group_by:
        ax.legend()
    return ax
