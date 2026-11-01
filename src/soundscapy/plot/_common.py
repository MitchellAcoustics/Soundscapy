"""Helpers shared by the plot functions: new Axes, group colours and the circumplex frame."""

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from soundscapy.instrument import Circumplex

__all__ = ["circumplex_axes", "group_colours", "new_axes"]

DEFAULT_STYLE: Final = "petroff10"
"""matplotlib's colour-vision-deficiency-safe style (matplotlib ≥ 3.10), the default (FR-8.3)."""

# Text placement in data coordinates: the axis-end labels sit just inside the ±1 square,
# the quadrant labels near its corners.
_AXIS_END = 0.98
_CORNER = 0.95
_SIGN = {"+": 1.0, "-": -1.0}
_TEXT_COLOUR = "0.3"


def _user_styled() -> bool:
    """Return whether the active colour cycle differs from matplotlib's own default.

    A style sheet, ``rc_context`` or ``style.context`` the user applied changes it; then the
    user's colours are used, otherwise soundscapy's default style (FR-8.3).
    """
    import matplotlib as mpl

    return mpl.rcParams["axes.prop_cycle"] != mpl.rcParamsDefault["axes.prop_cycle"]


def group_colours() -> list[str]:
    """Return the colours groups are drawn in, in order.

    Returns
    -------
    list of str
        The ``color`` entries of the active ``axes.prop_cycle`` when the user has styled
        matplotlib, otherwise those of the ``petroff10`` style.
    """
    import matplotlib as mpl
    import matplotlib.style

    if _user_styled():
        colours = mpl.rcParams["axes.prop_cycle"].by_key().get("color")
        if colours:
            return [str(colour) for colour in colours]
    with matplotlib.style.context(DEFAULT_STYLE):
        return [str(colour) for colour in mpl.rcParams["axes.prop_cycle"].by_key()["color"]]


def new_axes() -> "Axes":
    """Return the Axes of a new pyplot figure.

    The figure is created through pyplot, as matplotlib's pattern for libraries does, so a
    notebook shows it and ``plt.show()`` or ``plt.savefig()`` reach it; the plot functions
    never call ``show()``. Unless the user has styled matplotlib, the Axes is created under
    the ``petroff10`` style, so anything drawn on it later follows the same colours.

    Returns
    -------
    matplotlib.axes.Axes
        The one subplot of a new figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.style

    if _user_styled():
        return plt.subplots()[1]
    with matplotlib.style.context(DEFAULT_STYLE):
        return plt.subplots()[1]


def circumplex_axes(ax: "Axes", circumplex: "Circumplex", label_language: str | None) -> None:
    """Draw the circumplex frame: the ±1 square, the axis lines and their labels.

    At each end of each axis the label of the attribute there is written, for the x axis
    then the y axis; the axis ids are the axis titles, and each quadrant carries the
    label of its quadrant position: the circumplex layout of [@iso12913-3-2019].

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes to draw on.
    circumplex : Circumplex
        The instrument whose axes and labels are drawn.
    label_language : str or None
        ISO 639-3 code of the labels; ``None`` means the circumplex's reference
        language. A language with no translation falls back to the reference language.

    References
    ----------
    [@iso12913-3-2019]
    """
    language = label_language or ""
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.axhline(0.0, color="0.5", linewidth=0.8, zorder=0)
    ax.axvline(0.0, color="0.5", linewidth=0.8, zorder=0)
    x_axis, y_axis = circumplex.axes
    ax.set_xlabel(x_axis.id)
    ax.set_ylabel(y_axis.id)
    ends = (
        (x_axis.positive, (_AXIS_END, 0.0), "right", "bottom"),
        (x_axis.negative, (-_AXIS_END, 0.0), "left", "bottom"),
        (y_axis.positive, (0.0, _AXIS_END), "left", "top"),
        (y_axis.negative, (0.0, -_AXIS_END), "left", "bottom"),
    )
    for position, (x, y), ha, va in ends:
        label = circumplex.label(position, language)
        ax.text(x, y, label, ha=ha, va=va, color=_TEXT_COLOUR, fontsize="small")
    for key, position in circumplex.quadrants:
        x, y = _SIGN[key[0]] * _CORNER, _SIGN[key[1]] * _CORNER
        ax.text(
            x,
            y,
            circumplex.label(position, language),
            ha="right" if x > 0 else "left",
            va="top" if y > 0 else "bottom",
            color=_TEXT_COLOUR,
            fontsize="small",
        )
