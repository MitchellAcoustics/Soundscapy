"""Helpers shared by the plot functions: the circumplex axes and new Axes."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from soundscapy.instrument import Circumplex

__all__ = ["circumplex_axes", "new_axes"]

# Text placement in data coordinates: the axis-end labels sit just inside the ±1 square,
# the quadrant labels near its corners.
_AXIS_END = 0.98
_CORNER = 0.95
_SIGN = {"+": 1.0, "-": -1.0}
_TEXT_COLOUR = "0.3"


def new_axes() -> "Axes":
    """Return the Axes of a new pyplot-free figure.

    Returns
    -------
    matplotlib.axes.Axes
        One subplot of a new ``matplotlib.figure.Figure``; no pyplot state is touched,
        so Jupyter shows it only through the figure's rich display.
    """
    from matplotlib.figure import Figure

    return Figure().add_subplot()


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
