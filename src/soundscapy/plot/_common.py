"""Helpers shared by the plot functions: new Axes, group colours and the circumplex frame."""

from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.typing import ColorType

    from soundscapy.instrument import Circumplex

__all__ = ["circumplex_axes", "group_colours", "new_axes"]

DEFAULT_PALETTE: Final = "colorblind"
"""The seaborn palette that gives the default group colours: its colour-blind palette (FR-8.3)."""

# Text placement in data coordinates: the axis-end labels sit just inside the ±1 square,
# the quadrant labels near its corners.
_AXIS_END = 0.98
_CORNER = 0.95
_SIGN = {"+": 1.0, "-": -1.0}
_TEXT_COLOUR = "0.3"


def _default_colours() -> list[str]:
    """Return seaborn's ``DEFAULT_PALETTE`` as hex strings, in seaborn's order.

    seaborn ships no type information and returns RGB float triples; ``to_hex`` gives typed,
    exact values (seaborn's floats are its 8-bit hex values divided by 255).
    """
    import matplotlib.colors as mcolors
    import seaborn as sns

    return [mcolors.to_hex(rgb) for rgb in sns.color_palette(DEFAULT_PALETTE)]


def _user_cycle_is_set() -> bool:
    """Return whether a style sheet, ``rc_context`` or ``style.context`` changed the cycle.

    The user's styling is detected as a change of ``axes.prop_cycle`` away from matplotlib's
    own default, so a user who sets that default cycle explicitly cannot be told apart.
    """
    import matplotlib as mpl

    return mpl.rcParams["axes.prop_cycle"] != mpl.rcParamsDefault["axes.prop_cycle"]


def _style_context() -> AbstractContextManager[object]:
    """Return the context soundscapy's plots are drawn in: the user's style, or the default.

    When the user has styled matplotlib (see ``_user_cycle_is_set``), nothing is changed.
    Otherwise ``axes.prop_cycle`` is set to seaborn's colour-blind palette inside an
    ``rc_context``, which restores the user's rcParams on exit (FR-8.3).
    """
    import matplotlib as mpl
    import matplotlib.rcsetup

    if _user_cycle_is_set():
        return nullcontext()
    return mpl.rc_context({"axes.prop_cycle": matplotlib.rcsetup.cycler(color=_default_colours())})


def group_colours() -> list["ColorType"]:
    """Return the colours groups are drawn in, in order.

    Returns
    -------
    list of matplotlib colour
        The ``color`` entries of the active ``axes.prop_cycle``, as given, when the user has
        styled matplotlib, otherwise seaborn's colour-blind palette as hex strings. A user
        cycle without colours falls back to seaborn's colour-blind palette.
    """
    import matplotlib as mpl

    colours = (
        mpl.rcParams["axes.prop_cycle"].by_key().get("color") if _user_cycle_is_set() else None
    )
    return list(colours or _default_colours())


def new_axes() -> "Axes":
    """Return the Axes of a new pyplot figure.

    The figure is created through pyplot, as matplotlib's pattern for libraries does, so a
    notebook shows it and ``plt.show()`` or ``plt.savefig()`` reach it; the plot functions
    never call ``show()``. Unless the user has styled matplotlib, the Axes is created with
    seaborn's colour-blind palette as its colour cycle, so anything drawn on it later follows
    the same colours.

    Returns
    -------
    matplotlib.axes.Axes
        The one subplot of a new figure.
    """
    import matplotlib.pyplot as plt

    with _style_context():
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
