"""Which attribute angles a projection uses."""

from typing import Literal

from soundscapy._model import FrozenModel

__all__ = ["AngleSelection"]


class AngleSelection(FrozenModel):
    """The angle set each response is projected with.

    Attributes
    ----------
    default : {"equal"}, optional
        The angle set of every response. ``"equal"`` uses the circumplex's nominal angles
        and does not consult the response's language. Named, language-specific angle sets
        are not available in this version.
    """

    default: Literal["equal"] = "equal"
