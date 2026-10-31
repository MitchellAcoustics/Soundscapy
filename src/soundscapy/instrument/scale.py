"""The response scale of an instrument."""

from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import StrictFloat, model_validator

from soundscapy._model import FrozenModel, StrTuplePairs

__all__ = ["Scale"]


class Scale(FrozenModel):
    """A response scale.

    Parameters
    ----------
    minimum : float
        Lowest response value.
    maximum : float
        Highest response value; greater than ``minimum``.
    kind : {"ordinal", "interval"}, optional
        Ordinal scales need integer-valued bounds and accept only integer responses.
    labels : mapping or sequence of pairs, optional
        Per ISO 639-3 language code, one label per integer step from ``minimum`` to
        ``maximum``. Read from and written as a table of arrays.
    """

    minimum: StrictFloat
    maximum: StrictFloat
    kind: Literal["ordinal", "interval"] = "ordinal"
    labels: StrTuplePairs = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.minimum >= self.maximum:
            msg = f"minimum {self.minimum} must be less than maximum {self.maximum}"
            raise ValueError(msg)
        if self.kind == "ordinal" and not (
            float(self.minimum).is_integer() and float(self.maximum).is_integer()
        ):
            msg = f"an ordinal scale needs integer bounds, got {self.minimum} and {self.maximum}"
            raise ValueError(msg)
        if self.labels:
            steps = int(self.maximum - self.minimum) + 1
            for language, texts in self.labels:
                if len(texts) != steps:
                    msg = (
                        f"labels for {language!r} hold {len(texts)} entries; "
                        f"the scale has {steps} integer steps"
                    )
                    raise ValueError(msg)
        return self

    @property
    def midpoint(self) -> float:
        """Centre of the scale, ``(minimum + maximum) / 2``."""
        return (self.minimum + self.maximum) / 2

    @property
    def half_range(self) -> float:
        """Half the width of the scale, ``(maximum - minimum) / 2``."""
        return (self.maximum - self.minimum) / 2

    @property
    def range(self) -> float:
        """Width of the scale, ``maximum - minimum``."""
        return self.maximum - self.minimum

    def contains(self, values: ArrayLike) -> NDArray[np.bool_]:
        """Test which values are valid responses on this scale.

        Parameters
        ----------
        values : array_like of float
            Response values; NaN and infinities are never contained.

        Returns
        -------
        numpy.ndarray of bool
            True where the value is finite, within ``[minimum, maximum]`` and, for an
            ordinal scale, integer-valued.
        """
        arr = np.asarray(values, dtype=np.float64)
        ok = np.isfinite(arr) & (arr >= self.minimum) & (arr <= self.maximum)
        if self.kind == "ordinal":
            ok &= arr == np.floor(arr)  # NaN/inf are already False above
        return ok
