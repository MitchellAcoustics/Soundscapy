"""Projecting a survey's attribute scores into circumplex ``Coordinates``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from soundscapy._model import (
    _UNSET,
    COL_ANGLE_SET,
    COL_LANGUAGE,
    COL_RESPONSE_ID,
    FrozenModel,
    _Unset,
)
from soundscapy.distribution.coordinates import Coordinates, OriginSegment
from soundscapy.errors import SettingsError
from soundscapy.instrument import circumplex
from soundscapy.instrument.angles import AngleSelection
from soundscapy.projection.formulas import _FORMULATIONS, Formulation, project_scores
from soundscapy.provenance import envelope, hash_json

if TYPE_CHECKING:
    from soundscapy.survey.read import Survey

__all__ = ["ProjectionSettings", "project"]

_EQUAL: Final = "equal"
# Provenance keys a projection carries over from the survey's envelope unchanged.
_CARRIED: Final = (
    "input_hash",
    "mapping_hash",
    "data_refs",
    "response_id_source",
    "read",
    "id_columns",
    "response_id_columns",
    "response_key_blank_ok",
)


class ProjectionSettings(FrozenModel):
    """Settings of ``project()``.

    Attributes
    ----------
    formulation : {"canonical", "iso2019"}, optional
        The projection formula; see ``soundscapy.projection.formulas.project_scores()``.
    angles : AngleSelection, optional
        The angle set each response is projected with.
    """

    formulation: Formulation = "canonical"
    angles: AngleSelection = AngleSelection()


def _quoted(values: tuple[str, ...]) -> str:
    return ", ".join(f'"{value}"' for value in values)


def _settings(
    formulation: Formulation | _Unset,
    angles: Literal["equal"] | AngleSelection | _Unset,
) -> ProjectionSettings:
    """Check the arguments in soundscapy's own words and build the settings."""
    fields: dict[str, object] = {}
    if formulation is not _UNSET:
        if formulation not in _FORMULATIONS:
            msg = (
                f"formulation must be one of {_quoted(_FORMULATIONS)} in this version, "
                f"got {formulation!r}"
            )
            raise SettingsError(msg)
        fields["formulation"] = formulation
    if isinstance(angles, AngleSelection):
        fields["angles"] = angles
    elif angles is not _UNSET:
        if angles != _EQUAL:
            msg = (
                f'angles accepts only "{_EQUAL}" or an AngleSelection in this version, '
                f"got {angles!r}"
            )
            hint = "named and per-language angle sets arrive in a later version"
            raise SettingsError(msg, hint=hint)
        fields["angles"] = AngleSelection(default=_EQUAL)
    return ProjectionSettings.model_validate(fields)


def project(
    survey: Survey,
    *,
    formulation: Formulation | _Unset = _UNSET,
    angles: Literal["equal"] | AngleSelection | _Unset = _UNSET,
) -> Coordinates:
    """Project a survey's attribute scores onto the two circumplex axes.

    Only included responses are projected; excluded ones get NaN coordinates and an
    empty ``ssp_angle_set`` label. Every response uses the circumplex's nominal angles.
    The scale is the mapping's declared attribute scale, never the data's range. The
    status labels are the survey's own, and the survey is unchanged.

    Parameters
    ----------
    survey : Survey
        The survey, from ``read_survey()``.
    formulation : {"canonical", "iso2019"}, optional
        The projection formula; left unset, ``"canonical"``.
    angles : "equal" or AngleSelection, optional
        The angles; left unset, ``"equal"`` (the nominal angles).

    Returns
    -------
    Coordinates
        One row per survey response, in the survey's order, with the settings used and
        the provenance envelope.

    Raises
    ------
    SettingsError
        If ``formulation`` or ``angles`` is not a value this version accepts.
    FormulationError
        If ``"iso2019"`` is used with a scale other than 1 to 5.

    References
    ----------
    The canonical formulation: [@mitchell2026translation], proof p. 23, §4.5. The 2019
    formulation: [@iso12913-3-2019], as written in [@mitchell2022how], Eqs. 1 and 2.
    """
    settings = _settings(formulation, angles)
    circ = circumplex(survey.mapping.instrument, survey.mapping.instrument_version)
    scale = survey.mapping.groups.attributes.scale
    scores = survey.scores["attributes"]
    status = survey.status.label_arrays()
    included = survey.status.included

    n_included = int(included.sum())
    angles_deg = np.broadcast_to(
        np.asarray(circ.angles_deg, dtype=np.float64), (n_included, scores.shape[1])
    )
    xy = np.full((survey.n(), 2), np.nan, dtype=np.float64)
    xy[included] = project_scores(scores[included], angles_deg, scale, settings.formulation)

    angle_set = np.where(included, _EQUAL, "").astype(object)
    labels = {
        COL_RESPONSE_ID: survey.response_id,
        **status,
        COL_LANGUAGE: survey.language,
        COL_ANGLE_SET: angle_set,
        **{getattr(survey.mapping.ids, role): values for role, values in survey.ids.items()},
    }

    # read_survey wrote these keys with the types envelope takes; Any passes them on as is.
    carried: dict[str, Any] = {key: survey.provenance[key] for key in _CARRIED}
    provenance = envelope(operation="project", settings=settings, **carried)
    segment = OriginSegment(
        kind="survey",
        dataset=survey.source.name,
        n=survey.n(),
        input_hash=carried["input_hash"],
        mapping_hash=carried["mapping_hash"],
        effective_scale=scale,
        angle_sets=(_EQUAL,),
        settings_hash=hash_json(settings),
    )
    return Coordinates(
        xy=xy,
        circumplex=circ,
        labels=labels,
        origin=(segment,),
        settings=settings,
        provenance=provenance,
    )
