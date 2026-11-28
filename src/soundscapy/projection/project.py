"""Projecting a survey's attribute scores into circumplex ``Coordinates``."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from soundscapy import _data
from soundscapy._frames import column_text
from soundscapy._model import (
    _UNSET,
    COL_ANGLE_SET,
    COL_LANGUAGE,
    COL_RESPONSE_ID,
    FrozenModel,
    _Unset,
    settings_layer,
)
from soundscapy.distribution.coordinates import Coordinates, OriginSegment
from soundscapy.errors import (
    EqualAnglesFallbackWarning,
    LowConfidenceLanguageError,
    MappingError,
    MediumConfidenceLanguageWarning,
    SettingsError,
    UncoveredLanguageError,
    _plural,
    _shown,
)
from soundscapy.instrument import circumplex
from soundscapy.instrument.angles import AngleSelection, AngleSet, AngleSetRef, angle_set
from soundscapy.profiles import resolve_settings
from soundscapy.projection._angles import LanguageOutcome, ResolvedAngles, resolve_angles
from soundscapy.projection.formulas import _FORMULATIONS, Formulation, project_scores
from soundscapy.provenance import envelope, hash_json
from soundscapy.survey.status import Flag, Reason, Status

if TYPE_CHECKING:
    import os
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from soundscapy.instrument.angles import Policy
    from soundscapy.survey.read import Survey

__all__ = ["ProjectionSettings", "project"]

_EQUAL: Final = "equal"
# Provenance keys a projection carries over from the survey's envelope unchanged.
_CARRIED: Final = (
    "input_hash",
    "mapping_hash",
    "response_id_source",
    "read",
    "id_columns",
    "response_id_columns",
    "response_key_blank_ok",
    "quality_skipped",
)
_TIER_NAMES: Final = {"high": "High", "medium": "Medium", "low": "Low"}


class ProjectionSettings(FrozenModel):
    """Settings of ``project()``.

    Attributes
    ----------
    formulation : {"canonical", "iso2019", "aletta2024"}, optional
        The projection formula; see ``soundscapy.projection.formulas.project_scores()``.
    angles : AngleSelection, optional
        The angle set each response is projected with.
    """

    formulation: Formulation = "canonical"
    angles: AngleSelection = AngleSelection()


def _quoted(values: tuple[str, ...]) -> str:
    return ", ".join(f'"{value}"' for value in values)


def _angles_layer(angles: AngleSetRef | str | AngleSelection) -> dict[str, Any]:
    """Turn the ``angles`` keyword into one layer: a selection's set fields, or the default."""
    if isinstance(angles, AngleSelection):
        return settings_layer(angles, model=AngleSelection, name="angles")
    if isinstance(angles, AngleSetRef):
        return {"default": angles.label}
    if isinstance(angles, str):
        return {"default": angles}
    msg = (
        f'angles must be "{_EQUAL}", a "name@version" string, an AngleSetRef or an '
        f"AngleSelection, got {angles!r}"
    )
    raise SettingsError(msg)


def _keywords(
    formulation: Formulation | _Unset,
    angles: AngleSetRef | str | AngleSelection | _Unset,
    on_uncovered: Policy | _Unset,
    on_low_confidence: Policy | _Unset,
) -> dict[str, object]:
    """Gather the explicit keywords as one nested layer.

    The two policies join the ``angles`` table so every leaf merges over a profile's or a
    file's ``angles`` table rather than replacing it.
    """
    if formulation is not _UNSET and formulation not in _FORMULATIONS:
        msg = (
            f"formulation must be one of {_quoted(_FORMULATIONS)} in this version, "
            f"got {formulation!r}"
        )
        raise SettingsError(msg)
    selection: dict[str, Any] = {} if angles is _UNSET else _angles_layer(angles)
    if on_uncovered is not _UNSET:
        selection["on_uncovered"] = on_uncovered
    if on_low_confidence is not _UNSET:
        selection["on_low_confidence"] = on_low_confidence
    return {"formulation": formulation, "angles": selection or _UNSET}


def _passed_sets(angle_sets: Sequence[AngleSet] | _Unset) -> dict[str, AngleSet]:
    """Check the sets passed with ``angle_sets=`` and key them by label."""
    if angle_sets is _UNSET:
        return {}
    if isinstance(angle_sets, AngleSet | str):
        msg = f"angle_sets must be a sequence of AngleSet objects, got {type(angle_sets).__name__}"
        raise TypeError(msg)
    passed: dict[str, AngleSet] = {}
    for item in angle_sets:
        if not isinstance(item, AngleSet):
            msg = f"angle_sets must hold AngleSet objects, got {type(item).__name__}"
            raise TypeError(msg)
        if item.label in passed:
            msg = f"angle_sets lists {item.label} twice"
            raise SettingsError(msg)
        passed[item.label] = item
    return passed


def _named_sets(selection: AngleSelection, passed: Mapping[str, AngleSet]) -> dict[str, AngleSet]:
    """Every set the selection names: a passed set first, else the shipped one."""
    refs = [ref for _, ref in selection.per_language]
    if isinstance(selection.default, AngleSetRef):
        refs.append(selection.default)
    return {
        ref.label: passed.get(ref.label) or angle_set(ref.name, ref.version)
        for ref in dict.fromkeys(refs)
    }


def _check_language_mapped(survey: Survey, sets: Mapping[str, AngleSet]) -> None:
    """Refuse a named set when the mapping does not say where each response's language is."""
    language = survey.mapping.language
    if sets and not (language.column or language.value):
        msg = (
            f"angles names the angle set {_shown(sorted(sets))}, whose adjusted angles depend "
            "on each response's language, but the mapping has no [language] column or value"
        )
        hint = 'add [language] with column = "<column>" or value = "<iso 639-3 code>"'
        raise MappingError(msg, hint=hint)


def _check_blank_languages(survey: Survey, resolved: ResolvedAngles) -> None:
    if not resolved.blank_language.any():
        return
    ids = survey.response_id[resolved.blank_language].tolist()
    msg = (
        f"{len(ids)} included {_plural(len(ids), 'response')} with a blank language cannot "
        f"be projected with a named angle set: {_plural(len(ids), 'response')} {_shown(ids)}"
    )
    hint = 'code the language, pass a [language] value, or project with angles="equal"'
    raise MappingError(msg, hint=hint)


def _describe(outcome: LanguageOutcome) -> str:
    """``lang (Tier in set, no adjusted angles): N responses``, as the messages list it."""
    if outcome.tier is None:
        where = f"not in {outcome.set_label}"
    else:
        where = f"{_TIER_NAMES[outcome.tier]} in {outcome.set_label}"
        if outcome.tier_only:
            where += ", no adjusted angles"
    return f"{outcome.language} ({where}): {outcome.count} {_plural(outcome.count, 'response')}"


def _by_count(outcomes: tuple[LanguageOutcome, ...], *kinds: str) -> list[str]:
    """Describe the outcomes of ``kinds``, by count descending then language."""
    chosen = [o for o in outcomes if o.outcome in kinds]
    return [_describe(o) for o in sorted(chosen, key=lambda o: (-o.count, o.language))]


def _choices(override: str) -> str:
    """Name the override that applies and what its two other values do."""
    return (
        f'{override}="exclude" excludes these responses, {override}="equal" projects them '
        "with the nominal angles"
    )


def _raise_for_errors(resolved: ResolvedAngles) -> None:
    """Raise for the languages the settings refuse, every one of them listed."""
    uncovered = _by_count(resolved.outcomes, "error_uncovered")
    if uncovered:
        msg = (
            f"no validated angles for {len(uncovered)} {_plural(len(uncovered), 'language')}: "
            f"{_shown(uncovered)}; {_choices('on_uncovered')}"
        )
        raise UncoveredLanguageError(msg)
    low = _by_count(resolved.outcomes, "error_low")
    if low:
        msg = (
            f"only low-confidence angles for {len(low)} {_plural(len(low), 'language')}: "
            f"{_shown(low)}; {_choices('on_low_confidence')}"
        )
        raise LowConfidenceLanguageError(msg)


def _warn(resolved: ResolvedAngles) -> None:
    """One warning per category, after the result exists; ``stacklevel`` is ``project``'s caller."""
    medium = _by_count(resolved.outcomes, "medium")
    if medium:
        msg = (
            f"medium-confidence angles for {len(medium)} {_plural(len(medium), 'language')}: "
            f"{_shown(medium)}; the responses carry the flag {Flag.MEDIUM_CONFIDENCE_LANGUAGE}"
        )
        warnings.warn(msg, MediumConfidenceLanguageWarning, stacklevel=3)
    fallback = _by_count(resolved.outcomes, "low_equal", "uncovered_equal")
    if fallback:
        msg = (
            f"the nominal angles replaced the angle set for {len(fallback)} "
            f"{_plural(len(fallback), 'language')}: {_shown(fallback)}; the responses carry "
            f"the flag {Flag.EQUAL_ANGLES_FALLBACK}"
        )
        warnings.warn(msg, EqualAnglesFallbackWarning, stacklevel=3)


def _status_after(status: Status, resolved: ResolvedAngles) -> Status:
    """Add the projection's reasons and flags to the survey's status."""
    status = status.with_reason(resolved.uncovered_exclude, Reason.UNCOVERED_LANGUAGE)
    status = status.with_reason(resolved.low_exclude, Reason.LOW_CONFIDENCE_LANGUAGE)
    status = status.with_flag(resolved.medium, Flag.MEDIUM_CONFIDENCE_LANGUAGE)
    return status.with_flag(
        resolved.uncovered_equal | resolved.low_equal, Flag.EQUAL_ANGLES_FALLBACK
    )


def _set_refs(
    labels: Sequence[str], sets: Mapping[str, AngleSet], passed: Mapping[str, AngleSet]
) -> list[dict[str, Any]]:
    """One ``data_refs`` entry per set used: a shipped set by its pin, a passed one in full."""
    refs: list[dict[str, Any]] = []
    for label in labels:
        if label == _EQUAL:
            continue
        used = sets[label]
        ref: dict[str, Any] = {"kind": "angle_set", "name": used.name, "version": used.version}
        if label in passed:
            ref["hash"] = hash_json(used)
            ref["content"] = used.model_dump(mode="json", by_alias=True)
        else:
            ref["hash"] = _data.pin(f"angle_sets/{used.name}/{used.version}.toml")
        refs.append(ref)
    return refs


def _carried_columns(survey: Survey, carry: Sequence[str] | _Unset, taken: set[str]) -> list[str]:
    """Return the mapping's ``carry`` columns, then the keyword's new ones, each checked."""
    if carry is _UNSET:
        extra: Sequence[str] = ()
    elif isinstance(carry, str):
        msg = f"carry must be a list of column names, got the string {carry!r}"
        raise TypeError(msg)
    else:
        extra = carry
    columns = list(survey.mapping.carry)  # already checked against the id columns by the mapping
    new = [column for column in extra if column not in columns]
    for column in new:
        if column in taken:
            msg = f"carry names {column!r}, an id column, which is already a label"
            raise MappingError(msg)
    columns += new
    data_columns = list(survey.data.columns)
    for column in columns:
        if column not in data_columns:
            msg = (
                f"carry names {column!r}, which is not a column of the survey's data; "
                f"its columns are {data_columns}"
            )
            raise MappingError(msg)
    return columns


def project(
    survey: Survey,
    *,
    formulation: Formulation | _Unset = _UNSET,
    angles: AngleSetRef | AngleSelection | str | _Unset = _UNSET,
    on_uncovered: Policy | _Unset = _UNSET,
    on_low_confidence: Policy | _Unset = _UNSET,
    angle_sets: Sequence[AngleSet] | _Unset = _UNSET,
    settings: ProjectionSettings | Mapping[str, object] | str | os.PathLike[str] | _Unset = _UNSET,
    carry: Sequence[str] | _Unset = _UNSET,
    profile: str | _Unset = _UNSET,
) -> Coordinates:
    """Project a survey's attribute scores onto the two circumplex axes.

    Only included responses are projected; excluded ones get NaN coordinates and an
    empty ``ssp_angle_set`` label. The scale is the mapping's declared attribute scale,
    never the data's range. The survey is unchanged.

    Each response's angles come from its language and the angle selection: the set
    ``per_language`` names for the language, else ``default``. ``"equal"`` uses the
    circumplex's nominal angles and never reads the language. Under a named set a
    high-tier language projects with the set's angles and label; a medium-tier one the
    same, with the flag ``medium_confidence_language`` and one
    ``MediumConfidenceLanguageWarning``; a low-tier one with angles follows
    ``on_low_confidence``; a language the set does not cover, or lists with a tier and no
    angles, follows ``on_uncovered``. ``"error"`` raises after every language has been
    resolved, listing each with its count; ``"exclude"`` writes the reason
    ``uncovered_language`` or ``low_confidence_language``; ``"equal"`` projects with the
    nominal angles, the label ``equal``, the flag ``equal_angles_fallback`` and one
    ``EqualAnglesFallbackWarning``. The status labels are the survey's own plus these
    reasons and flags.

    Settings resolve by the one rule of spec 6.3.4. The base profile is ``profile``, else
    the ``extends`` of a settings file passed as ``settings``, else the profile the survey
    records (so a profile named on ``read_survey()`` carries over), else ``sal-2026``, whose
    projection section names ``satp@1.5``. Then each field takes the first value set, in
    this order: a keyword, ``settings=``, the file's ``[projection]`` section, the
    profile's section, the model default; the result validates once. A
    ``ProjectionSettings`` or ``AngleSelection`` instance contributes only the fields that
    were set on it. A settings file applies to the call it is passed to; one given to
    ``read_survey()`` does not reach this call. The result records
    ``provenance["profile"]`` (``{id, version, hash}``), ``provenance["overrides"]`` (every
    field that differs from the profile's section) and, when an explicit ``profile``
    replaced the survey's, ``provenance["profile_inherited"]``.

    After the id-role labels come the carried columns, one text label each: the mapping's
    ``carry`` entries in order, then ``carry=`` entries not already among them. A carried
    cell is the file's text verbatim, ``""`` where blank; it is never parsed, and carrying
    changes no number, so ``carry`` is not a settings field.

    Parameters
    ----------
    survey : Survey
        The survey, from ``read_survey()``.
    formulation : {"canonical", "iso2019", "aletta2024"}, optional
        The projection formula; left unset, the profile's (``"canonical"`` under
        ``sal-2026``).
    angles : AngleSetRef, "equal", "name@version" or AngleSelection, optional
        The angles: a set for every language (an ``AngleSetRef`` or its ``"name@version"``
        string), ``"equal"`` for the nominal angles, or a full ``AngleSelection``. Left
        unset, the profile's (``satp@1.5`` under ``sal-2026``).
    on_uncovered : {"error", "exclude", "equal"}, optional
        What a language absent from its set, or listed without angles, does; left unset,
        the profile's (``"error"`` in every shipped profile).
    on_low_confidence : {"error", "exclude", "equal"}, optional
        The same for a low-tier language that has angles; left unset, the profile's
        (``"error"`` in every shipped profile).
    angle_sets : sequence of AngleSet, optional
        Sets to resolve names against before the shipped ones; a passed set shadows a
        shipped set of the same ``name@version`` for this call, and its full table is
        recorded in the provenance so the result does not depend on a file.
    settings : ProjectionSettings, mapping, str or os.PathLike, optional
        Every projection setting at once, or the path of a settings file (``extends`` plus
        sections, as ``save_settings()`` writes it); the keywords win over it.
    carry : sequence of str, optional
        Further columns of the survey's data to carry as labels, beyond the mapping's.
    profile : str, optional
        The profile to resolve under: ``"id@version"``, or a bare id meaning the latest
        shipped version; ``list_profiles()`` lists them. It wins over a settings file's
        ``extends`` and over the survey's recorded profile.

    Returns
    -------
    Coordinates
        One row per survey response, in the survey's order, with the settings used and
        the provenance envelope. ``provenance["data_refs"]`` holds the survey's refs, then
        one entry per angle set a response used: a shipped set as its name, version and
        pinned hash, a passed set with its full table under ``content`` and its content
        hash. ``origin[0].angle_sets`` lists the distinct labels of the included rows.

    Raises
    ------
    SettingsError
        If ``formulation``, ``angles``, an override or ``settings`` is not a value this
        version accepts, the settings file is missing or invalid, or ``angle_sets`` lists
        one label twice.
    DataReferenceError
        If ``profile`` names no shipped profile (the message lists the shipped ones); or if
        a named set is neither passed nor shipped, or is for another circumplex or
        circumplex version than the survey's, or has another number of angles than the
        circumplex has positions.
    ProfileConstraintError
        If a field the profile locks resolved to another value than the profile's; the
        message names the field and suggests ``profile="sal-2026"``.
    MappingError
        If a named set is used and the mapping has no language column or value, or an
        included response has a blank language (the message names the responses); or if
        a carried column is not in the survey's data, or is an id column.
    UncoveredLanguageError
        Under ``on_uncovered="error"``, for a language absent from its set or listed
        without angles; the message names its tier where it has one.
    LowConfidenceLanguageError
        Under ``on_low_confidence="error"``, for a low-tier language with angles.
    FormulationError
        If ``"iso2019"`` is used with a scale other than 1 to 5 or with other angles than
        the nominal ones.
    TypeError
        If ``carry`` is a single string, or ``angle_sets`` is a string or a bare
        ``AngleSet`` rather than a sequence of them.

    Warns
    -----
    MediumConfidenceLanguageWarning
        Once per call, when a response projects with a medium-tier language's angles.
    EqualAnglesFallbackWarning
        Once per call, when the nominal angles replaced a set under an ``"equal"`` override.

    References
    ----------
    The canonical formulation: [@mitchell2026translation], proof p. 23, §4.5. The 2019
    formulation: [@iso12913-3-2019], as written in [@mitchell2022how], Eqs. 1 and 2. The
    2024 formulation and the first language-specific angles: [@aletta2024soundscape]. The
    tiers, and the rule that a language without validated angles is not projected with
    another language's: [@mitchell2026translation], chapter 4.
    """
    # A survey built from a record without a profile (an earlier version's) inherits none.
    inherited: Any = survey.provenance.get("profile")
    resolution = resolve_settings(
        ProjectionSettings,
        "projection",
        keywords=_keywords(formulation, angles, on_uncovered, on_low_confidence),
        settings=settings,
        profile=profile,
        inherited=inherited,
    )
    resolved_settings = resolution.settings
    passed = _passed_sets(angle_sets)
    sets = _named_sets(resolved_settings.angles, passed)
    _check_language_mapped(survey, sets)
    circ = circumplex(survey.mapping.instrument, survey.mapping.instrument_version)
    resolved = resolve_angles(
        survey.language, survey.status.included, resolved_settings.angles, sets, circ
    )
    _check_blank_languages(survey, resolved)
    _raise_for_errors(resolved)
    status = _status_after(survey.status, resolved)
    included = status.included

    scale = survey.mapping.groups.attributes.scale
    scores = survey.scores["attributes"]
    xy = np.full((survey.n(), 2), np.nan, dtype=np.float64)
    xy[included] = project_scores(
        scores[included], resolved.angles_deg[included], scale, resolved_settings.formulation
    )
    angle_set_labels: NDArray[np.object_] = np.where(included, resolved.labels, "").astype(object)
    used = sorted(set(angle_set_labels[included].tolist()))

    id_labels = {getattr(survey.mapping.ids, role): values for role, values in survey.ids.items()}
    labels = {
        COL_RESPONSE_ID: survey.response_id,
        **status.label_arrays(),
        COL_LANGUAGE: survey.language,
        COL_ANGLE_SET: angle_set_labels,
        **id_labels,
    }
    for column in _carried_columns(survey, carry, set(id_labels)):
        labels[column] = column_text(survey.data, column)

    # read_survey wrote these keys with the types envelope takes; Any passes them on as is.
    carried: dict[str, Any] = {key: survey.provenance[key] for key in _CARRIED}
    if resolution.profile_inherited is not None:
        carried["profile_inherited"] = resolution.profile_inherited.model_dump()
    survey_refs: Any = survey.provenance["data_refs"]
    provenance = envelope(
        operation="project",
        settings=resolved_settings,
        data_refs=[*survey_refs, *_set_refs(used, sets, passed)],
        profile=resolution.profile.model_dump(),
        overrides=resolution.overrides,
        **carried,
    )
    segment = OriginSegment(
        kind="survey",
        dataset=survey.source.name,
        n=survey.n(),
        input_hash=carried["input_hash"],
        mapping_hash=carried["mapping_hash"],
        effective_scale=scale,
        angle_sets=tuple(used),
        settings_hash=hash_json(resolved_settings),
    )
    coords = Coordinates(
        xy=xy,
        circumplex=circ,
        labels=labels,
        origin=(segment,),
        settings=resolved_settings,
        provenance=provenance,
    )
    _warn(resolved)
    return coords
