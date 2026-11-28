"""Every FrozenModel survives a TOML write and read with an equal value and an equal hash."""

from __future__ import annotations

import codecs
import encodings.aliases
import importlib
import pkgutil

import pytest
from hypothesis import given
from hypothesis import strategies as st

import soundscapy
from soundscapy import _data
from soundscapy._model import Citation, FrozenModel, from_toml, to_toml
from soundscapy.distribution.coordinates import OriginSegment
from soundscapy.instrument import Scale, circumplex
from soundscapy.instrument.angles import AngleSelection
from soundscapy.instrument.circumplex import Attribute, Axis, Circumplex, Translation
from soundscapy.projection.project import ProjectionSettings
from soundscapy.provenance import hash_file, hash_json
from soundscapy.survey.mapping import (
    Coding,
    FlagSpec,
    FlagSpecs,
    Groups,
    IdSpec,
    ItemGroup,
    LanguageSpec,
    SurveyMapping,
)
from soundscapy.survey.quality import QualitySettings
from soundscapy.survey.read import ImportSettings, ReadOptions, SourceRef
from tests._helpers import quality_settings

_CITATION = Citation(
    citation="A. Author (2026). A title.",
    doi_or_url="https://doi.org/10.0/x",
    licence="CC-BY-4.0",
    date="2026-10-31",
    notes="Ünïcode notes",
)
_SCALE = Scale(
    minimum=1.0,
    maximum=3.0,
    kind="ordinal",
    labels={"eng": ("low", "mid", "high"), "deu": ("niedrig", "mittel", "hoch")},
)
_CODING = Coding(
    missing_values=("", "-99"),
    value_map={"Strongly disagree": 1.0, "Strongly agree": 5.0},
    decimal=",",
    offset=1.0,
    reverse=("PAQ8",),
)
_COLUMNS = tuple((f"PAQ{i}", f"q{i}") for i in range(1, 9))
_ITEM_GROUP = ItemGroup(
    columns=_COLUMNS, scale=Scale(minimum=1.0, maximum=7.0, kind="interval"), coding=_CODING
)
_ID_SPEC = IdSpec(
    response_key=("LocationID", "RecordID"),
    response_key_blank_ok=("RecordID",),
    location="LocationID",
    session="SessionID",
    group="GroupID",
    record="RecordID",
)
_LANGUAGE = LanguageSpec(column="Lang", coding={"English": "eng", "Deutsch": "deu"})
_FLAG = FlagSpec(column="consent", true_values=("yes", "Yes"))
_FLAGS = FlagSpecs(
    consent=_FLAG,
    test=FlagSpec(column="test", true_values=("1",)),
    paper_input=FlagSpec(column="paper", true_values=("yes",)),
)
_GROUPS = Groups(
    attributes=_ITEM_GROUP,
    source_dominance=ItemGroup(
        columns=(("traffic_noise", "s1"), ("natural_sounds", "s4")),
        scale=Scale(minimum=1.0, maximum=5.0),
    ),
    overall_quality=ItemGroup(
        columns=(("overall_quality", "oq"),),
        scale=Scale(minimum=1.0, maximum=5.0),
        coding=Coding(value_map={"bad": 1.0, "good": 5.0}),
    ),
    appropriateness=ItemGroup(
        columns=(("appropriateness", "ap"),), scale=Scale(minimum=1.0, maximum=5.0)
    ),
    who5=ItemGroup(
        columns=tuple((f"who5_{i}", f"w{i}") for i in range(1, 6)),
        scale=Scale(minimum=1.0, maximum=6.0, kind="interval"),
    ),
)


def _source_ref() -> SourceRef:
    with _data.as_path("datasets/isd/subset-v1.0.csv") as path:
        return SourceRef(
            name=path.name,
            hash=hash_file(path),
            read_options=ReadOptions(format="csv", encoding="utf-8-sig", delimiter=","),
        )


def _circumplex() -> Circumplex:
    shipped = circumplex()
    return Circumplex.model_validate(
        {**shipped.model_dump(), "name": "custom", "version": "2", "citation": _CITATION}
    )


MODELS: list[FrozenModel] = [
    _CITATION,
    _SCALE,
    Attribute(position="PAQ3", angle_deg=92.5, labels={"eng": "eventful"}),
    Axis(id="X", positive="PAQ1", negative="PAQ5", labels={"eng": "Pleasantness"}),
    Translation(language="deu", labels={"PAQ1": "angenehm"}, citation=_CITATION),
    _circumplex(),
    _CODING,
    _ID_SPEC,
    _LANGUAGE,
    _ITEM_GROUP,
    _FLAG,
    _FLAGS,
    _GROUPS,
    SurveyMapping(
        schema_name="soundscapy.mapping",
        schema_version=1,
        ids=_ID_SPEC,
        language=_LANGUAGE,
        groups=_GROUPS,
        age="age",
        flags=_FLAGS,
        carry=("start_time", "Survey_Version"),
    ),
    ReadOptions(format="tsv", encoding="cp1252", delimiter=";"),
    ImportSettings(quality=QualitySettings(min_age=16.0), on_scale_violation="exclude"),
    QualitySettings(
        all_midpoint="off",
        excessive_missing_share=0.25,
        under_age="exclude",
        min_age=16.0,
        paper_input="exclude",
    ),
    _source_ref(),
    AngleSelection(),  # its one S1 field has one value
    ProjectionSettings(formulation="iso2019", angles=AngleSelection()),
    OriginSegment(
        kind="survey",
        dataset="subset-v1.0.csv",
        n=1552,
        input_hash="h1-sha256:" + "a" * 64,
        mapping_hash="h1-sha256:" + "b" * 64,
        effective_scale=_SCALE,
        angle_sets=("equal",),
        settings_hash="h1-sha256:" + "c" * 64,
    ),
]


def _assert_round_trips(model: FrozenModel) -> None:
    back = from_toml(type(model), to_toml(model))
    assert back == model
    assert hash_json(back) == hash_json(model)


@pytest.mark.parametrize("model", MODELS, ids=lambda m: type(m).__name__)
def test_every_frozen_model_round_trips(model):
    _assert_round_trips(model)


def _known(name: str) -> bool:
    try:
        codecs.lookup(name)
    except LookupError:
        return False
    return True


_ENCODINGS = sorted(
    name
    for name in {*encodings.aliases.aliases, *encodings.aliases.aliases.values()}
    if _known(name)
)


@given(
    fmt=st.sampled_from(["auto", "csv", "tsv"]),
    encoding=st.sampled_from(_ENCODINGS),
    # Lone surrogates are not text: TOML and UTF-8 cannot hold them.
    delimiter=st.just("auto") | st.characters(exclude_categories=("Cs",)),
)
def test_read_options_round_trip(fmt, encoding, delimiter):
    _assert_round_trips(ReadOptions(format=fmt, encoding=encoding, delimiter=delimiter))


@given(st.sampled_from(["error", "exclude"]), quality_settings())
def test_import_settings_round_trip(action, quality):
    _assert_round_trips(ImportSettings(quality=quality, on_scale_violation=action))


@given(quality_settings())
def test_quality_settings_round_trip(settings):
    _assert_round_trips(settings)


def _subclasses(cls: type) -> set[type]:
    found: set[type] = set()
    for sub in cls.__subclasses__():
        found |= {sub, *_subclasses(sub)}
    return found


def test_round_trip_covers_every_model():
    for info in pkgutil.walk_packages(soundscapy.__path__, "soundscapy."):
        importlib.import_module(info.name)
    models = {cls for cls in _subclasses(FrozenModel) if cls.__module__.startswith("soundscapy.")}
    covered = {type(m) for m in MODELS}
    missing = sorted(f"{cls.__module__}.{cls.__qualname__}" for cls in models - covered)
    assert not missing, f"add an instance of each to MODELS: {missing}"
