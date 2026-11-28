"""Profiles, settings files and the one settings-resolution rule (spec 6.3.4)."""

from __future__ import annotations

import re

import pytest

from soundscapy import _data
from soundscapy._model import _UNSET, FrozenModel, merge_settings
from soundscapy.errors import DataReferenceError, ProfileConstraintError, SettingsError
from soundscapy.instrument.angles import AngleSelection, AngleSetRef
from soundscapy.profiles import (
    SECTIONS,
    ProfileHeader,
    ProfileRef,
    list_profiles,
    read_profile,
    read_settings_file,
    resolve_settings,
    save_settings,
)
from soundscapy.projection.project import ProjectionSettings

_PROFILES = ("iso-12913-3-2019@1", "iso-12913-3-2024@1", "sal-2026@1")
_SATP = AngleSetRef(name="satp", version="1.5")


class _Inner(FrozenModel):
    depth: int = 0


class _Toy(FrozenModel):
    a: int = 0
    b: str = "x"
    inner: _Inner = _Inner()


def _toy(**kwargs):
    kwargs.setdefault("keywords", {})
    return resolve_settings(_Toy, "describe", **kwargs)


def _projection(**kwargs):
    kwargs.setdefault("keywords", {})
    return resolve_settings(ProjectionSettings, "projection", **kwargs)


# --- The shipped profiles -------------------------------------------------------------------------


def test_list_profiles_is_sorted_labels():
    assert list_profiles() == _PROFILES


def test_shipped_profiles_validate_and_pin():
    for label in _PROFILES:
        header, sections, digest = read_profile(label)
        ident, version = label.split("@")
        assert (header.id, header.version, header.label) == (ident, version, label)
        assert digest == _data.pin(f"profiles/{ident}/{version}.toml")
        assert header.citation.citation
        assert set(sections) <= set(SECTIONS)
        merge_settings(ProjectionSettings, [sections["projection"]], context=label)


def test_shipped_profile_contents():
    sal = _projection(profile="sal-2026")
    assert sal.settings == ProjectionSettings(angles=AngleSelection(default=_SATP))
    assert sal.profile == ProfileRef(
        id="sal-2026", version="1", hash=_data.pin("profiles/sal-2026/1.toml")
    )
    assert _projection(profile="iso-12913-3-2019").settings.formulation == "iso2019"
    assert _projection(profile="iso-12913-3-2024").settings.formulation == "canonical"
    header_2019, sections_2019, _ = read_profile("iso-12913-3-2019")
    header_2024, sections_2024, _ = read_profile("iso-12913-3-2024")
    assert (
        set(header_2019.locked)
        == set(header_2024.locked)
        == {
            "projection.formulation",
            "projection.angles.default",
            "projection.angles.per_language",
            "item_statistics.dispersion",
        }
    )
    assert sections_2019["item_statistics"] == {"dispersion": "range"}
    assert sections_2024["item_statistics"] == {"dispersion": "iqr"}
    assert "interpretation pending the owner's check of the text" in header_2024.notes
    header_sal, sections_sal, _ = read_profile("sal-2026")
    assert header_sal.locked == ()
    assert sections_sal["item_statistics"] == {"dispersion": "iqr"}
    assert "satp@1.5" in header_sal.notes


def test_bare_id_means_latest_and_full_label_is_exact():
    assert _projection(profile="sal-2026").profile.version == "1"
    assert _projection(profile="sal-2026@1").profile.label == "sal-2026@1"


@pytest.mark.parametrize("bad", ["x@1", "x", "sal-2026@9"])
def test_unknown_profile_lists_the_shipped_ones(bad):
    with pytest.raises(DataReferenceError, match=re.escape(str(list(_PROFILES)))):
        _projection(profile=bad)


# --- Precedence, rank by rank ---------------------------------------------------------------------


def test_default_profile_is_sal_2026():
    resolved = _projection()
    assert resolved.profile.label == "sal-2026@1"
    assert resolved.profile_inherited is None
    assert resolved.overrides == {}


def test_profile_section_beats_model_default():
    assert ProjectionSettings().angles.default == "equal"
    assert _projection().settings.angles.default == _SATP


def _write(tmp_path, text):
    path = tmp_path / "settings.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_ranks_file_settings_keyword(tmp_path):
    path = _write(tmp_path, 'extends = "sal-2026"\n[describe]\na = 1\nb = "file"\n')
    assert _toy().settings == _Toy()  # no [describe] section: the model default
    assert _toy(settings=path).settings == _Toy(a=1, b="file")
    assert _toy(settings=path, keywords={"a": 3}).settings == _Toy(a=3, b="file")
    assert _toy(settings={"a": 2}).settings == _Toy(a=2)
    assert _toy(settings=_Toy(a=2), keywords={"a": 3, "b": _UNSET}).settings == _Toy(a=3)
    assert _toy(keywords={"a": _UNSET, "b": _UNSET}).settings == _Toy()


def test_nested_layers_merge_leaf_by_leaf():
    resolved = _projection(settings={"angles": {"on_uncovered": "exclude"}})
    assert resolved.settings.angles == AngleSelection(default=_SATP, on_uncovered="exclude")
    assert resolved.overrides == {"projection.angles.on_uncovered": "exclude"}


def test_passed_model_contributes_only_set_fields():
    resolved = _projection(settings=ProjectionSettings())
    assert resolved.settings.angles.default == _SATP
    assert resolved.overrides == {}


def test_merged_result_validates_once_with_the_field_path():
    with pytest.raises(SettingsError, match="formulation"):
        _projection(keywords={"formulation": "nope"})
    with pytest.raises(SettingsError, match=re.escape("inner.depth")):
        _toy(settings={"inner": {"depth": "deep"}})


# --- Overrides ------------------------------------------------------------------------------------


def test_overrides_are_the_diff_against_the_profile():
    assert _projection(keywords={"angles": {"default": "equal"}}).overrides == {
        "projection.angles.default": "equal"
    }
    first = _projection(keywords={"formulation": "aletta2024"})
    assert first.overrides == {"projection.formulation": "aletta2024"}
    again = _projection(settings=first.settings)
    assert again.settings == first.settings
    assert again.overrides == first.overrides
    same = _projection(settings=_projection().settings)
    assert same.overrides == {}


# --- Locked fields --------------------------------------------------------------------------------


def test_locked_formulation_is_refused_with_a_suggestion():
    with pytest.raises(ProfileConstraintError, match=re.escape("projection.formulation")) as info:
        _projection(profile="iso-12913-3-2019", keywords={"formulation": "canonical"})
    assert 'profile="sal-2026"' in str(info.value)
    with pytest.raises(ProfileConstraintError, match=re.escape("projection.formulation")):
        _projection(profile="iso-12913-3-2024", keywords={"formulation": "aletta2024"})


def test_locked_value_equal_to_the_profile_passes():
    resolved = _projection(profile="iso-12913-3-2019", keywords={"angles": {"default": "equal"}})
    assert resolved.settings.angles.default == "equal"
    assert resolved.overrides == {}


def test_locked_table_is_locked_as_a_whole():
    with pytest.raises(ProfileConstraintError, match=re.escape("projection.angles.per_language")):
        _projection(
            profile="iso-12913-3-2019",
            keywords={"angles": {"per_language": {"cmn": "aletta-2024@1"}}},
        )


def test_unlocked_profile_records_the_override():
    resolved = _projection(profile="sal-2026", keywords={"formulation": "aletta2024"})
    assert resolved.settings.formulation == "aletta2024"
    assert resolved.overrides == {"projection.formulation": "aletta2024"}


def test_unlocked_field_under_a_locked_profile_is_an_override():
    resolved = _projection(
        profile="iso-12913-3-2019", keywords={"angles": {"on_uncovered": "exclude"}}
    )
    assert resolved.overrides == {"projection.angles.on_uncovered": "exclude"}


# --- Inherited profiles ---------------------------------------------------------------------------


def test_inherited_profile_is_the_base_when_none_is_given():
    inherited = _projection(profile="iso-12913-3-2019").profile
    resolved = _projection(inherited=inherited)
    assert resolved.profile == inherited
    assert resolved.profile_inherited is None
    assert resolved.settings.formulation == "iso2019"
    as_mapping = _projection(inherited=inherited.model_dump())
    assert as_mapping.profile == inherited


def test_explicit_profile_replaces_and_records_the_inherited_one():
    inherited = _projection(profile="iso-12913-3-2019").profile
    resolved = _projection(profile="sal-2026", inherited=inherited)
    assert resolved.profile.label == "sal-2026@1"
    assert resolved.profile_inherited == inherited
    assert resolved.overrides == {}
    same = _projection(profile="iso-12913-3-2019", inherited=inherited)
    assert same.profile_inherited is None


def test_settings_file_extends_outranks_the_inherited_profile(tmp_path):
    # Spec 11.4: an explicit file's extends ranks with an explicit profile, above the
    # profile the input recorded; the replaced one is recorded as inherited.
    inherited = _projection(profile="iso-12913-3-2019").profile
    path = _write(tmp_path, 'extends = "sal-2026"\n')
    resolved = _projection(settings=path, inherited=inherited)
    assert resolved.profile.label == "sal-2026@1"
    assert resolved.profile_inherited == inherited
    assert resolved.settings.formulation == "canonical"


# --- Settings files -------------------------------------------------------------------------------


def test_settings_file_extends_names_the_base(tmp_path):
    path = _write(tmp_path, 'extends = "iso-12913-3-2019"\n[projection]\nformulation = "iso2019"\n')
    assert read_settings_file(path) == (
        "iso-12913-3-2019",
        {"projection": {"formulation": "iso2019"}},
    )
    resolved = _projection(settings=path)
    assert resolved.profile.label == "iso-12913-3-2019@1"
    assert resolved.settings.formulation == "iso2019"
    assert resolved.overrides == {}


def test_settings_file_without_extends_lists_profiles_and_names_save_settings(tmp_path):
    path = _write(tmp_path, '[projection]\nformulation = "iso2019"\n')
    with pytest.raises(SettingsError, match="extends") as info:
        read_settings_file(path)
    message = str(info.value)
    assert all(label in message for label in _PROFILES)
    assert "save_settings" in message


def test_settings_file_unknown_section_is_refused(tmp_path):
    path = _write(tmp_path, 'extends = "sal-2026"\n[projektion]\nformulation = "iso2019"\n')
    with pytest.raises(SettingsError, match="projektion"):
        read_settings_file(path)


def test_settings_file_other_section_is_ignored(tmp_path):
    path = _write(tmp_path, 'extends = "sal-2026"\n[describe]\nby = "LocationID"\n')
    resolved = _projection(settings=path)
    assert resolved.settings == _projection().settings
    assert resolved.overrides == {}


def test_settings_file_missing_or_invalid_names_the_path(tmp_path):
    missing = tmp_path / "nope.toml"
    with pytest.raises(SettingsError, match=re.escape(str(missing))):
        read_settings_file(missing)
    broken = _write(tmp_path, "extends = \n")
    with pytest.raises(SettingsError, match=re.escape(str(broken))):
        _projection(settings=broken)


def test_settings_file_section_must_be_a_table(tmp_path):
    path = _write(tmp_path, 'extends = "sal-2026"\nprojection = 3\n')
    with pytest.raises(SettingsError, match="projection"):
        read_settings_file(path)


def test_explicit_profile_wins_over_extends_and_the_file_still_applies(tmp_path):
    path = _write(
        tmp_path, 'extends = "iso-12913-3-2019"\n[projection.angles]\non_uncovered = "exclude"\n'
    )
    resolved = _projection(settings=path, profile="sal-2026")
    assert resolved.profile.label == "sal-2026@1"
    assert resolved.settings.formulation == "canonical"
    assert resolved.settings.angles == AngleSelection(default=_SATP, on_uncovered="exclude")
    assert resolved.overrides == {"projection.angles.on_uncovered": "exclude"}


def test_settings_file_bad_value_fails_validation_with_the_path(tmp_path):
    path = _write(tmp_path, 'extends = "sal-2026"\n[projection]\nformulation = 3\n')
    with pytest.raises(SettingsError, match="formulation"):
        _projection(settings=path)


# --- save_settings --------------------------------------------------------------------------------


def test_save_settings_writes_extends_and_sections(tmp_path):
    path = tmp_path / "saved.toml"
    save_settings(path, extends="sal-2026", overrides={"projection.formulation": "aletta2024"})
    text = path.read_text(encoding="utf-8")
    assert 'extends = "sal-2026@1"' in text
    assert "[projection]" in text
    assert read_settings_file(path) == ("sal-2026@1", {"projection": {"formulation": "aletta2024"}})
    resolved = _projection(settings=path)
    assert resolved.settings == _projection(keywords={"formulation": "aletta2024"}).settings
    assert resolved.overrides == {"projection.formulation": "aletta2024"}


def test_save_settings_nests_dotted_keys(tmp_path):
    path = tmp_path / "saved.toml"
    overrides = {
        "projection.angles.on_uncovered": "exclude",
        "projection.angles.per_language.cmn": "aletta-2024@1",
        "item_statistics.dispersion": "range",
    }
    save_settings(path, extends="sal-2026@1", overrides=overrides)
    _, sections = read_settings_file(path)
    assert sections == {
        "projection": {
            "angles": {"on_uncovered": "exclude", "per_language": {"cmn": "aletta-2024@1"}}
        },
        "item_statistics": {"dispersion": "range"},
    }
    resolved = _projection(settings=path)
    assert resolved.overrides == {
        "projection.angles.on_uncovered": "exclude",
        "projection.angles.per_language": {"cmn": "aletta-2024@1"},
    }


def test_save_settings_refuses_a_locked_override(tmp_path):
    path = tmp_path / "saved.toml"
    with pytest.raises(ProfileConstraintError, match=re.escape("projection.formulation")):
        save_settings(
            path, extends="iso-12913-3-2019", overrides={"projection.formulation": "canonical"}
        )
    assert not path.exists()


def test_save_settings_validates_a_touched_section(tmp_path):
    path = tmp_path / "saved.toml"
    with pytest.raises(SettingsError, match="formulation"):
        save_settings(path, extends="sal-2026", overrides={"projection.formulation": "nope"})
    with pytest.raises(SettingsError, match="projektion"):
        save_settings(path, extends="sal-2026", overrides={"projektion.formulation": "iso2019"})
    with pytest.raises(SettingsError, match="formulation"):
        save_settings(path, extends="sal-2026", overrides={"formulation": "iso2019"})
    with pytest.raises(DataReferenceError):
        save_settings(path, extends="nope", overrides={})


def test_save_settings_overwrites(tmp_path):
    path = tmp_path / "saved.toml"
    save_settings(path, extends="sal-2026", overrides={"projection.formulation": "aletta2024"})
    save_settings(path, extends="iso-12913-3-2024", overrides={})
    assert read_settings_file(path) == ("iso-12913-3-2024@1", {})


# --- Models ---------------------------------------------------------------------------------------


def test_profile_header_reads_the_schema_alias():
    header = ProfileHeader.model_validate(
        {"schema": "soundscapy.profile", "schema_version": 1, "id": "p", "version": "2"}
    )
    assert header.schema_name == "soundscapy.profile"
    assert header.model_dump(mode="json", by_alias=True)["schema"] == "soundscapy.profile"
    assert header.label == "p@2"
    with pytest.raises(ValueError, match="schema_version"):
        ProfileHeader.model_validate({"schema": "soundscapy.profile", "schema_version": 2})


def test_profile_ref_label():
    ref = ProfileRef(id="sal-2026", version="1", hash="h1-sha256:" + "a" * 64)
    assert ref.label == "sal-2026@1"
