"""``save_settings``: the file it writes, the sections it checks and the locks it keeps."""

from __future__ import annotations

import re

import pytest

from soundscapy.errors import DataReferenceError, ProfileConstraintError, SettingsError
from soundscapy.profiles import read_settings_file, resolve_settings
from soundscapy.projection.project import ProjectionSettings
from soundscapy.workspace import save_settings


def _projection(**kwargs):
    kwargs.setdefault("keywords", {})
    return resolve_settings(ProjectionSettings, "projection", **kwargs)


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
