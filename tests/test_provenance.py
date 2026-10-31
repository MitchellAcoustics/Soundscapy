"""Tests for hashing and the provenance envelope."""

import importlib.metadata
import re
from pathlib import Path

import pytest

from soundscapy._model import FrozenModel
from soundscapy.errors import HashSchemeError
from soundscapy.provenance import (
    check_hash,
    envelope,
    hash_bytes,
    hash_file,
    hash_json,
    show_versions,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "soundscapy"


class _Settings(FrozenModel):
    clip: str = "none"


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (
            hash_bytes(b""),
            "h1-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            hash_json({}),
            "h1-sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        ),
        (
            hash_json({"b": [1.5, "é"], "a": 1}),
            "h1-sha256:2bb202b35963ad9e57f87e5828e072cc53f4720ecd10a7d625f850411a31c4c6",
        ),
        (
            hash_json({"clip": "none"}),
            "h1-sha256:7c457b553a03102b7220154eb95650e02d2840d02f397a2d226afba1fa85a1dd",
        ),
    ],
)
def test_golden_vectors(actual: str, expected: str) -> None:
    assert actual == expected


def test_iterable_hashes_as_concatenation() -> None:
    assert hash_bytes(iter([b"ab", b"c"])) == hash_bytes(b"abc")


def test_non_finite_json_names_the_key() -> None:
    with pytest.raises(ValueError, match="a"):
        hash_json({"a": float("inf")})
    with pytest.raises(ValueError, match=re.escape("b[1]")):
        hash_json({"b": [1.0, float("nan")]})


def test_non_json_value_raises_type_error() -> None:
    with pytest.raises(TypeError, match="a"):
        hash_json({"a": {1, 2}})  # ty: ignore[invalid-argument-type]  # deliberate: a set is not JSON
    with pytest.raises(TypeError, match="key"):
        hash_json({1: "x"})  # ty: ignore[invalid-argument-type]  # deliberate: int key


def test_hash_file_equals_hash_bytes(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"some bytes\n")
    assert hash_file(path) == hash_bytes(b"some bytes\n")


def test_model_and_its_dump_hash_alike() -> None:
    model = _Settings()
    assert hash_json(model) == hash_json(model.model_dump(mode="json"))


def test_check_hash() -> None:
    digest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert check_hash(f"h1-sha256:{digest}") == digest
    with pytest.raises(HashSchemeError):
        check_hash(f"sha256:{digest}")
    with pytest.raises(HashSchemeError):
        check_hash(f"h2-sha256:{digest}")
    with pytest.raises(HashSchemeError):
        check_hash("h1-sha256:abc")


def test_hashlib_only_in_provenance() -> None:
    offenders = [
        p.name
        for p in SRC.rglob("*.py")
        if re.search(r"^\s*(import hashlib|from hashlib)", p.read_text("utf-8"), re.MULTILINE)
    ]
    assert offenders == ["provenance.py"]


def test_envelope_keys() -> None:
    settings = _Settings()
    env = envelope(operation="project", settings=settings, input_hash=hash_bytes(b""))
    assert set(env) == {
        "soundscapy_version",
        "python_version",
        "platform",
        "libraries",
        "hash_scheme",
        "data_refs",
        "created_utc",
        "operation",
        "input_hash",
        "mapping_hash",
        "settings_hash",
        "upstream_run",
    }
    assert env["soundscapy_version"] == importlib.metadata.version("soundscapy")
    assert env["hash_scheme"] == "h1"
    assert env["settings_hash"] == hash_json(settings)
    assert env["mapping_hash"] == ""
    assert env["upstream_run"] == ""
    assert str(env["created_utc"]).endswith("Z")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", str(env["created_utc"]))
    libraries = env["libraries"]
    assert isinstance(libraries, dict)
    assert {"numpy", "pandas", "narwhals", "pydantic"} <= set(libraries)


def test_envelope_accepts_optional_keys_and_rejects_typos() -> None:
    common = {
        "operation": "project",
        "settings": _Settings(),
        "input_hash": hash_bytes(b""),
    }
    env = envelope(**common, quality_skipped=["r1"], extra_libraries=["pytest"])  # ty: ignore[invalid-argument-type]  # dict unpack of mixed values
    assert env["quality_skipped"] == ["r1"]
    assert "pytest" in env["libraries"]  # ty: ignore[unsupported-operator]  # libraries is a dict
    with pytest.raises(ValueError, match="qualty_skipped"):
        envelope(**common, qualty_skipped=[])  # ty: ignore[invalid-argument-type]  # dict unpack of mixed values


def test_show_versions_returns_and_never_prints(
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = show_versions()
    assert text.startswith("soundscapy: ")
    assert "numpy: " in text
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
