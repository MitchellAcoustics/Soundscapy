"""Unit tests for soundscapy._optional."""

import pytest

from soundscapy._optional import _install_hint, require_deps


def test_require_deps_passes_when_present():
    # `pytest` is necessarily available while running pytest itself.
    require_deps(["pytest"], extra="x")


def test_require_deps_raises_with_install_hint():
    with pytest.raises(ImportError) as excinfo:
        require_deps(["definitely_not_a_real_module_xyz"], extra="x")
    msg = str(excinfo.value)
    assert "pip install 'soundscapy[x]'" in msg
    assert "'definitely_not_a_real_module_xyz'" in msg


def test_require_deps_lists_only_missing():
    with pytest.raises(ImportError) as excinfo:
        require_deps(["pytest", "definitely_not_real_xyz"], extra="x")
    msg = str(excinfo.value)
    # Only the missing module should appear in the message.
    assert "'definitely_not_real_xyz'" in msg
    assert "'pytest'" not in msg


def test_dist_name_translation():
    hint = _install_hint("audio", ["acoustic_toolbox"])
    assert "'acoustic-toolbox'" in hint
    # The import name should not appear with its underscore form.
    assert "'acoustic_toolbox'" not in hint


def test_dist_name_passthrough():
    # Modules with no entry in _DIST_NAME use their import name verbatim.
    hint = _install_hint("r", ["rpy2"])
    assert "'rpy2'" in hint
