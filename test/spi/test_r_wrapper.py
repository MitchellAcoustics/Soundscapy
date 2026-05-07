"""
Tests for the R integration wrapper.

These tests check the R session management and data conversion functions.
They are skipped if rpy2 is not installed.
"""

import pytest

# === Pure-Python tests (no R required) ===


def test_ver_basic():
    """_ver parses simple dotted version strings into integer tuples."""
    from soundscapy.r_wrapper._r_wrapper import _ver

    assert _ver("3.6") == (3, 6)
    assert _ver("2.0.0") == (2, 0, 0)
    assert _ver("1.1") == (1, 1)


def test_ver_avoids_lexicographic_pitfall():
    """_ver must compare 1.10 as greater than 1.2 (not less, as strings would)."""
    from soundscapy.r_wrapper._r_wrapper import _ver

    assert _ver("1.10") > _ver("1.2")
    assert _ver("2.0.0") > _ver("1.9.9")
    assert _ver("3.6.0") == _ver("3.6.0")


# === End-to-end R tests ===


@pytest.mark.optional_deps("r")
class TestRWrapper:
    """Test the R wrapper functionality."""

    def test_initialize_r_session(self):
        """Test R session initialization."""
        from soundscapy.r_wrapper._r_wrapper import (
            initialize_r_session,
        )

        # This should not raise if R is available
        res = initialize_r_session()

        assert res is not None, "R session should be initialized successfully"
        assert res["r_session"] == "active", "R session should be active"
        assert res["circe_package"] == "embedded", "CircE should be loaded from embedded scripts"

    def test_reset_r_session(self):
        """Test R session package unloading."""
        from soundscapy.r_wrapper._r_wrapper import reset_r_session

        # This should not raise if R session is active
        res = reset_r_session()

        assert res, "R session packages should be unloaded successfully"

    def test_r_session_reinitialization(self):
        """Test that the R session can be reinitialized after reset."""
        from soundscapy.r_wrapper._r_wrapper import (
            initialize_r_session,
            reset_r_session,
        )

        # First initialize the R session
        res = initialize_r_session()
        assert res is not None, "R session should be initialized successfully"

        # Now reset it (unload packages; R process keeps running)
        reset_res = reset_r_session()
        assert reset_res, "R session packages should be unloaded successfully"

        # Reinitialize the R session
        reinit_res = initialize_r_session()
        assert reinit_res is not None, "R session should be reinitialized successfully"

    def test_check_sn_package(self):
        """Test that the R 'sn' package is available when R deps are installed."""
        import soundscapy.r_wrapper as sspyr

        # Should not raise — this test only runs (via optional_deps("r")) when
        # rpy2 is present and the tox commands_pre has installed sn.
        sspyr._r_wrapper.check_sn_package()

    def test_check_circe_package(self):
        """Test that the embedded CircE scripts are available when R deps are installed."""
        import soundscapy.r_wrapper as sspyr

        # Should not raise — this test only runs (via optional_deps("r")) when
        # rpy2 is present and the local embedded CircE scripts are available.
        sspyr._r_wrapper.check_circe_package()
