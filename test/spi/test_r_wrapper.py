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


class TestRWrapper:
    """Test the R wrapper session management."""

    def test_get_r_session_returns_active(self):
        """get_r_session() should return an active, initialised session."""
        from soundscapy.r_wrapper._r_wrapper import get_r_session

        r = get_r_session()
        assert r.active, "R session should be active after get_r_session()"
        assert r.sn is not None, "sn package should be loaded"
        assert r.base is not None, "base package should be loaded"

    def test_reset_r_session(self):
        """reset_r_session() should succeed and deactivate the session."""
        import soundscapy.r_wrapper._r_wrapper as rw

        res = rw.reset_r_session()
        assert res, "reset_r_session() should return True on success"
        # Access _state via module to get the newly-bound object after reset.
        assert not rw._state.active, "session should be inactive after reset"

    def test_r_session_reinitialization(self):
        """The session can be reset and then re-initialised transparently."""
        import soundscapy.r_wrapper._r_wrapper as rw

        rw.get_r_session()
        assert rw._state.active

        rw.reset_r_session()
        assert not rw._state.active

        rw.get_r_session()
        assert rw._state.active, (
            "session should be active again after re-initialisation"
        )
