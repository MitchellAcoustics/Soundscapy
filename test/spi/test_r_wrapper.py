"""
Tests for the R integration wrapper.

These tests check the R session management and data conversion functions.
They are skipped if rpy2 is not installed.
"""

import os

import pytest


def test_initialize_r_session_fails():
    """Test that R session initialization fails if R is not available."""
    # Skip if dependencies are actually installed
    if os.environ.get("SPI_DEPS") == "1":
        pytest.skip("SPI dependencies are installed")

    from soundscapy.spi._r_wrapper import initialize_r_session

    # Simulate R not being available
    with pytest.raises(ImportError) as excinfo:
        initialize_r_session()

    # Check for helpful error message
    assert "R installation" in str(excinfo.value)
    assert "install.packages('R')" in str(excinfo.value)


@pytest.mark.optional_deps("spi")
class TestRWrapper:
    """Test the R wrapper functionality."""

    def test_module_structure(self):
        """Test that the module structure exists."""
        import soundscapy.spi._r_wrapper

        # Module should exist but functions will be implemented later
        assert soundscapy.spi._r_wrapper is not None

    def test_initialize_r_session(self):
        """Test R session initialization."""
        from soundscapy.spi._r_wrapper import initialize_r_session

        # This should not raise if R is available
        res = initialize_r_session()

        assert res is not None, "R session should be initialized successfully"
        assert res["r_session"] == "active", "R session should be active"

    def test_shutdown_r_session(self):
        """Test R session cleanup."""
        from soundscapy.spi._r_wrapper import shutdown_r_session

        # This should not raise if R session is active
        res = shutdown_r_session()

        assert res, "R session should be shut down successfully"

    def test_r_session_reinitialization(self):
        """Test that the R session can be reinitialized after shutdown."""
        from soundscapy.spi._r_wrapper import initialize_r_session, shutdown_r_session

        # First initialize the R session
        res = initialize_r_session()
        assert res is not None, "R session should be initialized successfully"

        # Now shut it down
        shutdown_res = shutdown_r_session()
        assert shutdown_res, "R session should be shut down successfully"

        # Reinitialize the R session
        reinit_res = initialize_r_session()
        assert reinit_res is not None, "R session should be reinitialized successfully"

    def test_check_sn_package(self):
        """Test that the R 'sn' package availability is checked."""
        # Skip if dependencies are actually installed

        if os.environ.get("SPI_DEPS") == "1":
            from soundscapy.spi import _r_wrapper

            _r_wrapper.check_sn_package()

        else:
            with pytest.raises(ImportError) as excinfo:
                from soundscapy.spi import _r_wrapper

                _r_wrapper.check_sn_package()

            assert "R package 'sn'" in str(excinfo.value)
            assert "install.packages('sn')" in str(excinfo.value)
