"""
Tests for the R integration wrapper.

These tests check the R session management and data conversion functions.
They are skipped if rpy2 is not installed.
"""

import pytest
from unittest.mock import patch

# Comment out skip marker to test our implementation
# pytestmark = pytest.mark.skip(reason="SPI module still in development")


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
        from soundscapy.spi._r_wrapper import (
            initialize_r_session,
            is_session_active,
            shutdown_r_session,
        )

        # First ensure session is not active
        if is_session_active():
            shutdown_r_session()

        # Initialize session
        session_info = initialize_r_session()

        # Verify session is active
        assert is_session_active()
        assert session_info is not None
        assert session_info["r_session"] == "active"
        assert session_info["sn_package"] is not None
        assert session_info["stats_package"] is not None
        assert "rpy2_version" in session_info
        assert "r_version" in session_info
        assert "sn_version" in session_info

        # Clean up
        shutdown_r_session()

    def test_shutdown_r_session(self):
        """Test R session cleanup."""
        from soundscapy.spi._r_wrapper import (
            initialize_r_session,
            is_session_active,
            shutdown_r_session,
        )

        # First ensure session is active
        if not is_session_active():
            initialize_r_session()

        # Shutdown session
        result = shutdown_r_session()

        # Verify session is not active
        assert result is True
        assert not is_session_active()

        # Test idempotence - shutting down an already shutdown session
        result = shutdown_r_session()
        assert result is True  # Should return True, even if no session was active

    def test_r_session_reinitialization(self):
        """Test that the R session can be reinitialized after shutdown."""
        from soundscapy.spi._r_wrapper import (
            initialize_r_session,
            is_session_active,
            shutdown_r_session,
        )

        # First ensure session is not active
        if is_session_active():
            shutdown_r_session()

        # Initialize session first time
        session_info1 = initialize_r_session()
        assert is_session_active()

        # Shutdown session
        shutdown_r_session()
        assert not is_session_active()

        # Reinitialize session
        session_info2 = initialize_r_session()
        assert is_session_active()

        # Verify that both initializations returned valid session info
        assert session_info1 is not None
        assert session_info2 is not None
        assert session_info1["r_session"] == session_info2["r_session"] == "active"

        # Clean up
        shutdown_r_session()

    def test_get_r_session(self):
        """Test getting the R session and packages."""
        from soundscapy.spi._r_wrapper import (
            get_r_session,
            is_session_active,
            shutdown_r_session,
        )

        # First ensure session is not active
        if is_session_active():
            shutdown_r_session()

        # Get session (should initialize if not active)
        r_session, sn_package, stats_package = get_r_session()

        # Verify session objects
        assert r_session is not None
        assert sn_package is not None
        assert stats_package is not None
        assert is_session_active()

        # Clean up
        shutdown_r_session()

    def test_r_session_context(self):
        """Test the R session context manager."""
        from soundscapy.spi._r_wrapper import (
            r_session_context,
            is_session_active,
            shutdown_r_session,
        )

        # First ensure session is not active
        if is_session_active():
            shutdown_r_session()

        # Use context manager
        with r_session_context() as (r_session, sn_package, stats_package):
            # Verify session is active inside context
            assert is_session_active()
            assert r_session is not None
            assert sn_package is not None
            assert stats_package is not None

        # Verify session is still active after context exit
        # (context manager doesn't shut down the session to allow reuse)
        assert is_session_active()

        # Clean up
        shutdown_r_session()

    @patch("rpy2.robjects.packages.importr")
    def test_session_initialization_failure(self, mock_importr):
        """Test error handling during session initialization."""
        mock_importr.side_effect = Exception("Test exception - package not found")

        from soundscapy.spi._r_wrapper import initialize_r_session, shutdown_r_session, is_session_active

        # Ensure session is not active
        if is_session_active():
            shutdown_r_session()

        # Attempt to initialize session with mocked error
        with pytest.raises(RuntimeError) as excinfo:
            initialize_r_session()

        # Verify error message and session state
        assert "Failed to initialize R session" in str(excinfo.value)
        assert not is_session_active()

    # Data conversion tests will be added in Phase 1D
    @pytest.mark.skip(reason="To be implemented in Phase 1D")
    def test_python_to_r_conversion(self):
        """Test conversion of Python objects to R objects."""
        pass

    @pytest.mark.skip(reason="To be implemented in Phase 1D")
    def test_r_to_python_conversion(self):
        """Test conversion of R objects to Python objects."""
        pass
