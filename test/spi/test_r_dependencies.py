"""
Tests for the R dependency checking functionality.

These tests verify that the module properly checks for R and the 'sn' package
in addition to checking for rpy2.
"""

import pytest
from unittest.mock import patch, MagicMock

# Skip all tests if rpy2 is not importable at all (basic dependency check)
rpy2_importable = False
try:
    import rpy2
    rpy2_importable = True
except ImportError:
    pass


@pytest.mark.optional_deps("spi")
class TestRDependencies:
    """Test the R dependency checking functionality."""
    
    @pytest.mark.skipif(not rpy2_importable, reason="rpy2 is not installed")
    def test_check_r_availability(self):
        """Test that R availability is checked."""
        from soundscapy.spi import _r_wrapper
        
        # This should not raise if R is available
        _r_wrapper.check_r_availability()
    
    @pytest.mark.skipif(not rpy2_importable, reason="rpy2 is not installed")
    def test_check_sn_package(self):
        """Test that the R 'sn' package availability is checked."""
        from soundscapy.spi import _r_wrapper
        
        # This should not raise if 'sn' package is available
        _r_wrapper.check_sn_package()
    
    @patch("rpy2.robjects.packages.importr")
    def test_missing_sn_package(self, mock_importr):
        """Test error when 'sn' package is missing."""
        # Skip if rpy2 is not available at all
        if not rpy2_importable:
            pytest.skip("rpy2 is not installed")
        
        # Make importr raise an exception
        mock_importr.side_effect = Exception("R package 'sn' is not installed")
        
        from soundscapy.spi import _r_wrapper
        
        # Trying to check sn package should raise
        with pytest.raises(ImportError) as excinfo:
            _r_wrapper.check_sn_package()
        
        # Check for helpful error message
        assert "R package 'sn'" in str(excinfo.value)
        assert "install.packages('sn')" in str(excinfo.value)
    
    @patch("rpy2.robjects.r")
    def test_missing_r_installation(self, mock_r):
        """Test error when R is not installed."""
        # Skip if rpy2 is not available at all
        if not rpy2_importable:
            pytest.skip("rpy2 is not installed")
        
        # Make any R operation raise an exception
        mock_r.side_effect = Exception("R home is not defined")
        
        from soundscapy.spi import _r_wrapper
        
        # Trying to check R availability should raise
        with pytest.raises(ImportError) as excinfo:
            _r_wrapper.check_r_availability()
        
        # Check for helpful error message
        assert "R installation" in str(excinfo.value)