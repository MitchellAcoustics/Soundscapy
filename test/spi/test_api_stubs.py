"""
Tests for the SPI module API stubs.

These tests verify that the API stubs exist and raise NotImplementedError
until they are properly implemented in later phases.
"""

import pytest
import numpy as np

# Skip all tests in this file as SPI module is still in development
pytestmark = pytest.mark.skip(reason="SPI module still in development")


@pytest.mark.optional_deps("spi")
class TestApiStubs:
    """Test the SPI module API stubs."""
    
    def test_skew_normal_distribution_stub(self):
        """Test that SkewNormalDistribution stub exists and raises NotImplementedError."""
        from soundscapy.spi import SkewNormalDistribution
        
        with pytest.raises(NotImplementedError) as excinfo:
            distribution = SkewNormalDistribution()
        
        assert "To be implemented in Phase 2" in str(excinfo.value)
    
    def test_fit_skew_normal_stub(self):
        """Test that fit_skew_normal stub exists and raises NotImplementedError."""
        from soundscapy.spi import fit_skew_normal
        
        test_data = np.random.randn(100, 2)
        
        with pytest.raises(NotImplementedError) as excinfo:
            distribution = fit_skew_normal(test_data)
        
        assert "To be implemented in Phase 2" in str(excinfo.value)
    
    def test_calculate_spi_stub(self):
        """Test that calculate_spi stub exists and raises NotImplementedError."""
        from soundscapy.spi import calculate_spi
        
        with pytest.raises(NotImplementedError) as excinfo:
            spi_score = calculate_spi(None, None)
        
        assert "To be implemented in Phase 3" in str(excinfo.value)
    
    def test_calculate_spi_from_data_stub(self):
        """Test that calculate_spi_from_data stub exists and raises NotImplementedError."""
        from soundscapy.spi import calculate_spi_from_data
        
        test_data1 = np.random.randn(100, 2)
        test_data2 = np.random.randn(100, 2)
        
        with pytest.raises(NotImplementedError) as excinfo:
            spi_score = calculate_spi_from_data(test_data1, test_data2)
        
        assert "To be implemented in Phase 3" in str(excinfo.value)