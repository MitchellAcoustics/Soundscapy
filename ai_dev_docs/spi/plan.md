# SPI Feature Development Plan

## Overview

This document outlines the development plan for integrating the Soundscape Perception Indices (SPI) feature into Soundscapy, based on the research described in the paper repository (J2401_JASA_SSID-Single-Index). The implementation will use the R 'sn' package via rpy2 as an optional dependency.

## Goals

1. Implement SPI calculation functionality using the R 'sn' package
2. Provide a clean Python API that abstracts away R implementation details
3. Integrate with Soundscapy's optional dependency system
4. Create comprehensive tests to validate functionality
5. Document the feature thoroughly

## Non-Goals (for this phase)

1. Implementing SPI plotting functionality
2. Implementing optimization capabilities
3. Creating a pure Python implementation of multivariate skew-normal distribution
4. Exposing the full 'sn' package functionality

## Technical Design

### Module Architecture

```
src/soundscapy/spi/
├── __init__.py          # Public API
├── _optionals.py        # Optional dependency management for R/rpy2
├── _r_wrapper.py        # "Cranky" layer - R/rpy2 interface
├── distributions.py     # Skew-normal distribution interfaces
├── metrics.py           # SPI calculation functions
├── utils.py             # Utility functions
```

### Layered Architecture

Following the scientific Python design principles, we'll implement a layered architecture:

1. **"Cranky" Layer** (`_r_wrapper.py`):
   - Handles direct interaction with R via rpy2
   - Manages R session initialization and cleanup
   - Converts between R and Python data structures
   - Provides direct wrappers for specific R 'sn' functions
   - Implements robust error handling and validation
   - Not intended for direct user interaction

2. **"Friendly" Layer** (Public API in `__init__.py`):
   - Provides a clean, Pythonic interface to SPI functionality
   - Handles input validation and preprocessing
   - Provides sensible defaults for common use cases
   - Returns well-structured Python objects
   - Includes clear documentation and examples

### Key Components

#### 1. Optional Dependency Management

Extend Soundscapy's optional dependency system to include R and rpy2:

```python
# In _optionals.py
OPTIONAL_IMPORTS = {
    "rpy2": ("rpy2", "Provides R integration for SPI calculation"),
    "r_sn": (None, "R 'sn' package for skew-normal distributions")
}
```

#### 2. R Interface (`_r_wrapper.py`)

Core functions:
- `initialize_r_session()`: Set up R environment and load 'sn' package
- `shutdown_r_session()`: Clean up R resources
- `r_msn_fit()`: Fit multivariate skew-normal to data
- `r_msn_sample()`: Sample from multivariate skew-normal
- `r_msn_density()`: Calculate density values
- Data conversion utilities between R and Python

#### 3. Distribution Interface (`distributions.py`)

Define classes:
- `SkewNormalDistribution`: Represents a fitted multivariate skew-normal distribution
  - Properties: location, scale, shape (mean, covariance, alpha in SN terms)
  - Methods: sample(), pdf(), plot() (future)
  
Functions:
- `fit_skew_normal(data)`: Fit skew-normal to data
- `create_skew_normal(location, scale, shape)`: Create distribution from parameters

#### 4. SPI Calculation (`metrics.py`)

Functions:
- `calculate_ks_distance(dist1, dist2)`: Calculate K-S distance between distributions
- `calculate_spi(test_dist, target_dist)`: Calculate SPI score
- `calculate_spi_from_data(test_data, target_data)`: Calculate SPI score directly from data

### Public API (in `__init__.py`)

```python
# Core functionality
from .distributions import fit_skew_normal, create_skew_normal, SkewNormalDistribution
from .metrics import calculate_spi, calculate_spi_from_data

# Version and metadata
__version__ = "0.1.0"
```

## Testing Strategy

We'll adopt a test-first development approach with the following components:

### 1. Unit Tests

- Test R wrapper functions with mock data
- Test distribution fitting and sampling
- Test SPI calculation with known distributions
- Test error handling and edge cases

### 2. Integration Tests

- Test end-to-end workflow from data to SPI score
- Test compatibility with existing Soundscapy functionality

### 3. Test Data

- Create synthetic datasets with known distributions
- Use reference datasets from the paper (if available)
- Create test fixtures for common test scenarios

### 4. Test Structure

```
test/spi/
├── __init__.py
├── test_r_wrapper.py
├── test_distributions.py
├── test_metrics.py
├── conftest.py          # Test fixtures and utilities
```

## Development Phases

The SPI feature will be implemented in four phases, with detailed plans available in separate documents:

### Phase 1: Setup and R Integration ([phase1_plan.md](phase1_plan.md))
1. Create module structure
2. Implement optional dependency management for R/rpy2
3. Create R session initialization and cleanup functions
4. Add basic data conversion utilities
5. Write tests for R integration

### Phase 2: Distribution Functions ([phase2_plan.md](phase2_plan.md), [phase2_msn_details.md](phase2_msn_details.md))
1. Implement R wrappers for MSN fitting and sampling
2. Create SkewNormalDistribution class
3. Implement fit_skew_normal and create_skew_normal functions
4. Write tests for distribution functionality

### Phase 3: SPI Calculation ([phase3_plan.md](phase3_plan.md))
1. Implement K-S distance calculation
2. Implement SPI score calculation
3. Create high-level SPI calculation functions
4. Write tests for SPI calculation

### Phase 4: Documentation and Integration ([phase4_plan.md](phase4_plan.md))
1. Add comprehensive docstrings
2. Create usage examples
3. Update Soundscapy documentation
4. Final integration testing

## Dependencies

- R (>= 4.0.0)
- R 'sn' package
- rpy2 (>= 3.5.0)
- numpy
- scipy
- pandas

## Challenges and Considerations

### Installation Complexity

- Document R and rpy2 installation process
- Consider providing helper scripts for installation
- Add clear error messages for missing dependencies

### Cross-Platform Compatibility

- Test on multiple platforms (Linux, macOS, Windows)
- Handle platform-specific issues with R/rpy2

### Performance

- Monitor performance, especially for large datasets
- Consider caching or memoization for repeated calculations

### Error Handling

- Provide clear error messages for R-related issues
- Gracefully handle R exceptions
- Validate inputs thoroughly

## Future Enhancements

- Implement Python-native MSN distribution
- Add plotting capabilities
- Add optimization functionality