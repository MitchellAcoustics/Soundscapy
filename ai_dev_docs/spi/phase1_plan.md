# SPI Feature - Phase 1: Setup and R Integration (TDD Approach)

This document outlines the test-driven development (TDD) plan for Phase 1 of the SPI feature, focusing on setting up the module structure and implementing R integration in accordance with Soundscapy's existing optional dependency system.

## TDD Process Overview

For each component, we will follow this process:

1. Study the R sn package behavior for the functionality we need
2. Write failing tests that define expected behavior
3. Implement minimal code to make tests pass
4. Refactor while maintaining passing tests

## Objectives

1. Create basic module structure for the SPI feature
2. Integrate with Soundscapy's optional dependency system for R/rpy2
3. Create R session initialization and cleanup functionality
4. Implement data conversion between R and Python

## Aligning with Soundscapy's Optional Dependency System

Based on the updated guidelines in CONTRIBUTING.md, we will follow Soundscapy's simplified approach:

1. **Define the optional dependency group** in `pyproject.toml`:

   ```toml
   [project.optional-dependencies]
   spi = [
       "rpy2>=3.5.0",
   ]
   ```

2. **Implement direct dependency checks** in `spi/__init__.py`:

   ```python
   # Check for Python dependencies directly
   try:
       import rpy2.robjects as robjects
   except ImportError as e:
       raise ImportError(
           "Soundscape perception indices calculation requires additional dependencies. "
           "Install with: pip install soundscapy[spi]"
       ) from e

   # Check for R dependencies
   try:
       # Code to check R and sn package availability
       r_version = robjects.r("R.version.string")[0]
       robjects.r("library(sn)")
   except Exception as e:
       raise ImportError(
           f"Error with R dependencies: {str(e)}. "
           "Please ensure R and the 'sn' package are installed."
       ) from e

   # Now import module components
   from .distributions import SkewNormalDistribution, fit_skew_normal
   from .metrics import calculate_spi
   
   __all__ = ["SkewNormalDistribution", "fit_skew_normal", "calculate_spi"]
   ```

3. **Add to top-level exports** in `soundscapy/__init__.py`:

   ```python
   # Try to import optional SPI module
   try:
       from soundscapy import spi
       from soundscapy.spi import (
           SkewNormalDistribution, fit_skew_normal, calculate_spi, calculate_spi_from_data,
       )
       __all__.extend([
           "spi", "SkewNormalDistribution", "fit_skew_normal", 
           "calculate_spi", "calculate_spi_from_data",
       ])
   except ImportError:
       # SPI module not available
       pass
   ```

4. **Testing strategy** will use the established markers:

   ```python
   @pytest.mark.optional_deps('spi')
   def test_spi_functionality():
       # Test code here
   ```

## TDD Implementation Plan

### 1. Setup and Initial Tests

#### 1.1 Create Basic Module Structure

```bash
src/soundscapy/spi/
├── __init__.py          # Module dependency check 
├── _r_wrapper.py        # Will contain R integration
├── distributions.py     # Will contain distribution classes
├── metrics.py           # Will contain SPI calculation
├── utils.py             # Will contain utility functions
```

#### 1.2 Write Package Configuration Tests

Create a test file to verify the package is configurable as expected:

```bash
test/test_spi_import.py
```

Test cases:

- Test importing SPI module directly
- Test importing SPI module components
- Test importing SPI module when dependencies are missing
- Test error messages for missing dependencies

### 2. R Dependency Integration - TDD Cycle

#### 2.1 Study R sn Requirements

- Identify required R version and sn package version
- Document R sn installation requirements

#### 2.2 Write Dependency Tests

Add tests to verify:

- Detection of missing rpy2
- Detection of missing R installation
- Detection of missing R sn package
- Proper error messages for each case

#### 2.3 Implement R Dependency Checks

Implement direct dependency checks in `spi/__init__.py`:

- First directly import rpy2 with try/except
- Then verify R is available through rpy2
- Finally check if R sn package is available
- Provide clear error messages with installation instructions

#### 2.4 Refactor

- Ensure consistent error messages
- Improve installation instructions

### 3. R Session Management - TDD Cycle

#### 3.1 Study R Session Behavior in rpy2

- Document rpy2's session management approach
- Understand resource management needs

#### 3.2 Write Session Management Tests

```bash
test/spi/test_r_wrapper.py
```

Test cases:

- Test session initialization (success case)
- Test session initialization (failure case)
- Test session cleanup
- Test session reinitialization

#### 3.3 Implement Session Management

Create basic R session management in `_r_wrapper.py`:

- Initialize R session and load packages
- Clean up resources
- Handle session state

#### 3.4 Refactor

- Improve error handling
- Add logging
- Optimize session management

### 4. Data Conversion - TDD Cycle

#### 4.1 Study Required R/Python Data Conversions

- Identify required data types for MSN functions
- Document conversion requirements

#### 4.2 Write Conversion Tests

Extend tests with:

- Test numpy array to R matrix conversion
- Test R matrix to numpy array conversion
- Test R list to Python dictionary conversion
- Test conversion edge cases

#### 4.3 Implement Conversion Functions

Add minimal conversion functions to `_r_wrapper.py`:

- Python to R conversions
- R to Python conversions
- Edge case handling

#### 4.4 Refactor

- Optimize conversion
- Improve error handling
- Add comprehensive docstrings

### 5. Public API Design - TDD Cycle

#### 5.1 Study Key R sn Functions

- Identify core functions needed for SPI calculation
- Document parameter and return value formats

#### 5.2 Write API Interface Tests

Test the proposed API interfaces (with NotImplementedError for now):

```python
# API stubs
def fit_skew_normal(data, *, initial_params=None):
    """Fit a multivariate skew-normal distribution to data."""
    raise NotImplementedError("To be implemented in Phase 2")
    
class SkewNormalDistribution:
    """Represents a multivariate skew-normal distribution."""
    
    @property
    def location(self):
        raise NotImplementedError("To be implemented in Phase 2")
        
    # Other properties and methods
```

These tests will verify the API design without implementation.

## Test Structure and Integration with Existing Tests

Following Soundscapy's updated testing approach:

```bash
test/
├── spi/
    ├── __init__.py            # Empty package file
    ├── conftest.py            # Test fixtures (if needed)
    ├── test_api_stubs.py      # API stub tests
    ├── test_r_dependencies.py # R dependency tests
    ├── test_r_wrapper.py      # R integration tests
```

Testing strategy will use these approaches as defined in the updated CONTRIBUTING.md:

1. **Optional Module Tests**
   - Tests inside the module's test directory
   - Only collected when dependencies are available (using `pytest_ignore_collect`)
   - No special markers needed for tests within the module directory

2. **Integration Tests with Markers**
   - Use `@pytest.mark.optional_deps('spi')`
   - Will be marked as xfail when dependencies are missing

3. **In-Development Module Tests**
   - Use module-level `pytestmark = pytest.mark.skip(reason="...")` to skip all tests
   - Prevent collection errors during early development
   - All SPI tests will use this approach initially while the feature is under development

## Development Sequence

Following strict TDD, development will proceed in this sequence:

### Phase 1A: Package Configuration

1. Write tests for package import behavior
2. Implement minimal package structure 
3. Update `pyproject.toml` with SPI optional dependency
4. Create test directory structure with appropriate skip markers

### Phase 1B: R Dependency Checks

1. Write tests for R dependency checking
2. Implement direct dependency checks in module `__init__.py`
3. Add clear error messages with installation instructions

### Phase 1C: R Session Management

1. Write tests for R session management
2. Implement session initialization and cleanup
3. Add error handling and state tracking

### Phase 1D: Data Conversion

1. Write tests for data conversion functions
2. Implement minimal conversion functions
3. Refactor for robustness and edge cases

### Phase 1E: API Design Validation

1. Create API interface stubs
2. Write tests for API design
3. Validate API meets requirements

## Success Criteria

Phase 1 will be considered complete when:

1. Soundscapy can import the SPI module with proper dependency checks
2. All tests for R session management pass
3. All tests for data conversion pass
4. API interfaces are designed and verified
5. Tests follow Soundscapy's optional dependency testing conventions
6. Tests skip gracefully when dependencies are unavailable

## Next Steps

After completing Phase 1, we'll move to Phase 2, implementing the multivariate skew-normal distribution functionality using the R integration established in Phase 1, continuing with the TDD approach.
