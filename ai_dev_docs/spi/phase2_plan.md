# SPI Feature - Phase 2: Multivariate Skew-Normal Distribution (TDD Approach)

This document outlines the test-driven development (TDD) plan for Phase 2 of the SPI feature, focusing on implementing the multivariate skew-normal distribution functionality using the R integration established in Phase 1.

## TDD Process Overview

For each component, we will follow this process:
1. Study the R sn package implementation for the functionality we need
2. Write failing tests that define expected behavior
3. Implement minimal code to make tests pass
4. Refactor while maintaining passing tests

## Objectives

1. Implement R wrappers for multivariate skew-normal functions
2. Create the SkewNormalDistribution class with a clean API
3. Implement distribution fitting and sampling capabilities
4. Create factory functions for convenient distribution creation
5. Ensure mathematical correctness against reference implementations

## TDD Implementation Plan

### 1. R Function Wrappers - TDD Cycle

#### 1.1 Study R sn Functions
Study the R sn package documentation and implementation:
- `msn.mle()` - Maximum likelihood estimation
- `rmsn()` - Random sampling from multivariate skew-normal
- `dmsn()` - Multivariate skew-normal density function

Document the function signatures, parameters, return values, and behavior.

#### 1.2 Write R Wrapper Tests
Create tests in `test/spi/test_r_msn.py`:

Test cases:
- Test wrapping `msn.mle()` with realistic data
- Test wrapping `rmsn()` with various parameters
- Test wrapping `dmsn()` with sample points
- Test error handling for invalid parameters
- Test edge cases (e.g., high dimensions, degenerate cases)

#### 1.3 Implement R Function Wrappers
Add minimal implementations in `_r_wrapper.py`:
- Function to call R's `msn.mle()`
- Function to call R's `rmsn()`
- Function to call R's `dmsn()`
- Parameter validation and error handling

#### 1.4 Refactor
- Improve error messages
- Optimize data conversion
- Add comprehensive docstrings

### 2. SkewNormalDistribution Class - TDD Cycle

#### 2.1 Design SkewNormalDistribution API
Define the public API for the SkewNormalDistribution class:
- Constructor
- Properties for parameters (location, scale, shape)
- Methods for sampling and density calculation
- String representation

#### 2.2 Write SkewNormalDistribution Tests
Create tests in `test/spi/test_distributions.py`:

Test cases:
- Test creating a distribution with valid parameters
- Test parameter validation (types, shapes, constraints)
- Test property getters
- Test sampling method
- Test PDF calculation
- Test string representation
- Test for invalid parameters

#### 2.3 Implement SkewNormalDistribution
Add a minimal implementation in `distributions.py`:
- Constructor with parameter validation
- Properties to access parameters
- Methods for sampling and density calculation
- String representation

#### 2.4 Refactor
- Improve parameter validation
- Optimize internal data structures
- Add comprehensive docstrings

### 3. Factory Functions - TDD Cycle

#### 3.1 Design Factory Functions
Define factory functions for creating distributions:
- `fit_skew_normal()` - Fit a distribution to data
- `create_skew_normal()` - Create a distribution with given parameters

#### 3.2 Write Factory Function Tests
Add tests to `test/spi/test_distributions.py`:

Test cases:
- Test `fit_skew_normal()` with various datasets
- Test `create_skew_normal()` with various parameters
- Test parameter validation and error handling
- Test edge cases (e.g., small datasets, high dimensions)

#### 3.3 Implement Factory Functions
Add minimal implementations in `distributions.py`:
- `fit_skew_normal()` function
- `create_skew_normal()` function
- Parameter validation and error handling

#### 3.4 Refactor
- Improve error messages
- Optimize fitting procedure
- Add comprehensive docstrings

### 4. Mathematical Validation - TDD Cycle

#### 4.1 Study Mathematical Properties
Research the mathematical properties of multivariate skew-normal distributions:
- Moments (mean, variance, skewness)
- Marginal distributions
- Conditioning
- Transformations

#### 4.2 Write Mathematical Validation Tests
Create tests in `test/spi/test_math_validation.py`:

Test cases:
- Test mean and covariance calculations
- Test against known reference distributions
- Test marginal distributions
- Test statistical properties of samples
- Test consistency with R implementation

#### 4.3 Implement Statistical Methods
Add statistical methods to `SkewNormalDistribution`:
- Mean calculation
- Covariance calculation
- Method to compute statistical properties

#### 4.4 Refactor
- Optimize calculations
- Add comprehensive docstrings

## Testing Strategy

### Test Structure

```
test/spi/
├── test_r_msn.py           # Tests for R function wrappers
├── test_distributions.py   # Tests for distribution class and factory functions
├── test_math_validation.py # Tests for mathematical correctness
```

### Testing Approaches

#### Verification Tests
These tests ensure that our implementation matches the R implementation:
- Generate reference data in R
- Apply our implementation to the same data
- Compare results
- Include reference values in test data

#### Parameter Validation Tests
These tests ensure that our API correctly validates parameters:
- Test with invalid parameter types
- Test with invalid parameter shapes
- Test with invalid parameter values (e.g., non-positive-definite scale matrix)
- Verify error messages

#### Sample-Based Tests
These tests verify statistical properties using sampling:
- Generate large samples
- Calculate empirical statistics
- Compare to theoretical values
- Test for expected statistical properties

#### Edge Case Tests
These tests verify behavior in challenging cases:
- High dimensions
- Nearly singular covariance matrices
- Extreme skewness values
- Very small datasets

## Test Data

We will create several test datasets:
1. **Simple 2D datasets** for basic functionality testing
2. **Reference datasets** from the literature
3. **Synthetic datasets** with known properties
4. **Edge case datasets** for testing robustness

Each dataset will be accompanied by reference results from the R implementation for validation.

## API Design

The final public API will include:

```python
class SkewNormalDistribution:
    """Represents a multivariate skew-normal distribution.
    
    Parameters
    ----------
    location : array-like
        Location parameter (xi), shape (n_dimensions,)
    scale : array-like
        Scale matrix (Omega), shape (n_dimensions, n_dimensions)
    shape : array-like
        Shape parameter (alpha), shape (n_dimensions,)
    """
    
    def __init__(self, location, scale, shape):
        # Parameter validation and storage
        
    @property
    def location(self):
        """Get the location parameter (xi)."""
        
    @property
    def scale(self):
        """Get the scale matrix (Omega)."""
        
    @property
    def shape(self):
        """Get the shape parameter (alpha)."""
        
    @property
    def dimension(self):
        """Get the dimension of the distribution."""
        
    def sample(self, n_samples=1):
        """Generate samples from the distribution."""
        
    def pdf(self, points):
        """Calculate probability density function at points."""
        
    def __repr__(self):
        """String representation of the distribution."""

def fit_skew_normal(data, *, initial_params=None):
    """Fit a multivariate skew-normal distribution to data."""
    
def create_skew_normal(location, scale, shape):
    """Create a multivariate skew-normal distribution with given parameters."""
```

## Development Sequence

Following strict TDD, development will proceed in this sequence:

### Phase 2A: R Function Wrappers
1. Write tests for R function wrappers
2. Implement minimal R function wrappers
3. Refactor for robustness

### Phase 2B: SkewNormalDistribution Implementation
1. Write tests for SkewNormalDistribution
2. Implement SkewNormalDistribution class
3. Refactor for clarity and performance

### Phase 2C: Factory Functions
1. Write tests for factory functions
2. Implement factory functions
3. Refactor for usability

### Phase 2D: Mathematical Validation
1. Write tests for mathematical properties
2. Implement statistical methods
3. Validate against R implementation

## Success Criteria

Phase 2 will be considered complete when:

1. R function wrappers correctly call the R functions
2. SkewNormalDistribution class provides a clean API
3. Factory functions provide convenient creation methods
4. Mathematical validation tests pass
5. All code is well-documented
6. All tests pass with adequate coverage

## Next Steps

After completing Phase 2, we'll move to Phase 3, implementing the SPI calculation functionality using the multivariate skew-normal distribution implementation from Phase 2, continuing with the TDD approach.