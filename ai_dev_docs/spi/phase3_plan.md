# SPI Feature - Phase 3: SPI Calculation (TDD Approach)

This document outlines the test-driven development (TDD) plan for Phase 3 of the SPI feature, focusing on implementing the Soundscape Perception Indices (SPI) calculation using the multivariate skew-normal distribution implementation from Phase 2.

## TDD Process Overview

For each component, we will follow this process:
1. Study the reference implementation from the research paper
2. Write failing tests that define expected behavior
3. Implement minimal code to make tests pass
4. Refactor while maintaining passing tests

## Objectives

1. Implement the bivariate Kolmogorov-Smirnov distance metric
2. Implement the SPI score calculation
3. Create high-level functions for SPI calculation from data
4. Ensure mathematical correctness against reference implementation

## TDD Implementation Plan

### 1. Kolmogorov-Smirnov Distance - TDD Cycle

#### 1.1 Study Bivariate KS Implementation
Study the research paper and original implementation:
- Understand the bivariate Kolmogorov-Smirnov distance calculation
- Document the algorithm and parameters
- Identify edge cases and performance considerations

#### 1.2 Write KS Distance Tests
Create tests in `test/spi/test_metrics.py`:

Test cases:
- Test KS distance between identical distributions
- Test KS distance between clearly different distributions
- Test with various sample sizes
- Test with edge cases (e.g., highly skewed distributions)
- Test against reference values from the research implementation

#### 1.3 Implement KS Distance Function
Add a minimal implementation in `metrics.py`:
- Function to calculate bivariate KS distance
- Parameter validation and error handling
- Sampling approach for distribution comparison

#### 1.4 Refactor
- Optimize calculation for large samples
- Improve numerical stability
- Add comprehensive docstrings

### 2. SPI Score Calculation - TDD Cycle

#### 2.1 Study SPI Calculation
Study the research paper and original implementation:
- Understand the SPI score calculation formula
- Document the conversion from KS distance to SPI score
- Identify parameter constraints and edge cases

#### 2.2 Write SPI Calculation Tests
Add tests to `test/spi/test_metrics.py`:

Test cases:
- Test SPI score between identical distributions (should be 100)
- Test SPI score between maximally different distributions (should approach 0)
- Test with various distribution parameters
- Test against reference values from the research implementation

#### 2.3 Implement SPI Calculation
Add a minimal implementation in `metrics.py`:
- Function to calculate SPI score from two distributions
- Parameter validation and error handling
- Integration with KS distance calculation

#### 2.4 Refactor
- Optimize calculation
- Add comprehensive docstrings

### 3. High-Level SPI Functions - TDD Cycle

#### 3.1 Design High-Level API
Define high-level functions for SPI calculation:
- `calculate_spi()` - Calculate SPI between two distributions
- `calculate_spi_from_data()` - Calculate SPI directly from datasets

#### 3.2 Write High-Level Function Tests
Add tests to `test/spi/test_metrics.py`:

Test cases:
- Test `calculate_spi()` with various distribution pairs
- Test `calculate_spi_from_data()` with various datasets
- Test parameter validation and error handling
- Test integration with distribution fitting

#### 3.3 Implement High-Level Functions
Add minimal implementations in `metrics.py`:
- `calculate_spi()` function
- `calculate_spi_from_data()` function
- Parameter validation and error handling

#### 3.4 Refactor
- Improve error messages
- Optimize workflow
- Add comprehensive docstrings

### 4. Integration Testing - TDD Cycle

#### 4.1 Study End-to-End Workflow
Design end-to-end workflows for SPI calculation:
- Data preparation
- Distribution fitting
- SPI calculation
- Result interpretation

#### 4.2 Write Integration Tests
Create tests in `test/spi/test_integration.py`:

Test cases:
- Test end-to-end workflow with realistic data
- Test with various data types (numpy arrays, pandas DataFrames)
- Test error handling in integrated workflow
- Test consistency with the paper's results

#### 4.3 Address Integration Issues
Make necessary adjustments to ensure smooth integration:
- Fix any compatibility issues
- Standardize parameter handling
- Improve error messages

#### 4.4 Refactor
- Optimize the integrated workflow
- Ensure consistent API
- Add comprehensive documentation

## Testing Strategy

### Test Structure

```
test/spi/
├── test_metrics.py       # Tests for KS distance and SPI calculation
├── test_integration.py   # End-to-end integration tests
```

### Testing Approaches

#### Reference Value Tests
These tests verify our implementation against known reference values:
- Generate reference SPI scores using the original implementation
- Implement our calculation with the same inputs
- Compare results
- Document expected values in the tests

#### Edge Case Tests
These tests verify behavior in challenging cases:
- Identical distributions (SPI should be 100)
- Completely different distributions (SPI should approach 0)
- High-dimensional data
- Very small datasets
- Highly skewed distributions

#### Performance Tests
These tests verify performance with larger datasets:
- Test with increasing sample sizes
- Monitor calculation time
- Ensure reasonable performance with realistic data sizes

## Test Data

We will create several test datasets:
1. **Simple 2D datasets** for basic functionality testing
2. **Reference datasets** from the research paper
3. **Synthetic datasets** with known properties
4. **Realistic soundscape data** examples (if available)

Each dataset will be accompanied by reference SPI values where possible.

## API Design

The final public API will include:

```python
def calculate_ks_distance(dist1, dist2, n_samples=1000):
    """Calculate the bivariate Kolmogorov-Smirnov distance between two distributions.
    
    Parameters
    ----------
    dist1 : SkewNormalDistribution
        First distribution
    dist2 : SkewNormalDistribution
        Second distribution
    n_samples : int, optional
        Number of samples to use for distance calculation (default: 1000)
        
    Returns
    -------
    float
        The Kolmogorov-Smirnov distance, between 0 and 1
    """
    
def calculate_spi(dist1, dist2, n_samples=1000):
    """Calculate the Soundscape Perception Index (SPI) between two distributions.
    
    Parameters
    ----------
    dist1 : SkewNormalDistribution
        Test distribution
    dist2 : SkewNormalDistribution
        Target distribution
    n_samples : int, optional
        Number of samples to use for calculation (default: 1000)
        
    Returns
    -------
    float
        SPI score, between 0 and 100
    """
    
def calculate_spi_from_data(test_data, target_data):
    """Calculate the SPI score between two datasets.
    
    Parameters
    ----------
    test_data : array-like
        Test dataset, shape (n_samples, 2)
    target_data : array-like
        Target dataset, shape (n_samples, 2)
        
    Returns
    -------
    float
        SPI score, between 0 and 100
    """
```

## Development Sequence

Following strict TDD, development will proceed in this sequence:

### Phase 3A: Kolmogorov-Smirnov Distance
1. Write tests for KS distance calculation
2. Implement minimal KS distance function
3. Refactor for performance and stability

### Phase 3B: SPI Score Calculation
1. Write tests for SPI score calculation
2. Implement minimal SPI calculation function
3. Refactor for clarity and performance

### Phase 3C: High-Level Functions
1. Write tests for high-level API functions
2. Implement high-level functions
3. Refactor for usability

### Phase 3D: Integration Testing
1. Write integration tests
2. Address any integration issues
3. Refactor for a smooth workflow

## Success Criteria

Phase 3 will be considered complete when:

1. Bivariate KS distance calculation is correctly implemented
2. SPI score calculation matches the reference implementation
3. High-level API functions provide a user-friendly interface
4. Integration tests pass with realistic data
5. All code is well-documented
6. All tests pass with adequate coverage

## Next Steps

After completing Phase 3, we'll move to Phase 4, focusing on comprehensive documentation, usability improvements, and final integration testing.