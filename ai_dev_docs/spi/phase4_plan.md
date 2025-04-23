# SPI Feature - Phase 4: Documentation and Integration

This document outlines the plan for Phase 4 of the SPI feature, focusing on comprehensive documentation, final integration with Soundscapy, and ensuring a polished user experience.

## Objectives

1. Create comprehensive API documentation with docstrings
2. Develop a tutorial notebook for the SPI functionality
3. Ensure seamless integration with the rest of Soundscapy
4. Create installation and configuration documentation

## Implementation Plan

### 1. API Documentation

Create comprehensive API documentation following Soundscapy's NumPy docstring style:

- Add detailed docstrings to all public functions and classes
- Include:
  - Parameter descriptions with types and defaults
  - Return value descriptions
  - Notes sections with implementation details
  - References to the original research paper
  - Examples that can be tested with xdoctest

Example docstring format:

```python
def calculate_spi(dist1, dist2, n_samples=1000):
    """Calculate the Soundscape Perception Index (SPI) between two distributions.
    
    Parameters
    ----------
    dist1 : SkewNormalDistribution
        Test distribution
    dist2 : SkewNormalDistribution
        Target distribution
    n_samples : int, optional
        Number of samples to use for calculation, by default 1000
        
    Returns
    -------
    float
        SPI score, between 0 and 100
        
    Notes
    -----
    The SPI score is calculated as 100 * (1 - D_KS), where D_KS is the
    bivariate Kolmogorov-Smirnov distance between the two distributions.
    
    References
    ----------
    .. [1] Mitchell, A., Aletta, F., Oberman, T., Kang, J. (2023)
           Making urban soundscape evaluation simpler ...
           Journal of the Acoustical Society of America, ...
           
    Examples
    --------
    >>> from soundscapy.spi import create_skew_normal, calculate_spi
    >>> dist1 = create_skew_normal([0, 0], [[1, 0], [0, 1]], [0, 0])
    >>> dist2 = create_skew_normal([0.5, 0.5], [[1, 0], [0, 1]], [0, 0])
    >>> spi = calculate_spi(dist1, dist2)
    >>> print(f"SPI score: {spi:.1f}")
    SPI score: 85.3
    """
```

### 2. Tutorial Notebook Development

Create a Jupyter notebook tutorial for the SPI functionality, placed in `docs/tutorials/`:

#### Notebook Structure: `SoundscapePerceptionIndices.ipynb`

1. **Introduction**
   - Overview of the SPI method
   - References to the original research
   - Use cases and applications

2. **Installation and Setup**
   - Installing the optional dependencies
   - Setting up R and required packages
   - Verifying the installation

3. **Basic SPI Calculation**
   - Creating skew-normal distributions
   - Calculating SPI between distributions
   - Interpreting the results

4. **Working with Soundscape Data**
   - Loading and preparing soundscape data
   - Fitting distributions to data
   - Calculating SPI from raw data

5. **Real-world Example**
   - A complete example with real or synthetic data
   - Interpretation of results
   - Practical applications

6. **Advanced Topics**
   - Working with different distribution parameters
   - Creating custom target distributions
   - Tips for efficient usage

The notebook should be designed to work with `nbmake` for automated testing:

```python
# Add this at the end of code cells to ensure nbmake can verify outputs
# Output should match expected value within tolerance
assert 80 <= spi <= 90, f"Expected SPI between 80 and 90, got {spi}"
```

### 3. Integration with Soundscapy

Ensure seamless integration with the rest of Soundscapy:

1. **Optional Dependency Setup**
   - Update `pyproject.toml` for SPI dependencies
   - Update `_optionals.py` for SPI module

2. **Top-level API Integration**
   - Update `soundscapy/__init__.py` with SPI exports
   - Ensure proper error handling for missing dependencies

3. **Module Structure**
   - Finalize the module structure
   - Ensure all imports work correctly
   - Create proper `__all__` definitions

4. **Installation Documentation**
   - Add SPI installation instructions to README
   - Create clear documentation for R dependencies

### 4. Documentation for mkdocs

Prepare documentation for inclusion in the Soundscapy documentation:

1. **API Reference**
   - Create a new page in `docs/reference/spi.md`
   - Document all public API functions and classes
   - Include mathematical background
   - Add installation instructions

2. **Update Main Documentation**
   - Add SPI to the feature list
   - Update navigation
   - Add cross-references where appropriate

3. **Citations and References**
   - Add proper citations to the original research
   - Update the CITATION.cff file if appropriate

## Deliverables

### 1. Code Documentation

- Comprehensive docstrings for all public components
- Example code snippets that work with xdoctest
- Internal documentation for complex algorithms

### 2. Tutorial Notebook

- `docs/tutorials/SoundscapePerceptionIndices.ipynb`
- Complete with explanations, code, and examples
- Working with nbmake for automated testing

### 3. API Reference Documentation

- `docs/reference/spi.md` with API documentation
- Mathematical background and method description
- Installation and setup instructions

### 4. Integration Components

- Updates to `pyproject.toml` for dependencies
- Updates to `_optionals.py` for optional imports
- Updates to top-level imports and exports

## Development Process

### Phase 4A: API Documentation
1. Add comprehensive docstrings to all public functions and classes
2. Include testable examples with xdoctest
3. Document internal components as appropriate

### Phase 4B: Tutorial Development
1. Create the tutorial notebook structure
2. Develop step-by-step examples
3. Add explanations and context
4. Ensure the notebook runs correctly with nbmake

### Phase 4C: Reference Documentation
1. Create the API reference documentation
2. Add mathematical background
3. Include installation instructions
4. Update cross-references

### Phase 4D: Final Integration
1. Update dependency definitions
2. Update top-level exports
3. Update README and other documentation
4. Verify all components work together

## Success Criteria

Phase 4 will be considered complete when:

1. All public API components have comprehensive docstrings
2. Examples run successfully with xdoctest
3. The tutorial notebook runs successfully with nbmake
4. The reference documentation is complete
5. Integration with Soundscapy is seamless
6. All documentation is accessible and clear

## Future Work

After completing Phase 4, potential future enhancements include:

1. **Plotting Integration**: Add visualization capabilities
2. **Optimization Features**: Implement target derivation 
3. **Pure Python Implementation**: Replace R dependency with Python
4. **Performance Optimization**: Improve performance for large datasets