# Changelog

All notable changes to the Soundscapy project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 0.8.0

### Added

- New **Soundscape Perception Indices (SPI)** module
  - Provides tools for calculating Soundscape Perception Indices
  - Implements Multi-dimensional Skewed Normal (MSN) distribution for soundscape analysis
  - Includes R wrapper for core statistical functionality
  - Adds SPI score calculation and visualization

- Completely redesigned **ISOPlot API** with layered plotting approach
  - New flexible and extensible plotting interface
  - Layered architecture for combining different plot types
  - Enhanced subplot support for comparing multiple conditions
  - Improved parameter handling for consistent styling
  - Added SPI visualization integration

### Changed

- Replaced CircumplexPlot with the new ISOPlot interface
- Removed Pydantic ParamModels in favor of dataclass-based parameter handling
- Improved logging system with new sspylogging module
- Enhanced plotting documentation, testing, and configuration
- Refactored subplot creation with improved code organization
- Removed Plotly dependency to simplify installation
- Updated notebook tutorials to demonstrate new interfaces

### Developer Experience

- Added pre-commit hooks and improved CI/CD pipeline
- Enhanced type hints and documentation throughout the codebase
- Updated GitHub issue templates and workflow configurations
- Improved test coverage for core functionality

## [0.7.6] - 2024-11-06

### Changed

- Improved handling of optional dependencies to provide better error messages and IDE support
- Audio components (like `Binaural`) can now be imported directly from the top-level package
  (`from soundscapy import Binaural`) while maintaining helpful error messages when
  dependencies are missing
- Centralized optional dependency configuration in `_optionals.py` for better maintainability

### Developer Notes

- No changes required to existing code using audio components
- The new system provides better IDE completion support while maintaining the same runtime behavior
- Optional components can still be imported from their original location
  (`from soundscapy.audio import Binaural`) or from the top level
  (`from soundscapy import Binaural`)

## [0.7.5]

### Added

#### Support for Optional Dependencies

Soundscapy splits its functionality into optional modules to reduce the number of dependencies required for basic functionality. By default, Soundscapy includes the survey data processing and plotting functionality.

If you would like to use the binaural audio processing and psychoacoustics functionality, you will need to install the optional `audio` dependency:

```bash
pip install soundscapy[audio]
```

To install all optional dependencies, use the following command:

```bash
pip install soundscapy[all]
```

### Developer notes

#### Dev Container Configuration

- Added a new `devcontainer.json` file to configure the development container with specific features and VSCode extensions. (`.devcontainer/devcontainer.json` [.devcontainer/devcontainer.jsonR1-R69](diffhunk://#diff-24ad71c8613ddcf6fd23818cb3bb477a1fb6d83af4550b0bad43099813088686R1-R69))
- Updated `.dockerignore` to exclude the virtual environment directory. (`.devcontainer/.dockerignore` [.devcontainer/.dockerignoreR1](diffhunk://#diff-7691e653179b9ed2292151d962426f76e6f5378e4989e741859bdfcbcef16b97R1))

#### GitHub Workflows

- Removed old CI, release, and test-release workflows. (`.github/workflows/ci.yml` [[1]](diffhunk://#diff-b803fcb7f17ed9235f1e5cb1fcd2f5d3b2838429d4368ae4c57ce4436577f03fL1-L40) `.github/workflows/release.yml` [[2]](diffhunk://#diff-87db21a973eed4fef5f32b267aa60fcee5cbdf03c67fafdc2a9b553bb0b15f34L1-L33) `.github/workflows/test-release.yml` [[3]](diffhunk://#diff-191bb5b4e97db48c9d0bdb945dd00e17b53249422f60a642e9e8d73250b5913aL1-L53)
- Added a new workflow for tagged releases to automate the release process, including building and publishing to PyPI and TestPyPI. (`.github/workflows/tag-release.yml` [.github/workflows/tag-release.ymlR1-R138](diffhunk://#diff-21e1251c1676ed10064d2d98ab1a8f6471a9718058bd316970abe934169f2b60R1-R138))
- Added a new workflow for testing tagged releases, including installation from TestPyPI and running tests. (`.github/workflows/test-tag-release.yml` [.github/workflows/test-tag-release.ymlR1-R114](diffhunk://#diff-11b7dedbf7b09ab5a0bd90aa70d8a2eda1918dab64a511c82104706cfa09f3b7R1-R114))
- Added new workflows for running tests on the main codebase and tutorial notebooks. (`.github/workflows/test.yml` [[1]](diffhunk://#diff-faff1af3d8ff408964a57b2e475f69a6b7c7b71c9978cccc8f471798caac2c88R1-R52) `.github/workflows/test-tutorials.yml` [[2]](diffhunk://#diff-01bd86ab14c3e8d7d1382e5ed2172404eb7d3c46bbffeffe09fc11431885e2a0R1-R42)

## [0.7.3]

### Improved

- Allowed the user to request files to be resampled upon loading. This is necessary for Mosqito metrics, which requires (and will itself resample) the audio files to be 48 kHz. The user can specify the desired sample rate in `Binaural.from_wav()` and higher level functions like `AudioAnalysis.analyse_file`, `AudioAnalysis.analyse_folder`.

## [0.7.0]

Complete refactoring of `Soundscapy`, splitting it into multiple modules (`surveys`, `databases`, `audio`, `plotting`), and improving the overall structure and functionality of the package. Also added more comprehensive documentation and test coverage.

### General Changes

#### Added

- New `soundscapy/surveys/survey_utils.py` for shared utilities
  - Implemented `PAQ` enum for Perceptual Attribute Questions
  - Added `return_paqs` function for filtering PAQ columns
  - Created `rename_paqs` function for standardizing PAQ column names
- Centralized logging configuration in `soundscapy/logging.py`
  - Added support for environment variables to configure logging:
    - `SOUNDSCAPY_LOG_LEVEL` for setting log level
    - `SOUNDSCAPY_LOG_FILE` for specifying a log file
  - Implemented checks for Jupyter notebook environment in logging configuration
  - Added `set_log_level` function to allow dynamic adjustment of log level at runtime
  - Introduced global variable `GLOBAL_LOG_LEVEL` to manage log level across different environments
    - Implemented `setup_logger` function for initializing the logger with custom options
    - Set default logger to WARNING level with console output
    - Created `get_logger` function to retrieve the configured logger
- New processing module `soundscapy/surveys/processing.py` with enhanced functionality
  - Implemented `ISOCoordinates` and `SSMMetrics` dataclasses
  - Added `calculate_iso_coords` function for ISO coordinate calculations
  - Created `add_iso_coords` function to add ISO coordinates to DataFrames
  - Implemented `likert_data_quality` function for data quality checks
  - Added `simulation` function for generating random PAQ responses
  - Created `ssm_metrics` function for Structural Summary Method calculations
- Comprehensive docstrings and doctest examples in `isd.py` and `satp.py`
- New test cases in `test_isd.py` to cover refactored functionality

#### Changed

- Modified default logging level to WARNING for better control over log output
- Refactored `isd.py` to use new processing and survey utility functions
  - Updated `load`, `load_zenodo`, and `validate` functions
  - Refactored selection functions (`select_record_ids`, `select_group_ids`, etc.)
  - Updated `describe_location` and `soundscapy_describe` functions
- Refactored `satp.py` to align with new package structure
  - Updated `load_zenodo` and `load_participants` functions
  - Added doctest examples for all functions
- Modified `__init__.py` to initialize the logger when the package is imported
- Updated import statements across modules to use the new package structure
- Standardized function signatures and return types across all modules
- Changed to Rye as the dependency and environment manager for the project

#### Improved

- Enhanced error handling and input validation in database modules
  - Added type hints to all functions for better code readability and IDE support
  - Implemented more specific exception handling
- Optimized data processing functions for better performance
- Improved code organization and modularity
  - Separated concerns between data loading, processing, and analysis
- Enhanced documentation with more detailed explanations and examples
- Standardized coding style across all modules (using Black formatter)

#### Deprecated

- Removed `remove_lockdown` function in `isd.py` (redundant since the release of ISD v1.0)

#### Removed

- Eliminated redundant code and unused functions across modules

#### Fixed

- Resolved issues with inconsistent PAQ naming conventions
- Fixed bugs in ISO coordinate calculations and SSM metric computations
- Resolved issue where Jupyter notebooks were overriding the default log level

#### Security

- Implemented input validation to prevent potential security vulnerabilities

#### Development

- Implemented a more robust logging system using loguru
  - Added ability to easily change log levels for debugging and development
  - Enabled file logging for persistent log storage
- Enhanced test coverage for core functionality
- Added doctest examples to ensure documentation accuracy and serve as functional tests
- Implemented consistent error messages and logging across the package

#### Documentation

- Added comprehensive docstrings to all functions and classes
- Included usage examples in function docstrings
- Updated README with new package structure and usage instructions
- Created this CHANGELOG to track all significant changes to the project

### Changes to Plotting Module

#### Code Structure

- Split the original circumplex.py into multiple files: backends.py, circumplex_plot.py, plot_functions.py, stylers.py, and plotting_utils.py (implied).
- Introduced abstract base class PlotBackend and concrete implementations SeabornBackend and PlotlyBackend.

#### New Features

- Added support for Plotly backend alongside Seaborn.
- Introduced CircumplexPlot class for creating and managing plots.
- Added StyleOptions dataclass for better style management.
- Implemented simple_density plot type.

#### Improved Customization

- Created CircumplexPlotParams dataclass for better parameter management.
- Added more customization options for plots (e.g., incl_outline, fill, alpha).

#### Enhancements

- Improved type hinting throughout the codebase.
- Added docstrings to classes and functions.
- Implemented PlotType and Backend enums for better type safety.

#### Refactoring

- Moved plotting logic from functions to methods in backend classes.
- Simplified scatter and density functions by leveraging CircumplexPlot class.

#### Removed Features

- Removed jointplot function (marked as TODO in CircumplexPlot class).

#### Constants and Utilities

- Moved constants (e.g., DEFAULT_XLIM, DEFAULT_YLIM) to a separate utilities file.
- Created ExtraParams TypedDict for additional plotting parameters.

## [0.6.2]

### Added

- Changed the name of the `calculate_paq_coords` to `calculate_iso_coords` to better reflect the function's purpose.
- Updated the formula for `calculate_iso_coords` to the more generalised form given in Aletta et. al. (2024).
