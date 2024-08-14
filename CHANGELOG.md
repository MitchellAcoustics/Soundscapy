# Changelog

All notable changes to the Soundscapy project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Complete refactoring of `Soundscapy`, splitting it into multiple modules (`surveys`, `databases`, `audio`, `plotting`), and improving the overall structure and functionality of the package. Also added more comprehensive documentation and test coverage.

### Added
- New `soundscapy/surveys/survey_utils.py` for shared utilities
  - Implemented `PAQ` enum for Perceptual Attribute Questions
  - Added `return_paqs` function for filtering PAQ columns
  - Created `rename_paqs` function for standardizing PAQ column names
- Centralized logging configuration in `soundscapy/logging.py`
  - Added support for environment variables to configure logging:
    - `SOUNDSCAPY_LOG_LEVEL` for setting log level
    - `SOUNDSCAPY_LOG_FILE` for specifying a log file
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

### Changed
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

### Improved
- Enhanced error handling and input validation in database modules
  - Added type hints to all functions for better code readability and IDE support
  - Implemented more specific exception handling
- Optimized data processing functions for better performance
- Improved code organization and modularity
  - Separated concerns between data loading, processing, and analysis
- Enhanced documentation with more detailed explanations and examples
- Standardized coding style across all modules (using Black formatter)

### Deprecated
- Removed `remove_lockdown` function in `isd.py` (redundant since the release of ISD v1.0)

### Removed
- Eliminated redundant code and unused functions across modules

### Fixed
- Resolved issues with inconsistent PAQ naming conventions
- Fixed bugs in ISO coordinate calculations and SSM metric computations

### Security
- Implemented input validation to prevent potential security vulnerabilities

### Development
- Implemented a more robust logging system using loguru
  - Added ability to easily change log levels for debugging and development
  - Enabled file logging for persistent log storage
- Enhanced test coverage for core functionality
- Added doctest examples to ensure documentation accuracy and serve as functional tests
- Implemented consistent error messages and logging across the package

### Documentation
- Added comprehensive docstrings to all functions and classes
- Included usage examples in function docstrings
- Updated README with new package structure and usage instructions
- Created this CHANGELOG to track all significant changes to the project

## [0.6.2]

### Added

- Changed the name of the `calculate_paq_coords` to `calculate_iso_coords` to better reflect the function's purpose.
- Updated the formula for `calculate_iso_coords` to the more generalised form given in Aletta et. al. (2024).
