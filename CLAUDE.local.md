# Soundscapy Development Guide

## Build & Test Commands

- Run all tests: `uv run pytest`
- Run single test: `uv run pytest test/path/to/test.py::test_function`
- Run tests with specific marker: `uv run pytest -m "optional_deps('audio')"`
- Run with parallel execution: `uv run pytest -xvs`
- Code formatter: `uv run ruff format .`
- Code linting: `uv run ruff check .`
- Build package: `uv build`

## Code Style

- Use python 3.10+ type hints for all function parameters and return values (e.g. `str | None = None` rather than `Optional[str]`)
- Type annotations should follow Python 3.9+ conventions and use the built in datatypes dict, tuple, etc., rather than Dict, Tuple, etc.
- Use docstrings in NumPy format for all public functions and classes
- Variable naming: snake_case for variables/functions, PascalCase for classes
- Prefer explicit error handling with try/except blocks
- Prefer pathlib.Path over string paths
- Use loguru for logging through the soundscapy.logging module
- Use optional dependencies system for features with heavy dependencies

## Scientific Python Design Principles

- Detailed design principles are in `ai_dev_docs/design.md`
- Keep I/O separate from scientific logic
- Take a layered approach to complexity and permissiveness:
  - Separate into two layers: a thin "friendly" layer on top of a "cranky" layer that takes in only exactly what it needs and does the actual work.
  - The cranky layer should be easy to test; it should be constrained about what it accepts and what it returns.
  - This layered design makes it possible to write _many_ friendly layers with different opinions and different defaults.
- Prefer duck typing and protocols over explicit type checks
- Prefer functions over classes when possible
- Avoid changing state; use immutable objects where appropriate
- Use standard scientific types (numpy arrays, pandas DataFrames) over custom classes
- Write specific, helpful error messages
- Prefer small, focused functions over complex multi-purpose ones
- Functions should have stable return types regardless of inputs
- Use keyword-only arguments for optional parameters
- Make complexity explicit rather than hiding it

## SPI Feature Development

- Detailed development plans are in `ai_dev_docs/spi/`
- Using test-driven development (TDD) approach
- The feature relies on R integration through rpy2 as an optional dependency
- Implementation is based on bivariate skew-normal distributions for soundscape perception indices
- Follow layered design with "friendly" and "cranky" layers
- This feature is based on research from J2401_JASA_SSID-Single-Index repository
- Reference the original paper for mathematical background
- Use Soundscapy's optional dependency system for R/rpy2 integration