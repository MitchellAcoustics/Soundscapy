# Contributing to Soundscapy

## Adding Optional Dependencies

Soundscapy uses a modular system for optional features. Follow these steps to add new optional dependency groups:

Here's a guide for adding new optional dependencies to Soundscapy:

### 1. Define the Dependency Group

Add a new group to `MODULE_GROUPS` in `_optionals.py`:

```python
MODULE_GROUPS = {
    "new_group": {
        "modules": ("package1", "package2"),  # Required module names
        "install": "soundscapy[new_group]",   # pip install target
        "description": "description of functionality", 
    },
    # ... existing groups
}
```

### 2. Add to `pyproject.toml` and install dependencies

Add the new optional dependencies group to the `pyproject.toml` file using `uv`:

```bash
uv add package1 package2 --optional new_group
```

```toml
[project.optional-dependencies]
new_group = [
    "package1>=1.0.0",
    "package2>=2.0.0"
]
```

### 3. Implement Conditional Imports

In the relevant module's `__init__.py` file, follow this pattern:

```python
from soundscapy._optionals import OptionalDependencyManager

manager = OptionalDependencyManager.get_instance()
if manager.check_module_group("new_group"):
    from .submodule import Feature1, Feature2
    __all__ = ["Feature1", "Feature2"]
else:
    __all__ = []  # Empty if deps not available
```

### 4. Use in Code

When using stand-alone optional features, always check for availability:

```python
def my_function():
    if not OptionalDependencyManager.get_instance().check_module_group("new_group"):
        raise ImportError("This feature requires soundscapy[new_group]")
    # Feature implementation...
```

### Best Practices

- Keep related dependencies grouped together
- Use meaningful group names that describe functionality
- Provide helpful error messages suggesting how to install missing deps
- Document optional features in the module docstrings
- Add appropriate tests for both presence and absence of optional deps

See `audio/__init__.py`

 for a complete example of how optional dependencies are used in the audio module.

## Additional Development Guidelines

- Run tests with `pytest` before submitting PRs
- Follow the existing code style (Ruff formatter)
