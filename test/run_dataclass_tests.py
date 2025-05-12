#!/usr/bin/env python
"""
Script to run tests for the dataclass-based ParamModel implementation.
"""

import sys

import pytest

if __name__ == "__main__":
    # Run the tests for the dataclass-based ParamModel
    result = pytest.main(["-xvs", "test/test_dataclass_param_model.py"])

    # Exit with the same code as pytest
    sys.exit(result)
