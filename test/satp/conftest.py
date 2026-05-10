"""Skip the entire test/satp/ tree if R extras aren't installed."""

import pytest

pytest.importorskip("rpy2")
