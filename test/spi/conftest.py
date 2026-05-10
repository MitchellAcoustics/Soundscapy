"""Skip the entire test/spi/ tree if R extras aren't installed."""

import pytest

pytest.importorskip("rpy2")
