"""Skip the entire test/audio/ tree if audio extras aren't installed."""

import pytest

pytest.importorskip("acoustic_toolbox")
pytest.importorskip("maad")
pytest.importorskip("mosqito")
pytest.importorskip("tqdm")
