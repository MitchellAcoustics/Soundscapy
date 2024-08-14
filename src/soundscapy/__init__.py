"""
Soundscapy is a Python library for soundscape analysis and visualisation.
"""

import importlib.metadata

from soundscapy import plotting
from soundscapy.audio import AnalysisSettings, get_default_yaml
from soundscapy.audio.binaural import Binaural
from soundscapy.databases import araus, isd, satp
from soundscapy.surveys import processing

__version__ = importlib.metadata.version("soundscapy")

__all__ = [
    "AnalysisSettings",
    "Binaural",
    "plotting",
    "araus",
    "isd",
    "satp",
    "processing",
    "get_default_yaml",
]
