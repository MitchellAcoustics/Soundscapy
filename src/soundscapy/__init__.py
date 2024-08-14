"""
Soundscapy is a Python library for soundscape analysis and visualisation.
"""

import importlib.metadata

__version__ = importlib.metadata.version("soundscapy")

from soundscapy import plotting
from soundscapy.audio import AnalysisSettings, get_default_yaml
from soundscapy.audio.binaural import Binaural
from soundscapy.databases import araus, isd, satp
from soundscapy.surveys import processing
from soundscapy.logging import get_logger

# Initialize the logger
logger = get_logger()

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

# Set up the logger when this module is imported
logger.info("Soundscapy v%s loaded", __version__)
