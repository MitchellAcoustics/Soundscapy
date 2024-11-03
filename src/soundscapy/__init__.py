"""
Soundscapy is a Python library for soundscape analysis and visualisation.
"""

from loguru import logger
import importlib.metadata

__version__ = importlib.metadata.version("soundscapy")

from soundscapy import plotting
from soundscapy.audio import AnalysisSettings, ConfigManager, AudioAnalysis
from soundscapy.audio.binaural import Binaural
from soundscapy.databases import araus, isd, satp
from soundscapy.surveys import processing
from soundscapy.plotting import scatter_plot, density_plot


__all__ = [
    "AudioAnalysis",
    "AnalysisSettings",
    "Binaural",
    "plotting",
    "araus",
    "isd",
    "satp",
    "processing",
    "ConfigManager",
    "scatter_plot",
    "density_plot",
]

# Set up the logger when this module is imported
logger.info("Soundscapy v%s loaded", __version__)
