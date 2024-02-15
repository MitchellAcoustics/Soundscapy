"""
Soundscapy is a Python library for soundscape analysis and visualisation.
"""

import importlib.metadata

from soundscapy import plotting
from soundscapy.analysis._AnalysisSettings import AnalysisSettings, get_default_yaml
from soundscapy.analysis._Binaural import Binaural
from soundscapy.databases import araus, isd, satp
from soundscapy.utils import surveys

__version__ = importlib.metadata.version("soundscapy")
