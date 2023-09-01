"""
Soundscapy is a Python library for soundscape analysis and visualisation.
"""

import importlib.metadata

from soundscapy.analysis._AnalysisSettings import AnalysisSettings, get_default_yaml
from soundscapy.analysis._Binaural import Binaural
from soundscapy.utils import _sspy_accessor, surveys
from soundscapy.databases import isd, araus, satp
from soundscapy import plotting

__version__ = importlib.metadata.version("soundscapy")
