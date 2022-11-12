"""
Soundscapy is a Python library for soundscape analysis and visualisation.
"""

from soundscapy.analysis._AnalysisSettings import AnalysisSettings
from soundscapy.analysis._Binaural import Binaural
from soundscapy.databases import isd, araus
from soundscapy import _sspy_accessor

from soundscapy.plotting.circumplex import scatter, density, jointplot
