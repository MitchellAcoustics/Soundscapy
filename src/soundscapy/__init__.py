"""Soundscape survey analysis and assessment following ISO 12913.

Soundscapy is a scientific package for soundscape engineering: survey analysis,
psychoacoustic analysis of calibrated recordings, and perception modelling. This
module exposes only the installed package version while the rewrite is under way.
"""

import importlib.metadata

__version__ = importlib.metadata.version("soundscapy")
__all__ = ["__version__"]
