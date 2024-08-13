"""
soundscapy.audio
================

This module provides tools for working with audio signals, particularly binaural recordings.

Key Components:
- Binaural: A class for processing and analyzing binaural audio signals.
- Various metric calculation functions for audio analysis.

The module integrates with external libraries such as mosqito, maad, and python-acoustics
to provide a comprehensive suite of audio analysis tools.

Example:
    >>> from soundscapy.audio import Binaural
    >>> signal = Binaural.from_wav("audio.wav")
    >>> results = signal.process_all_metrics(analysis_settings)

See Also:
    soundscapy.audio.binaural: For detailed Binaural class documentation.
    soundscapy.audio.metrics: For individual metric calculation functions.
"""

from soundscapy.audio.analysis_settings import AnalysisSettings, get_default_yaml
from soundscapy.audio.binaural import Binaural

__all__ = ["Binaural", "AnalysisSettings", "get_default_yaml"]
