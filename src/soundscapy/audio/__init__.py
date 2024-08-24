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
    >>> # xdoctest: +SKIP
    >>> from soundscapy.audio import Binaural  # doctest: +SKIP
    >>> signal = Binaural.from_wav("audio.wav")  # doctest: +SKIP
    >>> results = signal.process_all_metrics(analysis_settings)  # doctest: +SKIP

See Also:
    soundscapy.audio.binaural: For detailed Binaural class documentation.
    soundscapy.audio.metrics: For individual metric calculation functions.
"""

from soundscapy.audio.analysis_settings import AnalysisSettings, ConfigManager
from soundscapy.audio.audio_analysis import AudioAnalysis
from soundscapy.audio.binaural import Binaural
from soundscapy.audio.metrics import (
    add_results,
    prep_multiindex_df,
    process_all_metrics,
)
from soundscapy.audio.parallel_processing import parallel_process

__all__ = [
    "AudioAnalysis",
    "Binaural",
    "AnalysisSettings",
    "ConfigManager",
    "process_all_metrics",
    "prep_multiindex_df",
    "add_results",
    "parallel_process",
]
