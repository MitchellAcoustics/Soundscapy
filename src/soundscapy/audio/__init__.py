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
    >>> from soundscapy.audio import BinauralSignal  # doctest: +SKIP
    >>> signal = Binaural.from_wav("audio.wav")  # doctest: +SKIP
    >>> results = signal.process_all_metrics(analysis_settings)  # doctest: +SKIP

See Also:
    soundscapy.audio.binaural: For detailed Binaural class documentation.
    soundscapy.audio.metrics: For individual metric calculation functions.
"""

from soundscapy.audio.analysis_settings import AnalysisSettings, get_default_yaml
from soundscapy.audio.binaural_signal import BinauralSignal
from soundscapy.audio.metric_registry import MetricRegistry
from soundscapy.audio.mosqito_metrics import LoudnessZWTV, LoudnessZWTVResult

# Global instance
metric_registry = MetricRegistry()

# Register the Zwicker Time Varying Loudness metric
metric_registry.register("loudness_zwtv", LoudnessZWTV, LoudnessZWTVResult)

__all__ = ["BinauralSignal", "AnalysisSettings", "get_default_yaml", "metric_registry"]
