"""
Provides tools for working with audio signals, particularly binaural recordings.

Key Components:

- Binaural: A class for processing and analyzing binaural audio signals.
- Various metric calculation functions for audio analysis.

The module integrates with external libraries such as mosqito, maad,
and acoustic_toolbox to provide a comprehensive suite of audio analysis tools.

Examples
--------
>>> # doctest: +SKIP
>>> from soundscapy.audio import Binaural
>>> signal = Binaural.from_wav("audio.wav")
>>> results = signal.process_all_metrics(analysis_settings)

See Also
--------
- `soundscapy.audio.binaural`: For detailed Binaural class documentation.
- `soundscapy.audio.metrics`: For individual metric calculation functions.

Notes
-----
This module requires the `soundscapy[audio]` optional dependencies.

"""

# ignore module level import order because we need to check dependencies first

# Now we can import our modules that depend on the optional packages
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
    "AnalysisSettings",
    "AudioAnalysis",
    "Binaural",
    "ConfigManager",
    "add_results",
    "parallel_process",
    "prep_multiindex_df",
    "process_all_metrics",
]
