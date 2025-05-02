"""
Provides tools for working with audio signals, particularly binaural recordings.

Key Components:
- Binaural: A class for processing and analyzing binaural audio signals.
- Various metric calculation functions for audio analysis.

The module integrates with external libraries such as mosqito, maad,
and acoustic_toolbox to provide a comprehensive suite of audio analysis tools.

Example:
    >>> # xdoctest: +SKIP
    >>> from soundscapy.audio import Binaural
    >>> signal = Binaural.from_wav("audio.wav")
    >>> results = signal.process_all_metrics(analysis_settings)

See Also:
    soundscapy.audio.binaural: For detailed Binaural class documentation.
    soundscapy.audio.metrics: For individual metric calculation functions.

"""

# ruff: noqa: E402
# ignore module level import order because we need to check dependencies first

# Check for required dependencies directly
# This will raise ImportError if any dependency is missing
try:
    import acoustic_toolbox  # noqa: F401
    import maad  # noqa: F401
    import mosqito  # noqa: F401
    import tqdm  # noqa: F401
except ImportError as e:
    msg = (
        "Audio analysis functionality requires additional dependencies. "
        "Install with: pip install soundscapy[audio]"
    )
    raise ImportError(msg) from e

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
