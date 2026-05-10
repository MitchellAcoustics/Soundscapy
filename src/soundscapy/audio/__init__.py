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

# ruff: noqa: E402
from soundscapy._optional import require_deps

require_deps(["acoustic_toolbox", "maad", "mosqito", "tqdm"], extra="audio")

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
