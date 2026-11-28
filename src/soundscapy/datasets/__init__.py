"""Bundled and downloadable datasets.

``isd`` is the International Soundscape Database (ISD), with a bundled subset of 1552
responses; ``course_demo`` is the renamed ISD extract of 456 responses that the IoA
soundscape course uses.
"""

from __future__ import annotations

from soundscapy.datasets import course_demo, isd

__all__ = ["course_demo", "isd"]
