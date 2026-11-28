"""Soundscape survey analysis and assessment following ISO 12913.

Soundscapy is a scientific package for soundscape engineering: survey analysis,
psychoacoustic analysis of calibrated recordings, and perception modelling. Import it as
``import soundscapy as sspy``; the names here cover reading a survey, projecting it onto
the circumplex, attaching the results to your own frame and plotting them
(``sspy.plot``); ``QualitySettings`` holds the response quality rules, and
``list_profiles`` and ``save_settings`` the analysis profiles and settings files.
matplotlib is imported only when a plot function runs.
"""

from __future__ import annotations

import importlib.metadata

from soundscapy import datasets, errors, plot, workspace
from soundscapy._log import set_log_level
from soundscapy.datasets.isd import load as load_isd
from soundscapy.distribution.coordinates import Coordinates
from soundscapy.instrument import Circumplex, Scale, circumplex
from soundscapy.instrument.angles import AngleSelection, AngleSet, AngleSetRef
from soundscapy.profiles import list_profiles, save_settings
from soundscapy.projection.project import ProjectionSettings, project
from soundscapy.provenance import show_versions
from soundscapy.survey.mapping import Coding, SurveyMapping, read_mapping, write_mapping
from soundscapy.survey.quality import QualitySettings
from soundscapy.survey.read import ReadOptions, Survey, read_survey
from soundscapy.workspace import attach

__version__ = importlib.metadata.version("soundscapy")
__all__ = [
    "AngleSelection",
    "AngleSet",
    "AngleSetRef",
    "Circumplex",
    "Coding",
    "Coordinates",
    "ProjectionSettings",
    "QualitySettings",
    "ReadOptions",
    "Scale",
    "Survey",
    "SurveyMapping",
    "__version__",
    "attach",
    "circumplex",
    "datasets",
    "errors",
    "list_profiles",
    "load_isd",
    "plot",
    "project",
    "read_mapping",
    "read_survey",
    "save_settings",
    "set_log_level",
    "show_versions",
    "workspace",
    "write_mapping",
]
