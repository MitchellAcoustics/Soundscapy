"""Soundscape survey analysis and assessment following ISO 12913.

Soundscapy is a scientific package for soundscape engineering: survey analysis,
psychoacoustic analysis of calibrated recordings, and perception modelling. Import it as
``import soundscapy as sspy``; the names here cover reading a survey, projecting it onto
the circumplex, attaching the results to your own frame and plotting them
(``sspy.plot``). matplotlib is imported only when a plot function runs.
"""

import importlib.metadata

from soundscapy import datasets, errors, plot, workspace
from soundscapy._log import set_log_level
from soundscapy.datasets.isd import load as load_isd
from soundscapy.distribution.coordinates import Coordinates
from soundscapy.instrument import Circumplex, Scale, circumplex
from soundscapy.instrument.angles import AngleSelection
from soundscapy.projection.project import ProjectionSettings, project
from soundscapy.provenance import show_versions
from soundscapy.survey.mapping import Coding, SurveyMapping, read_mapping, write_mapping
from soundscapy.survey.read import ReadOptions, Survey, read_survey
from soundscapy.workspace import attach

__version__ = importlib.metadata.version("soundscapy")
__all__ = [
    "AngleSelection",
    "Circumplex",
    "Coding",
    "Coordinates",
    "ProjectionSettings",
    "ReadOptions",
    "Scale",
    "Survey",
    "SurveyMapping",
    "__version__",
    "attach",
    "circumplex",
    "datasets",
    "errors",
    "load_isd",
    "plot",
    "project",
    "read_mapping",
    "read_survey",
    "set_log_level",
    "show_versions",
    "workspace",
    "write_mapping",
]
