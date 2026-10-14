# SPDX-License-Identifier: BSD-3-Clause
# Copyright (C) 2026  Andrew Mitchell

"""
Universal constants shared across the soundscapy package.

This module is the single home for the PAQ definitions, the standard Likert
scales, and the circumplex angle sets used by the survey, database, plotting,
and analysis modules. Import from here rather than redefining them.
"""

from dataclasses import dataclass, field
from enum import Enum

from plot_likert.scales import Scale


class PAQ(Enum):
    """Enumeration of Perceptual Attribute Questions (PAQ) names and IDs."""

    PLEASANT = ("pleasant", "PAQ1")
    VIBRANT = ("vibrant", "PAQ2")
    EVENTFUL = ("eventful", "PAQ3")
    CHAOTIC = ("chaotic", "PAQ4")
    ANNOYING = ("annoying", "PAQ5")
    MONOTONOUS = ("monotonous", "PAQ6")
    UNEVENTFUL = ("uneventful", "PAQ7")
    CALM = ("calm", "PAQ8")

    def __init__(self, label: str, id: str) -> None:  # noqa: A002
        """
        Initialize a PAQ enum member.

        Parameters
        ----------
        label
            The descriptive label for the PAQ (e.g., 'pleasant').
        id
            The standard identifier for the PAQ (e.g., 'PAQ1').

        """
        self.label = label
        self.id = id


PAQ_LABELS = [paq.label for paq in PAQ]
PAQ_IDS = [paq.id for paq in PAQ]


@dataclass
class LikertScale:
    """
    Contains different Likert scale options for survey questions.

    This class provides standardized 5-point Likert scales questions commonly used
    in acoustic and soundscape surveys.

    Attributes
    ----------
    PAQ
        Agreement scale from "Strongly disagree" to "Strongly agree"
    SOURCE
        Source perception scale from "Not at all" to "Dominates completely"
    OVERALL
        Quality assessment scale from "Very bad" to "Very good"
    APPROPRIATE
        Appropriateness scale from "Not at all" to "Perfectly"
    LOUD
        Loudness perception scale from "Not at all" to "Extremely"
    OFTEN
        Frequency scale with first-time option from "Never / This is my first time here" to "Very often"
    VISIT
        Standard frequency scale from "Never" to "Very often"

    """  # noqa: E501

    paq: Scale = field(
        default_factory=lambda: [
            "Strongly disagree",
            "Somewhat disagree",
            "Neutral",
            "Somewhat agree",
            "Strongly agree",
        ]
    )
    source: Scale = field(
        default_factory=lambda: [
            "Not at all",
            "A little",
            "Moderately",
            "A lot",
            "Dominates completely",
        ]
    )
    overall: Scale = field(
        default_factory=lambda: [
            "Very bad",
            "Bad",
            "Neither bad nor good",
            "Good",
            "Very good",
        ]
    )
    appropriate: Scale = field(
        default_factory=lambda: [
            "Not at all",
            "A little",
            "Moderately",
            "A lot",
            "Perfectly",
        ]
    )
    loud: Scale = field(
        default_factory=lambda: [
            "Not at all",
            "A little",
            "Moderately",
            "Very",
            "Extremely",
        ]
    )
    often: Scale = field(
        default_factory=lambda: [
            "Never / This is my first time here",
            "Rarely",
            "Sometimes",
            "Often",
            "Very often",
        ]
    )
    visit: Scale = field(
        default_factory=lambda: [
            "Never",
            "Rarely",
            "Sometimes",
            "Often",
            "Very often",
        ]
    )


LIKERT_SCALES = LikertScale()

EQUAL_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)

# Language-specific angles for PAQs as defined in Aletta et. al. (2024)
LANGUAGE_ANGLES = {
    "eng": (0, 46, 94, 138, 177, 241, 275, 340),
    "arb": (0, 36, 45, 135, 167, 201, 242, 308),
    "cmn": (0, 18, 38, 154, 171, 196, 217, 318),
    "hrv": (0, 84, 93, 160, 173, 243, 273, 354),
    "nld": (0, 43, 111, 125, 174, 257, 307, 341),
    "deu": (0, 64, 97, 132, 182, 254, 282, 336),
    "ell": (0, 72, 86, 133, 161, 233, 267, 328),
    "ind": (0, 53, 104, 123, 139, 202, 284, 308),
    "ita": (0, 57, 104, 143, 170, 274, 285, 336),
    "spa": (0, 41, 103, 147, 174, 238, 279, 332),
    "swe": (0, 66, 87, 146, 175, 249, 275, 335),
    "tur": (0, 55, 97, 106, 157, 254, 289, 313),
}
