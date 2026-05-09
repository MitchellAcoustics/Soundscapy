"""
Module for handling the Soundscape Attributes Translation Project (SATP) database.

This module provides functions for loading and processing data from the
Soundscape Attributes Translation Project database. It includes utilities
for data retrieval from Zenodo and basic data loading operations.

Examples
--------
>>> import soundscapy.databases.satp as satp
>>> df = satp.load_zenodo()
>>> isinstance(df, pd.DataFrame)
True
>>> 'Language' in df.columns
True
>>> satp.load_participants()
Traceback (most recent call last):
    ...
ValueError: Participant data is only available for SATP versions up to v1.2.1.
>>> participants = satp.load_participants(version="v1.2")
>>> isinstance(participants, pd.DataFrame)
True
>>> 'Age' in participants.columns
True

"""

from __future__ import annotations

from enum import Enum
from functools import total_ordering

import pandas as pd
from loguru import logger
from packaging.version import Version
from typing_extensions import Self

_ZENODO_BASE = "https://zenodo.org/record/{record}/files/SATP%20Dataset%20{label}.xlsx"


@total_ordering
class SATPVersion(Enum):
    """
    Versioned SATP dataset releases on Zenodo.

    Each member stores the canonical version string and its Zenodo download
    URL. Version strings are normalised on lookup so ``"1.5"``, ``"v1.5"``,
    and ``"V1.5"`` all resolve to the same member.  The string ``"latest"``
    resolves to the first (newest) member.

    Examples
    --------
    >>> SATPVersion("v1.2").url
    'https://zenodo.org/record/7143599/files/SATP%20Dataset%20v1.2.xlsx'
    >>> SATPVersion("1.2") is SATPVersion("V1.2")
    True
    >>> SATPVersion("latest") is SATPVersion.V1_5
    True
    >>> SATPVersion("invalid")
    Traceback (most recent call last):
        ...
    ValueError: 'invalid' is not a valid SATPVersion

    """

    # Declare the extra attribute so type checkers know it exists on members.
    url: str

    # Members are declared newest-first so that ``latest()`` relies on
    # ``next(iter(cls))`` rather than a hardcoded name.
    V1_5 = ("v1.5", _ZENODO_BASE.format(record="18715282", label="v1.5"))
    V1_5RC1 = ("v1.5rc1", _ZENODO_BASE.format(record="18362685", label="v1.5rc1"))
    V1_4 = ("v1.4", _ZENODO_BASE.format(record="10993139", label="v1.4"))
    V1_3_1 = ("v1.3.1", _ZENODO_BASE.format(record="10159673", label="v1.3.1"))
    V1_3 = ("v1.3", _ZENODO_BASE.format(record="10159673", label="v1.3.1"))
    # v1.2.1 was a metadata patch; the data file is identical to v1.2.
    V1_2_1 = ("v1.2.1", _ZENODO_BASE.format(record="7143599", label="v1.2"))
    V1_2 = ("v1.2", _ZENODO_BASE.format(record="7143599", label="v1.2"))

    def __new__(cls, version: str, url: str) -> Self:
        """Create a new member with a canonical version string and download URL."""
        obj = object.__new__(cls)
        obj._value_ = version
        obj.url = url
        return obj

    @classmethod
    def _missing_(cls, value: object) -> SATPVersion | None:
        """Normalise version strings and resolve ``"latest"``."""
        if not isinstance(value, str):
            return None
        if value.lower() == "latest":
            return cls.latest()
        # Strip leading v/V, lowercase, then re-add the canonical "v" prefix.
        normalised = "v" + value.lstrip("vV").lower()
        for member in cls:
            if member.value == normalised:
                return member
        return None

    @classmethod
    def latest(cls) -> SATPVersion:
        """Return the most recent released version (first declared member)."""
        return next(iter(cls))

    def __lt__(self, other: object) -> bool:
        """Return True if this version is older than other."""
        if not isinstance(other, SATPVersion):
            return NotImplemented
        return Version(str(self)) < Version(str(other))

    def __str__(self) -> str:
        """Return the canonical version string."""
        return self.value


def load_zenodo(version: str = "latest") -> pd.DataFrame:
    """
    Load the SATP dataset from Zenodo.

    Parameters
    ----------
    version
        Version of the dataset to load. The default is "latest".

    Returns
    -------
    :
        DataFrame containing the SATP dataset.

    """
    resolved = SATPVersion(version)
    logger.debug(f"Fetching SATP dataset URL for version: {resolved}")
    data = pd.read_excel(resolved.url, engine="openpyxl", sheet_name="Main Merge")
    logger.info(f"Loaded SATP dataset version {resolved} from Zenodo")
    return data


def load_participants(version: str = "latest") -> pd.DataFrame:
    """
    Load the SATP participants dataset from Zenodo.

    Parameters
    ----------
    version
        Version of the dataset to load. The default is "latest".

    Returns
    -------
    :
        DataFrame containing the SATP participants dataset.

    """
    resolved = SATPVersion(version)
    if SATPVersion(version) > SATPVersion.V1_2_1:
        msg = "Participant data is only available for SATP versions up to v1.2.1."
        raise ValueError(msg)
    logger.debug(f"Fetching SATP dataset URL for version: {resolved}")
    data = pd.read_excel(resolved.url, engine="openpyxl", sheet_name="Participants")
    data = data.drop(columns=["Unnamed: 3", "Unnamed: 4"])
    logger.info(f"Loaded SATP participants dataset version {resolved} from Zenodo")
    return data
