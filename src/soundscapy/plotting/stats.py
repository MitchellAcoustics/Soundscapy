"""
Custom Stat components for Soundscape plots using seaborn.objects.

This module contains custom Stat classes that extend the functionality of
seaborn.objects.Plot for creating soundscape plots. These stats handle
specialized data transformations like converting PAQ data to ISO coordinates.
"""

import numpy as np
import seaborn.objects as so

from soundscapy.surveys.survey_utils import EQUAL_ANGLES


class SoundscapeCoordinates(so.Stat):
    """
    Stat for calculating ISO coordinates from PAQ data.

    This stat transforms PAQ data (perceptual attribute questions) into
    ISOPleasant and ISOEventful coordinates based on the ISO 12913-3 standard.
    """

    def __init__(
        self,
        paq_cols=("PAQ1", "PAQ2", "PAQ3", "PAQ4", "PAQ5", "PAQ6", "PAQ7", "PAQ8"),
        angles=EQUAL_ANGLES,
        val_range=(5, 1),
        output_cols=("ISOPleasant", "ISOEventful"),
        **kwargs,
    ):
        """
        Initialize the ISO coordinates stat.

        Parameters
        ----------
        paq_cols : tuple, default=("PAQ1", "PAQ2", "PAQ3", "PAQ4", "PAQ5", "PAQ6", "PAQ7", "PAQ8")
            Column names for PAQ data
        angles : tuple, default=EQUAL_ANGLES
            Angles for each PAQ in degrees
        val_range : tuple, default=(5, 1)
            (max, min) range of original PAQ responses
        output_cols : tuple, default=("ISOPleasant", "ISOEventful")
            Column names for output coordinates
        **kwargs :
            Additional keyword arguments passed to parent class

        """
        self.paq_cols = paq_cols
        self.angles = angles
        self.val_range = val_range
        self.output_cols = output_cols
        super().__init__(**kwargs)

    def _apply(self, data, **kwargs):
        """
        Calculate ISO coordinates from PAQ data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with PAQ columns
        **kwargs :
            Additional keyword arguments

        Returns
        -------
        pd.DataFrame
            Data with added ISO coordinate columns

        """
        # Check if required columns exist
        if not all(col in data.columns for col in self.paq_cols):
            # Return original data if PAQ columns aren't present
            return data

        # Get only PAQ columns
        paq_df = data[list(self.paq_cols)]

        # Calculate scale factor
        scale = max(self.val_range) - min(self.val_range)

        # Calculate coordinates
        iso_pleasant = paq_df.apply(
            lambda row: self._adj_iso_pl(row, self.angles, scale), axis=1
        )
        iso_eventful = paq_df.apply(
            lambda row: self._adj_iso_ev(row, self.angles, scale), axis=1
        )

        # Add calculated coordinates to data
        transformed = data.assign(
            **{self.output_cols[0]: iso_pleasant, self.output_cols[1]: iso_eventful}
        )

        return transformed

    def _adj_iso_pl(self, values, angles, scale):
        """
        Calculate adjusted ISOPleasant value.

        Parameters
        ----------
        values : pd.Series
            PAQ values for a single sample
        angles : tuple
            Angles for each PAQ in degrees
        scale : float
            Scale factor for normalization

        Returns
        -------
        float
            Adjusted ISOPleasant value

        """
        iso_pl = sum(
            np.cos(np.deg2rad(angle)) * value
            for angle, value in zip(angles, values, strict=False)
        )
        return iso_pl / (
            scale / 2 * sum(abs(np.cos(np.deg2rad(angle))) for angle in angles)
        )

    def _adj_iso_ev(self, values, angles, scale):
        """
        Calculate adjusted ISOEventful value.

        Parameters
        ----------
        values : pd.Series
            PAQ values for a single sample
        angles : tuple
            Angles for each PAQ in degrees
        scale : float
            Scale factor for normalization

        Returns
        -------
        float
            Adjusted ISOEventful value

        """
        iso_ev = sum(
            np.sin(np.deg2rad(angle)) * value
            for angle, value in zip(angles, values, strict=False)
        )
        return iso_ev / (
            scale / 2 * sum(abs(np.sin(np.deg2rad(angle))) for angle in angles)
        )
