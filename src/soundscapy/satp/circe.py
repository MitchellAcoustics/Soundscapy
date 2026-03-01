"""
Circumplex SEM Analysis for Soundscape Attributes Translation Project (SATP).

This module provides tools for analyzing soundscape perception data using circumplex
Structural Equation Modeling (SEM). It includes data validation schemas, model fitting
classes, and analysis workflows for the Soundscape Attributes Translation Project.

The module supports various circumplex model types (unconstrained, equal angles,
equal communalities, and full circumplex) and provides automated data preprocessing
including ipsatization (participant-wise centering).

Functions
---------
ipsatize : function
    Participant-wise centering of PAQ ratings
fit_circe : function
    Fit circumplex SEM models and return a tidy DataFrame

Classes
-------
CircModelE : Enum
    Enumeration of available circumplex model types
SATPSchema : DataFrameModel
    Pandera schema for validating SATP data format
CircE : dataclass
    Results container for a fitted circumplex model
"""

import dataclasses
import warnings
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera import Field
from pandera.typing.pandas import DataFrame, Series

import soundscapy.r_wrapper as sspyr
from soundscapy import PAQ_IDS, PAQ_LABELS, get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Column alias lookup — built once at module load.
# Maps any lowercase column name to its canonical schema name.
# Covers: PAQ label names (e.g. "pleasant" → "PAQ1"), PAQ IDs in any case
# (e.g. "paq1" → "PAQ1"), and the participant field ("PARTICIPANT" → "participant").
# ---------------------------------------------------------------------------
_COLUMN_ALIASES: dict[str, str] = {
    **{
        label.lower(): paq_id
        for label, paq_id in zip(PAQ_LABELS, PAQ_IDS, strict=False)
    },
    **{paq_id.lower(): paq_id for paq_id in PAQ_IDS},
    "participant": "participant",
}


class CircModelE(str, Enum):
    """Enumeration of circumplex model types."""

    UNCONSTRAINED = "unconstrained"
    EQUAL_ANG = "equal_ang"
    EQUAL_COM = "equal_com"
    CIRCUMPLEX = "circumplex"

    @property
    def equal_ang(self) -> bool:
        """
        Whether this model constrains all angles to be equally spaced.

        True for EQUAL_ANG and CIRCUMPLEX; False for UNCONSTRAINED and EQUAL_COM.
        """
        return self in {CircModelE.EQUAL_ANG, CircModelE.CIRCUMPLEX}

    @property
    def equal_com(self) -> bool:
        """
        Whether this model constrains all communalities to be equal.

        True for EQUAL_COM and CIRCUMPLEX; False for UNCONSTRAINED and EQUAL_ANG.
        """
        return self in {CircModelE.EQUAL_COM, CircModelE.CIRCUMPLEX}


class SATPSchema(pa.DataFrameModel):
    """
    Pandera schema for validating SATP (Soundscape Attributes Translation Project) data.

    This schema validates DataFrame columns containing PAQ ratings
    and participant identifiers. PAQ ratings must be between 0 and 100.
    """

    PAQ1: Series[float] = Field(ge=0, le=100)
    PAQ2: Series[float] = Field(ge=0, le=100)
    PAQ3: Series[float] = Field(ge=0, le=100)
    PAQ4: Series[float] = Field(ge=0, le=100)
    PAQ5: Series[float] = Field(ge=0, le=100)
    PAQ6: Series[float] = Field(ge=0, le=100)
    PAQ7: Series[float] = Field(ge=0, le=100)
    PAQ8: Series[float] = Field(ge=0, le=100)

    participant: Series[str] | None = Field(nullable=True)

    class Config:
        """Configuration for the schema validation behavior."""

        drop_invalid_rows = True
        strict = "filter"

    @pa.dataframe_parser
    def column_alias(cls, df: DataFrame) -> DataFrame:  # noqa: N805
        """
        Parse and rename DataFrame columns to match the schema.

        Uses ``_COLUMN_ALIASES`` (module-level constant) for a single-pass
        case-insensitive lookup.  Handles PAQ label names (e.g. ``"pleasant"``
        or ``"Pleasant"`` → ``"PAQ1"``), PAQ IDs in any case (``"paq1"`` →
        ``"PAQ1"``), and the participant field in any capitalisation
        (``"PARTICIPANT"`` → ``"participant"``).

        Parameters
        ----------
        df
            Input DataFrame to rename columns for

        Returns
        -------
        DataFrame with renamed columns matching the schema

        """
        rename_dict = {
            col: _COLUMN_ALIASES[col.lower()]
            for col in df.columns
            if col.lower() in _COLUMN_ALIASES
        }
        return df.rename(columns=rename_dict)


# Ideal 45°-spaced circumplex positions (degrees), used for GDIFF calculation.
# Two orderings handle datasets where PAQ1 is anchored near 0° vs near 315°.
_IDEAL_ANGLES = np.array([0, 45, 90, 135, 180, 225, 270, 315])
_IDEAL_ANGLES_REV = np.array([0, 315, 270, 225, 180, 135, 90, 45])

# Threshold for the sum of the first three angles (PAQ1-PAQ3).
# If sum > 300, it indicates the angles are likely in the reversed orientation
# (e.g., 0 + 315 + 270 = 585) rather than standard (0 + 45 + 90 = 135).
_ANGLE_REV_THRESHOLD = 300


@dataclasses.dataclass
class CircE:
    """
    Results container for a fitted CircE (circumplex SEM) model.

    Attributes
    ----------
    model
        The circumplex model type that was fitted.
    datasource
        Source identifier for the dataset.
    language
        Language code for the dataset.
    n
        Number of observations (complete cases) used to fit the model.
    m
        Number of common factors.
    chisq
        Chi-squared fit statistic.
    d
        Model degrees of freedom.
    p
        p-value for the chi-squared statistic.
    cfi
        Comparative Fit Index.
    gfi
        Goodness of Fit Index.
    agfi
        Adjusted Goodness of Fit Index.
    srmr
        Standardised Root Mean Square Residual.
    mcsc
        Mean Communality Squared Cosines.
    rmsea
        Root Mean Square Error of Approximation.
    rmsea_l
        Lower bound of the 90% confidence interval for RMSEA.
    rmsea_u
        Upper bound of the 90% confidence interval for RMSEA.
    polar_angles
        Estimated polar angles (degrees) for each PAQ item, with PAQ_IDS as
        the index. Only available for models with free angle parameters
        (UNCONSTRAINED, EQUAL_COM). ``None`` for EQUAL_ANG and CIRCUMPLEX.

    """

    model: CircModelE
    datasource: str
    language: str
    n: int
    m: int | None
    chisq: float | None
    d: int | None
    p: float | None
    cfi: float | None
    gfi: float | None
    agfi: float | None
    srmr: float | None
    mcsc: float | None
    rmsea: float | None
    rmsea_l: float | None
    rmsea_u: float | None
    polar_angles: pd.Series | None = None  # PAQ_IDS index, angle estimates

    @classmethod
    def from_bfgs(
        cls,
        bfgs_model: Any,
        datasource: str,
        language: str,
        circ_model: CircModelE,
        n: int,
    ) -> "CircE":
        """Create a CircE instance from a fitted BFGS model."""
        fit_stats = sspyr.extract_bfgs_fit(bfgs_model)
        polar_angles = None
        # Only extract polar angles for models where angles are free parameters.
        # The R key is "polar.angles" (dot), not "polar_angles" (underscore).
        if circ_model in (CircModelE.UNCONSTRAINED, CircModelE.EQUAL_COM):
            raw_pa = fit_stats.get("polar.angles")
            if raw_pa is not None:
                # raw_pa is a DataFrame with index=PAQ_IDS and columns from
                # CircE_BFGS: ["estimates", "(L;", "U)"].  Use the label
                # "estimates" directly; fall back to first column if the
                # CircE API ever changes its output names.
                pa_df = pd.DataFrame(raw_pa)
                if "estimates" in pa_df.columns:
                    estimates = pa_df["estimates"].to_numpy()
                else:
                    estimates = pa_df.iloc[:, 0].to_numpy()
                polar_angles = pd.Series(estimates, index=PAQ_IDS)

        return cls(
            model=circ_model,
            datasource=datasource,
            language=language,
            n=n,
            m=fit_stats.get("m", None),
            chisq=fit_stats.get("chisq", None),
            d=fit_stats.get("d", None),
            p=fit_stats.get("p", None),
            cfi=fit_stats.get("cfi", None),
            gfi=fit_stats.get("gfi", None),
            agfi=fit_stats.get("agfi", None),
            srmr=fit_stats.get("srmr", None),
            mcsc=fit_stats.get("mcsc", None),
            rmsea=fit_stats.get("rmsea", None),
            rmsea_l=fit_stats.get("rmsea.l", None),
            rmsea_u=fit_stats.get("rmsea.u", None),
            polar_angles=polar_angles,
        )

    @classmethod
    def compute_bfgs_fit(
        cls,
        data_cor: pd.DataFrame,
        n: int,
        datasource: str,
        language: str,
        circ_model: CircModelE,
    ) -> "CircE":
        """
        Compute and return a CircE from the given correlation matrix.

        Parameters
        ----------
        data_cor
            Correlation matrix of the PAQ data (8x8).
        n
            Number of observations used to compute ``data_cor``.
            This is used by ``CircE_BFGS`` for chi-square and RMSEA calculations
            and must be the row count of the *complete-case* data.
        datasource
            Source identifier for the dataset.
        language
            Language code for the dataset.
        circ_model
            Circumplex model type to fit.

        Examples
        --------
        >>> import soundscapy as sspy
        >>> data = sspy.isd.load()
        >>> data_paqs = data[PAQ_IDS]
        >>> data_paqs = data_paqs.dropna()
        >>> data_cor = data_paqs.corr()
        >>> n = len(data_paqs)
        >>> circ_model = sspy.satp.CircModelE.CIRCUMPLEX
        >>> circe_res = sspy.satp.CircE.compute_bfgs_fit(
        ... data_cor, n, "ISD", "EN", circ_model)
        ...

        """
        bfgs_model = sspyr.bfgs(
            data_cor=data_cor,
            n=n,
            scales=PAQ_IDS,
            m_val=3,
            equal_ang=circ_model.equal_ang,
            equal_com=circ_model.equal_com,
        )
        return cls.from_bfgs(bfgs_model, datasource, language, circ_model, n)

    @property
    def gdiff(self) -> float | None:
        """
        RMSD between fitted polar angles and ideal circumplex spacing.

        Measures how closely the unconstrained angle estimates match perfect
        45°-spaced circumplex positions.  Only defined for models with free
        angles (UNCONSTRAINED, EQUAL_COM); returns ``None`` for EQUAL_ANG and
        CIRCUMPLEX (where ``polar_angles`` is ``None``).

        A smaller value indicates better agreement with circumplex structure.

        Returns
        -------
        float or None
            Rounded RMSD value (2 decimal places), or ``None`` if angles are
            fixed by the model.

        """
        if self.polar_angles is None:
            return None
        obs = self.polar_angles.to_numpy()
        # Choose reference based on whether PAQ1 is anchored near 315° or 0°.
        reference = (
            _IDEAL_ANGLES_REV if obs[:3].sum() > _ANGLE_REV_THRESHOLD else _IDEAL_ANGLES
        )
        return round(float(np.sqrt(np.mean((obs - reference) ** 2))), 2)

    def to_dict(self) -> dict[str, Any]:
        """
        Return all model fit statistics as a flat dictionary.

        Polar angle columns (PAQ1-PAQ8) are expanded as individual keys.
        For models with fixed angles (EQUAL_ANG, CIRCUMPLEX), PAQ values
        are ``None``.

        Returns
        -------
        dict
            Flat dictionary suitable for constructing a pandas DataFrame row.

        """
        base = {
            "datasource": self.datasource,
            "language": self.language,
            "model": self.model.value,
            "n": self.n,
            "m": self.m,
            "chisq": self.chisq,
            "d": self.d,
            "p": self.p,
            "cfi": self.cfi,
            "gfi": self.gfi,
            "agfi": self.agfi,
            "srmr": self.srmr,
            "mcsc": self.mcsc,
            "rmsea": self.rmsea,
            "rmsea_l": self.rmsea_l,
            "rmsea_u": self.rmsea_u,
            "gdiff": self.gdiff,
        }
        if self.polar_angles is not None:
            base.update(self.polar_angles.to_dict())
        else:
            base.update({paq: None for paq in PAQ_IDS})
        return base


def ipsatize(data: pd.DataFrame, by: str = "participant") -> pd.DataFrame:
    """
    Ipsatize (participant-wise center) PAQ ratings.

    Each participant's responses are centered around their own mean for each
    PAQ column. This removes individual response style differences (e.g., a
    tendency to rate everything high or low) while preserving relative patterns.

    Parameters
    ----------
    data
        DataFrame containing PAQ columns and a grouping column.
    by
        Column name to group by for centering. Default is ``"participant"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with participant-centered PAQ values.
        The ``by`` column is consumed by the groupby and dropped from the result.

    """
    return data.groupby(by).transform(lambda x: x - x.mean())


def fit_circe(
    data: pd.DataFrame,
    language: str,
    datasource: str,
    *,
    models: list[CircModelE] | None = None,
    ipsatize_data: bool = True,
) -> pd.DataFrame:
    """
    Fit circumplex SEM models to PAQ data and return a tidy DataFrame.

    Validates input data, optionally ipsatizes responses, computes a
    complete-case correlation matrix, and fits the requested circumplex
    model types using Browne's BFGS optimisation via the R ``CircE`` package.

    Parameters
    ----------
    data
        DataFrame with PAQ1-PAQ8 and a ``participant`` column.
        Column aliases (e.g. PAQ label names, ``Participant``) are accepted
        and renamed automatically by the schema validator.
    language
        Language code for the dataset (e.g. ``"eng"``, ``"fra"``).
        Stored in the results; not used for computation.
    datasource
        Dataset identifier (e.g. ``"SATP"``, ``"ISD"``).
        Stored in the results; not used for computation.
    models
        List of model types to fit. Default: all four ``CircModelE`` variants.
    ipsatize_data
        Whether to apply participant-wise centering before fitting.
        Set to ``False`` if the data is already ipsatized.

    Returns
    -------
    pd.DataFrame
        One row per fitted model. Columns: ``datasource``, ``language``,
        ``model``, ``n``, ``m``, ``chisq``, ``d``, ``p``, ``cfi``, ``gfi``,
        ``agfi``, ``srmr``, ``mcsc``, ``rmsea``, ``rmsea_l``, ``rmsea_u``,
        ``gdiff``, ``PAQ1``-``PAQ8``.
        ``PAQ1``-``PAQ8`` contain fitted polar angle estimates for free-angle
        models (UNCONSTRAINED, EQUAL_COM); ``None`` for constrained models.
        Rows for models that fail to converge contain an ``error`` column.

    Examples
    --------
    >>> import soundscapy as sspy
    >>> from soundscapy.satp import fit_circe
    >>> data = sspy.isd.load()
    >>> data = data.rename(columns={'SessionID': 'participant'})
    >>> results = fit_circe(data, language='eng', datasource='ISD')
    >>> results.shape[0]
    4

    """
    warnings.warn(
        "The SATP analysis module is experimental. Use with caution.",
        UserWarning,
        stacklevel=2,
    )
    if len(data) == 0:
        msg = (
            "No complete cases found: input DataFrame is empty. "
            "Check that data contains valid rows with PAQ1-PAQ8 and participant column."
        )
        raise ValueError(msg)
    validated = SATPSchema.validate(data, lazy=True)
    if ipsatize_data and "participant" not in validated.columns:
        msg = (
            "ipsatize_data=True requires a 'participant' column. "
            "Pass ipsatize_data=False if your data is already ipsatized."
        )
        raise ValueError(msg)
    processed = ipsatize(validated) if ipsatize_data else validated

    # Use listwise deletion (complete cases only) — consistent with R's na.omit().
    complete = processed[PAQ_IDS].dropna()
    n = len(complete)
    if n == 0:
        msg = (
            "No complete cases found after validation and ipsatization. "
            "Check that PAQ1-PAQ8 are not all NaN and participant column is present."
        )
        raise ValueError(msg)
    corr = complete.corr()

    circ_models = models if models is not None else list(CircModelE)
    rows: list[dict] = []
    fit_exceptions = (ValueError, np.linalg.LinAlgError, RuntimeError)
    for model in circ_models:
        try:
            circe = CircE.compute_bfgs_fit(corr, n, datasource, language, model)
            rows.append(circe.to_dict())
        except fit_exceptions as e:  # noqa: PERF203
            warnings.warn(f"{model.value} raised {e}", stacklevel=2)
            # Populate all expected columns with None so that pandas does not
            # promote numeric columns (e.g. n, d) to float64 across all rows
            # when mixing sparse error dicts with full success dicts.
            rows.append(
                {
                    "language": language,
                    "datasource": datasource,
                    "model": model.value,
                    "n": n,
                    "m": None,
                    "chisq": None,
                    "d": None,
                    "p": None,
                    "cfi": None,
                    "gfi": None,
                    "agfi": None,
                    "srmr": None,
                    "mcsc": None,
                    "rmsea": None,
                    "rmsea_l": None,
                    "rmsea_u": None,
                    "gdiff": None,
                    **{paq: None for paq in PAQ_IDS},
                    "error": str(e),
                }
            )

    return pd.DataFrame(rows)
