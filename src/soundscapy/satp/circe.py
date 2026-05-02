"""
Circumplex SEM Analysis for Soundscape Attributes Translation Project (SATP).

This module provides tools for analyzing soundscape perception data using circumplex
Structural Equation Modeling (SEM). It includes data validation schemas, model fitting
classes, and analysis workflows for the Soundscape Attributes Translation Project.

The module supports various circumplex model types (unconstrained, equal angles,
equal communalities, and full circumplex) and provides automated data preprocessing
including within-person centering (column-wise centering per participant).

Functions
---------
normalize_polar_angles
    Correct reflected polar-angle solutions to canonical orientation
person_center
    Column-wise within-participant centering of PAQ ratings
fit_circe
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
from typing import Any, Literal

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera import Field
from pandera.errors import SchemaErrors
from pandera.typing.pandas import DataFrame, Series
from rpy2.rinterface_lib.embedded import RRuntimeError

import soundscapy.r_wrapper as sspyr
from soundscapy import PAQ_IDS, PAQ_LABELS, get_logger
from soundscapy.surveys.processing import ipsatize

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

    # `| None` makes the column optional (absent is allowed);
    # `nullable=True` permits null *values* within the column when present.
    participant: Series[str] | None = Field(nullable=True)

    class Config:
        """Configuration for the schema validation behavior."""

        drop_invalid_rows = False
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
        :

        """
        rename_dict = {
            col: _COLUMN_ALIASES[col.lower()]
            for col in df.columns
            if col.lower() in _COLUMN_ALIASES
        }
        return df.rename(columns=rename_dict)


# Ideal 45°-spaced circumplex positions (degrees) in canonical counter-clockwise
# order, used for GDIFF calculation after angles have been normalised.
_IDEAL_ANGLES = np.array([0, 45, 90, 135, 180, 225, 270, 315])


def normalize_polar_angles(angles: pd.Series) -> pd.Series:
    """
    Return polar angles in canonical (counter-clockwise) orientation.

    CircE's BFGS optimisation may converge to a mathematically equivalent
    *reflected* solution in which the PAQ attributes are arranged in clockwise
    (decreasing) order rather than the canonical counter-clockwise (increasing)
    order.  Both solutions fit the correlation data equally well, but the
    reflected form is inconsistent with the standard circumplex ordering
    (pleasant → vibrant → eventful → …) and will produce incorrect GDIFF values
    if compared against the ideal equally-spaced angles.

    Detection uses a monotonicity check on the first three angles after PAQ1:
    if the angle for PAQ2 exceeds PAQ3, or PAQ3 exceeds PAQ4, the solution is
    reflected.  This is more robust than a threshold-on-sum heuristic because
    it tests the structural property of the orientation directly.

    When reflection is detected, ``360 - angle`` is applied to PAQ2-PAQ8.
    PAQ1 is anchored at 0° and is left unchanged.

    Parameters
    ----------
    angles
        Series of polar angle estimates (degrees) with PAQ_IDS as the index.
        Typically the ``polar_angles`` attribute of a `CircE` instance.

    Returns
    -------
    :
        Polar angles in canonical (counter-clockwise) orientation, with the
        same index as the input.

    Examples
    --------
    >>> from soundscapy.surveys.survey_utils import PAQ_IDS
    >>> import pandas as pd
    >>> reflected = pd.Series([0.0, 315.0, 270.0, 225.0, 180.0, 135.0, 90.0, 45.0], index=PAQ_IDS)
    >>> normalize_polar_angles(reflected).tolist()
    [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

    """
    # Monotonicity check (positional): canonical ordering has PAQ2 < PAQ3 < PAQ4.
    if angles.iloc[1] > angles.iloc[2] or angles.iloc[2] > angles.iloc[3]:
        corrected = angles.copy()
        corrected.iloc[1:] = 360 - corrected.iloc[1:]
        return corrected
    return angles


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
                polar_angles = normalize_polar_angles(
                    pd.Series(estimates, index=PAQ_IDS)
                )

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
        :
            Rounded RMSD value (2 decimal places), or ``None`` if angles are
            fixed by the model.

        """
        if self.polar_angles is None:
            return None
        obs = self.polar_angles.to_numpy()
        # Angles are always in canonical orientation (normalised in from_bfgs),
        # so we compare directly against the ideal 45°-spaced positions.
        return round(float(np.sqrt(np.mean((obs - _IDEAL_ANGLES) ** 2))), 2)

    def to_dict(self) -> dict[str, Any]:
        """
        Return all model fit statistics as a flat dictionary.

        Polar angle columns (PAQ1-PAQ8) are expanded as individual keys.
        For models with fixed angles (EQUAL_ANG, CIRCUMPLEX), PAQ values
        are ``None``.

        Returns
        -------
        :
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
            base.update(dict.fromkeys(PAQ_IDS))
        return base


@dataclasses.dataclass
class CircEResults:
    """
    Collection of fitted CircE models returned by `fit_circe`.

    Holds both successfully-fitted `CircE` instances and any error rows
    from models that failed to converge.  Access the full tidy DataFrame via
    `table`; access individual model results via `for_model`.

    Attributes
    ----------
    models
        Successfully-fitted `CircE` results, in fitting order.
    language
        Language code passed to `fit_circe`.
    datasource
        Dataset identifier passed to `fit_circe`.
    error_rows
        Dicts for model runs that raised an exception during fitting.
        Each dict contains ``language``, ``datasource``, ``model``, ``n``,
        and an ``error`` key with the exception message.

    """

    models: list[CircE]
    language: str
    datasource: str
    error_rows: list[dict] = dataclasses.field(default_factory=list)

    def __len__(self) -> int:
        """Total number of model runs (successful + failed)."""
        return len(self.models) + len(self.error_rows)

    @property
    def table(self) -> pd.DataFrame:
        """
        Full tidy DataFrame of all model fit statistics.

        One row per model (including error rows).  Columns match those
        described in `fit_circe`.  Integer columns (``n``, ``d``, ``m``)
        use pandas nullable ``Int64`` dtype so that ``None`` in error rows does
        not promote the whole column to ``float64``.
        """
        _order = {m.value: i for i, m in enumerate(CircModelE)}
        rows = [m.to_dict() for m in self.models] + self.error_rows
        result = pd.DataFrame(rows)
        if "model" in result.columns:
            result = (
                result.assign(_ord=result["model"].map(_order))
                .sort_values("_ord")
                .drop("_ord", axis=1)
                .reset_index(drop=True)
            )
        for _int_col in ("n", "d", "m"):
            if _int_col in result.columns:
                result[_int_col] = result[_int_col].astype(pd.Int64Dtype())
        return result

    def for_model(self, model: CircModelE) -> CircE:
        """
        Return the fitted `CircE` result for a specific model type.

        Parameters
        ----------
        model
            The `CircModelE` variant to retrieve.

        Raises
        ------
        KeyError
            If no successful result exists for the requested model (e.g. it
            failed to converge).

        """
        for m in self.models:
            if m.model is model:
                return m
        msg = f"No successful result for model {model.value!r}"
        raise KeyError(msg)

    def _repr_html_(self) -> str:
        return self.table._repr_html_()

    def __repr__(self) -> str:
        return (
            f"CircEResults(language={self.language!r}, "
            f"datasource={self.datasource!r}, "
            f"{len(self.models)} fitted, {len(self.error_rows)} failed)"
        )


def person_center(data: pd.DataFrame, by: str = "participant") -> pd.DataFrame:
    """
    Center PAQ ratings within each participant (column-wise within-person centering).

    !!! warning "Deprecated v0.8.0"
        Use `soundscapy.surveys.ipsatize` with ``method="column_wise"``
        instead.  For the centering that matches the published SATP analysis,
        use ``method="grand_mean"`` (the default of
        `~soundscapy.surveys.ipsatize`).

    This function applies **column-wise** centering: for every PAQ column
    independently, each participant's mean across their observations is
    subtracted (8 centering scalars per participant).

    !!! note
        This is *not* the centering described in the original SATP R
        implementation (Aletta et al., 2024), which applies grand-mean
        centering (one scalar per participant across all PAQ columns and
        observations).  Use `soundscapy.surveys.ipsatize` with
        ``method="grand_mean"`` to match the R reference implementation.

    Parameters
    ----------
    data
        DataFrame containing PAQ columns and a participant grouping column.
    by
        Column to group by for centering. Default is ``"participant"``.

    Returns
    -------
    :
        DataFrame containing only the PAQ columns (not ``by``), with
        column-wise participant-centred values.

    """
    import warnings

    warnings.warn(
        "person_center() is deprecated; use ipsatize(method='column_wise') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ipsatize(data, method="column_wise", participant_col=by)


def fit_circe(
    data: pd.DataFrame,
    language: str,
    datasource: str,
    *,
    models: list[CircModelE] | None = None,
    center_by_participant: bool = True,
    errors: Literal["raise", "warn"] = "raise",
) -> "CircEResults":
    """
    Fit circumplex SEM models to PAQ data and return a tidy DataFrame.

    Validates input data, optionally applies grand-mean within-person centering
    (matching the published SATP analysis), computes a complete-case correlation
    matrix, and fits the requested circumplex model types using Browne's BFGS
    optimisation via the R ``CircE`` package.

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
        Passing ``[]`` returns an empty `CircEResults`
        (``len(result) == 0``).
    center_by_participant
        Whether to apply grand-mean within-person centering (via
        `~soundscapy.surveys.ipsatize` with ``method="grand_mean"``)
        before fitting.  Set to ``False`` if the data is already centered or
        if no centering is desired.
    errors
        How to handle rows that fail schema validation (PAQ values outside
        ``[0, 100]``, missing required columns, etc.):

        ``"raise"`` *(default)* — raise a `pandera.errors.SchemaErrors`
        immediately, listing every failing row and constraint.

        ``"warn"`` — emit a `UserWarning` describing the failing rows
        and continue with the valid rows only.

        !!! note
            If you pass *already-centered* data, set
            ``center_by_participant=False`` to skip the internal centering step;
            otherwise pass raw ``[0, 100]``-range data and use the default
            ``center_by_participant=True``.  Passing pre-centered data without
            disabling centering will cause schema validation to reject the
            negative values.

    Returns
    -------
    :
        Collection of fitted models.  Access the tidy DataFrame via
        ``.table``; access individual model results via ``.for_model()``.
        Failed models are stored in ``.error_rows`` and included in
        ``.table``.

    Examples
    --------
    >>> import soundscapy as sspy
    >>> from soundscapy.satp import fit_circe
    >>> data = sspy.isd.load()
    >>> data = data.rename(columns={'SessionID': 'participant'})
    >>> results = fit_circe(data, language='eng', datasource='ISD', errors='warn')
    >>> len(results)
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
            "Check that data contains valid rows with PAQ1-PAQ8 columns."
        )
        raise ValueError(msg)

    try:
        validated = SATPSchema.validate(data, lazy=True)
    except SchemaErrors as exc:
        if errors == "raise":
            raise
        bad_idx = exc.failure_cases["index"].dropna().unique()
        clean = data.loc[~data.index.isin(bad_idx)]
        warnings.warn(
            f"Dropping {len(data) - len(clean)} rows that failed schema validation "
            f"({len(clean)} rows remain). "
            "Pass errors='raise' to raise an error instead.",
            UserWarning,
            stacklevel=2,
        )
        try:
            validated = SATPSchema.validate(clean, lazy=True)
        except SchemaErrors as exc2:
            raise SchemaErrors(
                schema_errors=exc2.schema_errors,
                data=exc2.data,
            ) from exc2

    if center_by_participant and "participant" not in validated.columns:
        msg = (
            "center_by_participant=True requires a 'participant' column. "
            "Pass center_by_participant=False if your data is already centered."
        )
        raise ValueError(msg)
    processed = (
        ipsatize(validated, method="grand_mean", participant_col="participant")
        if center_by_participant
        else validated
    )

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
    fitted: list[CircE] = []
    error_rows: list[dict] = []
    fit_exceptions = (ValueError, np.linalg.LinAlgError, RuntimeError, RRuntimeError)
    for model in circ_models:
        try:
            circe = CircE.compute_bfgs_fit(corr, n, datasource, language, model)
            fitted.append(circe)
        except fit_exceptions as e:  # noqa: PERF203
            warnings.warn(f"{model.value} raised {e}", stacklevel=2)
            # Populate all expected columns with None so that pandas does not
            # promote numeric columns (e.g. n, d) to float64 across all rows
            # when mixing sparse error dicts with full success dicts.
            error_rows.append(
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
                    **dict.fromkeys(PAQ_IDS),
                    "error": str(e),
                }
            )

    return CircEResults(
        models=fitted,
        language=language,
        datasource=datasource,
        error_rows=error_rows,
    )
