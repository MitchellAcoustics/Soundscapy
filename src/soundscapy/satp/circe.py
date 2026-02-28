"""
Circumplex SEM Analysis for Soundscape Attributes Translation Project (SATP).

This module provides tools for analyzing soundscape perception data using circumplex
Structural Equation Modeling (SEM). It includes data validation schemas, model fitting
classes, and analysis workflows for the Soundscape Attributes Translation Project.

The module supports various circumplex model types (unconstrained, equal angles,
equal communalities, and full circumplex) and provides automated data preprocessing
including ipsatization (participant-wise centering).

Classes
-------
CircModelE : Enum
    Enumeration of available circumplex model types
SATPSchema : DataFrameModel
    Pandera schema for validating SATP data format
ModelType : dataclass
    Wrapper for circumplex model properties
CircE : dataclass
    Results container for fitted circumplex models
SATP : class
    Main analysis class for SATP workflow

Functions
---------
length_1_array_to_number : function
    Validator for converting single-element arrays to scalars
"""

import warnings
from enum import Enum
from functools import partial
from typing import Annotated, Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera import Field
from pandera.typing.pandas import DataFrame, Series
from pydantic import BeforeValidator, ConfigDict
from pydantic.dataclasses import dataclass

import soundscapy.r_wrapper as sspyr
from soundscapy import PAQ_IDS, PAQ_LABELS, get_logger

logger = get_logger()

# Create a partial Field function that allows NaN values for optional data columns
AllowNan = partial(pa.Field, nullable=True)


class CircModelE(str, Enum):
    """Enumeration of circumplex model types."""

    UNCONSTRAINED = "unconstrained"
    EQUAL_ANG = "equal_ang"
    EQUAL_COM = "equal_com"
    CIRCUMPLEX = "circumplex"


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

    participant: Series[str]

    class Config:
        """Configuration for the schema validation behavior."""

        drop_invalid_rows = True
        strict = "filter"

    @pa.dataframe_parser
    def column_alias(cls, df: DataFrame) -> DataFrame:  # noqa: N805
        """
        Parse and rename DataFrame columns to match the schema.

        Maps PAQ label names to standardized PAQ IDs and converts
        'Participant' column to lowercase 'participant'.

        Parameters
        ----------
        df
            Input DataFrame to rename columns for

        Returns
        -------
        DataFrame with renamed columns matching the schema

        """
        rename_dict = dict(zip(PAQ_LABELS, PAQ_IDS, strict=False))
        rename_dict.update(
            {
                "Participant": "participant",
            }
        )
        return df.rename(columns=rename_dict)


@dataclass
class ModelType:
    """A data class representing a circumplex model type with its properties."""

    name: CircModelE

    @property
    def equal_ang(self) -> bool:
        """
        Check if the model uses equal angles constraint.

        True for EQUAL_ANG (angles only) and CIRCUMPLEX (both constraints).
        EQUAL_COM has free angles (False); UNCONSTRAINED has neither (False).
        """
        return self.name in {CircModelE.EQUAL_ANG, CircModelE.CIRCUMPLEX}

    @property
    def equal_com(self) -> bool:
        """
        Check if the model uses equal communalities constraint.

        True for EQUAL_COM (communalities only) and CIRCUMPLEX (both).
        EQUAL_ANG has free communalities (False); UNCONSTRAINED neither (False).
        """
        return self.name in {CircModelE.EQUAL_COM, CircModelE.CIRCUMPLEX}


def length_1_array_to_number(v: np.ndarray | float | None) -> float | None:
    """
    Convert a single-element numpy array to a scalar number.

    This validator function is used with Pydantic to handle R-returned values
    that come as single-element arrays but should be treated as scalars.

    Parameters
    ----------
    v
        Input value that may be a single-element array, scalar, or None

    Returns
    -------
    Converted scalar value or None if input was None

    Raises
    ------
    ValueError
        If input is an array with more than one element

    """
    """Validate a length-1 numpy array to a float."""
    if v is None or isinstance(v, float | int):
        return v
    if isinstance(v, np.ndarray) and v.size == 1:
        return float(v.item())
    msg = "Value must be a numpy array with a single element."
    raise ValueError(msg)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CircE:
    """A data class to hold the results of a CircE model fitting."""

    model_type: ModelType
    datasource: str
    language: str
    n: Annotated[int, BeforeValidator(length_1_array_to_number)]
    m: Annotated[int, BeforeValidator(length_1_array_to_number)]
    chisq: Annotated[float, BeforeValidator(length_1_array_to_number)]
    df: Annotated[int, BeforeValidator(length_1_array_to_number)]
    p: Annotated[float, BeforeValidator(length_1_array_to_number)]
    cfi: Annotated[float, BeforeValidator(length_1_array_to_number)]
    gfi: Annotated[float, BeforeValidator(length_1_array_to_number)]
    agfi: Annotated[float, BeforeValidator(length_1_array_to_number)]
    srmr: Annotated[float, BeforeValidator(length_1_array_to_number)]
    mcsc: Annotated[float, BeforeValidator(length_1_array_to_number)]
    rmsea: Annotated[float, BeforeValidator(length_1_array_to_number)]
    rmsea_l: Annotated[float, BeforeValidator(length_1_array_to_number)]
    rmsea_u: Annotated[float, BeforeValidator(length_1_array_to_number)]
    polar_angles: pd.DataFrame | None = None

    @classmethod
    def from_bfgs(
        cls,
        bfgs_model: Any,
        datasource: str,
        language: str,
        model_type: ModelType,
        n: int,
    ) -> "CircE":
        """Create a CircE instance from a fitted BFGS model."""
        fit_stats = sspyr.extract_bfgs_fit(bfgs_model)
        polar_angles = None
        # Only extract polar angles for models where angles are free parameters.
        # model_type.name is the CircModelE enum; compare against that, not the
        # ModelType dataclass wrapper (which would never compare equal to an enum).
        # The R key is "polar.angles" (dot), not "polar_angles" (underscore).
        if model_type.name in (CircModelE.UNCONSTRAINED, CircModelE.EQUAL_COM):
            raw_pa = fit_stats.get("polar.angles")
            if raw_pa is not None:
                polar_angles = pd.DataFrame(raw_pa).T

        return cls(
            model_type=model_type,
            datasource=datasource,
            language=language,
            n=n,
            m=fit_stats.get("m", None),
            chisq=fit_stats.get("chisq", None),
            df=fit_stats.get("d", None),
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
        Compute and return a CircEResult from the given data correlation matrix.

        Parameters
        ----------
        data_cor
            Correlation matrix of the PAQ data (8x8).
        n
            Number of observations (participants) used to compute ``data_cor``.
            This is used by ``CircE_BFGS`` for chi-square and RMSEA calculations
            and must be the row count of the *original* data, not of the
            correlation matrix.
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
        model_type = ModelType(name=circ_model)
        bfgs_model = sspyr.bfgs(
            data_cor=data_cor,
            n=n,
            scales=PAQ_IDS,
            m_val=3,
            equal_ang=model_type.equal_ang,
            equal_com=model_type.equal_com,
        )
        return cls.from_bfgs(bfgs_model, datasource, language, model_type, n)


class SATP:
    """
    Soundscape Attributes Translation Project (SATP) analysis class.

    This class handles the analysis of soundscape perception data using
    circumplex SEM models. It validates input data, performs ipsatization (centering),
    and fits various circumplex model types to correlation matrices.

    Attributes
    ----------
    data
        Validated DataFrame containing PAQ ratings and participant identifiers
    language
        Language code for the dataset
    datasource
        Source identifier for the dataset
    model_results
        Dictionary storing fitted CircE models for each model type

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame(
    ... {'PAQ1': [50.0, 60.0], 'PAQ2': [40.0, 70.0], 'PAQ3': [20.0, 10.0],
    ... 'PAQ4': [50.0, 50.0], 'PAQ5': [100.0, 0.0], 'PAQ6': [10.0, 10.0],
    ... 'PAQ7': [50.0, 50.0], 'PAQ8': [45.0, 24.0], 'participant': ['A', 'B']})
    >>> satp = SATP(data, language='EN', datasource='test')
    >>> satp.run()

    """

    def __init__(
        self,
        data: pd.DataFrame,
        language: str | None,
        datasource: str | None,
        *,
        ipsatize_data: bool = True,
    ) -> None:
        """
        Initialize SATP analysis with input data.

        Parameters
        ----------
        data
            DataFrame containing PAQ ratings and participant identifiers
        language
            Language code for the dataset (e.g., 'EN', 'FR')
        datasource
            Source identifier for the dataset
        ipsatize_data
            Whether to apply ipsatization (participant-wise centering)

        Raises
        ------
        ValidationError
            If data doesn't conform to SATPSchema requirements

        """
        warnings.warn(
            "The SATP analysis module is experimental. Use with caution.",
            UserWarning,
            stacklevel=2,
        )
        # Initialize processing flags and store raw data
        self._ipsatized = False
        self._raw_data = data
        # Validate input data against schema requirements
        self.data: DataFrame = SATPSchema.validate(data, lazy=True)

        # Apply ipsatization if requested
        if ipsatize_data:
            self.ipsatize()

        # Store metadata for model results
        self.language = language
        self.datasource = datasource
        # Initialize containers for model results and any errors
        self.model_results: dict[CircModelE, CircE | None] = {
            CircModelE.UNCONSTRAINED: None,
            CircModelE.EQUAL_COM: None,
            CircModelE.EQUAL_ANG: None,
            CircModelE.CIRCUMPLEX: None,
        }
        self._errors: dict[CircModelE, Exception] = {}

    @property
    def data_corr(self) -> pd.DataFrame:
        """
        Compute and return the correlation matrix of PAQ ratings.

        Returns
        -------
        Correlation matrix of the validated PAQ data

        """
        return self.data.corr()

    def ipsatize(self) -> None:
        """
        Apply ipsatization (participant-wise centering) to the data.

        Ipsatization centers each participant's responses around their mean,
        removing individual response style differences while preserving
        relative response patterns.

        Calling this method a second time is a no-op (guarded by
        ``_ipsatized``): after the first call the ``participant`` column is
        dropped by ``groupby.transform``, so a second call would raise
        ``KeyError``.
        """
        if self._ipsatized:
            logger.warning("Data has already been ipsatized; skipping.")
            return
        # Apply ipsatization transformation and update flag
        self.data = self._ipsatize_df(self.data, by="participant")
        self._ipsatized = True

    @staticmethod
    def _ipsatize_df(df: DataFrame, by: str = "participant") -> DataFrame:
        """
        Apply ipsatization transformation to a DataFrame.

        Parameters
        ----------
        df
            Input DataFrame to transform
        by
            Column name to group by for centering (default: "participant")

        Returns
        -------
        DataFrame with participant-centered values

        """
        # Group by specified column and center each group around its mean
        return df.groupby(by).transform(lambda x: x - x.mean())

    def run(self, circ_model: CircModelE | None = None) -> None:
        """
        Fit circumplex models to the correlation matrix.

        Parameters
        ----------
        circ_model
            Specific model type to fit. If None, fits all model types.

        Notes
        -----
        Results are stored in self.model_results. Any fitting errors are
        captured in self._errors and warnings are issued.

        """
        # Determine which models to fit
        circ_models_to_run = [*CircModelE] if circ_model is None else [circ_model]
        n = len(self.data)
        # Fit each requested model, capturing any errors
        for model in circ_models_to_run:
            try:
                self.model_results[model] = CircE.compute_bfgs_fit(
                    self.data_corr, n, self.datasource, self.language, model
                )
            except Exception as e:  # noqa: BLE001, PERF203
                # Log fitting errors but continue with other models
                warnings.warn(f"{model.value} raised {e}", stacklevel=2)
                self._errors[model] = e
