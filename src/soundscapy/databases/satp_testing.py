"""
Module for translation testing of soundscape attributes based on SATP methodology.

This module provides functions for computing translation quality metrics and
statistical tests for validating translations of soundscape attributes. It is
based on the methodology from the Soundscape Attributes Translation Project (SATP).

The translation quality is assessed using multiple criteria:
- APPR: Appropriateness (0-1 scale, normalized from 0-10)
- UNDR: Understandability (0-1 scale, normalized from 0-10)
- CLAR: Clarity (computed from association with word and counter-word)
- ANTO: Antonymy (for main axes only)
- ORTH: Orthogonality (for main axes only)
- NCON: Non-confusability (for main axes only)
- IBAL: Importance balance
- CONN: Connectedness (for derived axes only)

Examples
--------
>>> import pandas as pd
>>> import soundscapy.databases.satp_testing as satp_testing  # doctest: +SKIP
>>> # Compute main axis criteria from raw data
>>> df = pd.DataFrame({  # doctest: +SKIP
...     'COUNTRY': ['SG', 'SG'],
...     'APPR': [8.0, 9.0],
...     'UNDR': [7.5, 8.5],
...     'ANTO': [8.0, 7.0],
...     'BIAS': [5.0, 4.5],
...     'ASSOCCW': [6.0, 5.5],
...     'IMPCCW': [4.0, 3.5],
...     'ASSOCW': [7.0, 6.5],
...     'IMPCW': [5.0, 4.5],
...     'CANDIDATE': ['pleasant', 'pleasant']
... })
>>> result = satp_testing.compute_main_axis_criteria(df)  # doctest: +SKIP
>>> 'CLAR' in result.columns  # doctest: +SKIP
True

References
----------
Based on the R code from the SATP project:
https://github.com/ntudsp/satp-zsm-stage1

"""

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

# Constants for statistical tests
MANN_WHITNEY_NUM_GROUPS = 2  # Mann-Whitney test requires exactly 2 groups


def compute_main_axis_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute translation quality criteria for main axis attributes.

    Main axes are the primary soundscape dimensions (e.g., pleasant-annoying,
    eventful-uneventful) that have antonymic relationships.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw survey responses with columns:
        - APPR: Appropriateness rating (0-10 scale)
        - UNDR: Understandability rating (0-10 scale)
        - ANTO: Antonymy rating (0-10 scale)
        - BIAS: Bias rating (0-10 scale)
        - ASSOCW: Association with word (0-10 scale)
        - ASSOCCW: Association with counter-word (0-10 scale)
        - IMPCW: Importance with word (0-10 scale)
        - IMPCCW: Importance with counter-word (0-10 scale)
        Additional columns (e.g., COUNTRY, CANDIDATE) will be preserved.

    Returns
    -------
    pd.DataFrame
        DataFrame with computed criteria:
        - APPR: Normalized appropriateness (0-1 scale)
        - UNDR: Normalized understandability (0-1 scale)
        - ANTO: Normalized antonymy (0-1 scale)
        - CLAR: Clarity (0-1 scale, higher is better)
        - ORTH: Orthogonality (0-1 scale, higher is better)
        - NCON: Non-confusability (0-1 scale, higher is better)
        - IBAL: Importance balance (0-1 scale, higher is better)
        Plus all original non-computed columns.

    Notes
    -----
    The criteria are computed as follows:
    - APPR, UNDR, ANTO: Normalized from 0-10 to 0-1 scale
    - CLAR: 1 - 0.5*(ASSOCW/10) - 0.5*(ASSOCCW/10)
    - ORTH: 1 - 2*|BIAS/10 - 0.5|
    - NCON: 1 - 0.5*(IMPCW/10 + IMPCCW/10)
    - IBAL: 1 - |IMPCCW/10 - IMPCW/10|

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'APPR': [8.0, 9.0],
    ...     'UNDR': [7.5, 8.5],
    ...     'ANTO': [8.0, 7.0],
    ...     'BIAS': [5.0, 4.5],
    ...     'ASSOCCW': [6.0, 5.5],
    ...     'IMPCCW': [4.0, 3.5],
    ...     'ASSOCW': [7.0, 6.5],
    ...     'IMPCW': [5.0, 4.5],
    ...     'CANDIDATE': ['pleasant', 'pleasant']
    ... })
    >>> result = compute_main_axis_criteria(df)
    >>> result['APPR'].iloc[0]
    0.8
    >>> round(result['CLAR'].iloc[0], 2)
    0.35
    >>> result['ORTH'].iloc[0]
    1.0

    """
    result = df.copy()

    # Normalize 0-10 scale to 0-1 scale
    result["APPR"] = result["APPR"] / 10
    result["UNDR"] = result["UNDR"] / 10
    result["ANTO"] = result["ANTO"] / 10

    # Compute derived criteria
    result["CLAR"] = 1 - 0.5 * result["ASSOCW"] / 10 - 0.5 * result["ASSOCCW"] / 10
    result["ORTH"] = 1 - 2 * np.abs(result["BIAS"] / 10 - 0.5)
    result["NCON"] = 1 - 0.5 * (result["IMPCW"] / 10 + result["IMPCCW"] / 10)
    result["IBAL"] = 1 - np.abs(result["IMPCCW"] / 10 - result["IMPCW"] / 10)

    logger.debug("Computed main axis criteria for dataset")
    return result


def compute_derived_axis_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute translation quality criteria for derived axis attributes.

    Derived axes are secondary soundscape dimensions (e.g., vibrant, calm,
    monotonous, chaotic) that do not necessarily have direct antonyms.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw survey responses with columns:
        - APPR: Appropriateness rating (0-10 scale)
        - UNDR: Understandability rating (0-10 scale)
        - ASSOCW: Association with word (0-10 scale)
        - ASSOCCW: Association with counter-word (0-10 scale)
        - IMPCW: Importance with word (0-10 scale)
        - IMPCCW: Importance with counter-word (0-10 scale)
        Additional columns (e.g., COUNTRY, CANDIDATE) will be preserved.

    Returns
    -------
    pd.DataFrame
        DataFrame with computed criteria:
        - APPR: Normalized appropriateness (0-1 scale)
        - UNDR: Normalized understandability (0-1 scale)
        - CLAR: Clarity (0-1 scale, higher is better)
        - CONN: Connectedness (0-1 scale, higher is better)
        - IBAL: Importance balance (0-1 scale, higher is better)
        Plus all original non-computed columns.

    Notes
    -----
    The criteria are computed as follows:
    - APPR, UNDR: Normalized from 0-10 to 0-1 scale
    - CLAR: 1 - 0.5*(ASSOCW/10) - 0.5*(ASSOCCW/10)
    - CONN: 0.5*(IMPCW/10 + IMPCCW/10)
    - IBAL: 1 - |IMPCCW/10 - IMPCW/10|

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'APPR': [8.0, 9.0],
    ...     'UNDR': [7.5, 8.5],
    ...     'ASSOCCW': [6.0, 5.5],
    ...     'IMPCCW': [4.0, 3.5],
    ...     'ASSOCW': [7.0, 6.5],
    ...     'IMPCW': [5.0, 4.5],
    ...     'CANDIDATE': ['vibrant', 'vibrant']
    ... })
    >>> result = compute_derived_axis_criteria(df)
    >>> result['APPR'].iloc[0]
    0.8
    >>> result['CONN'].iloc[0]
    0.45

    """
    result = df.copy()

    # Normalize 0-10 scale to 0-1 scale
    result["APPR"] = result["APPR"] / 10
    result["UNDR"] = result["UNDR"] / 10

    # Compute derived criteria
    result["CLAR"] = 1 - 0.5 * result["ASSOCW"] / 10 - 0.5 * result["ASSOCCW"] / 10
    result["CONN"] = 0.5 * (result["IMPCW"] / 10 + result["IMPCCW"] / 10)
    result["IBAL"] = 1 - np.abs(result["IMPCCW"] / 10 - result["IMPCW"] / 10)

    logger.debug("Computed derived axis criteria for dataset")
    return result


def summarize_main_axis(
    df: pd.DataFrame,
    by_country: bool = False,  # noqa: FBT001, FBT002
) -> pd.DataFrame:
    """
    Summarize main axis criteria by candidate translation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with computed main axis criteria (output of
        compute_main_axis_criteria).
    by_country : bool, optional
        If True, group by both COUNTRY and CANDIDATE.
        If False, group by CANDIDATE only. Default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame with mean values for each criterion, grouped by
        CANDIDATE (and COUNTRY if by_country=True).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'CANDIDATE': ['pleasant', 'pleasant', 'annoying', 'annoying'],
    ...     'COUNTRY': ['SG', 'MY', 'SG', 'MY'],
    ...     'APPR': [0.8, 0.85, 0.75, 0.8],
    ...     'UNDR': [0.75, 0.8, 0.7, 0.75],
    ...     'CLAR': [0.6, 0.65, 0.55, 0.6],
    ...     'ANTO': [0.8, 0.85, 0.75, 0.8],
    ...     'ORTH': [0.9, 0.95, 0.85, 0.9],
    ...     'NCON': [0.7, 0.75, 0.65, 0.7],
    ...     'IBAL': [0.85, 0.9, 0.8, 0.85]
    ... })
    >>> result = summarize_main_axis(df, by_country=False)
    >>> len(result)
    2
    >>> 'APPR' in result.columns
    True

    """
    criteria = ["APPR", "UNDR", "CLAR", "ANTO", "ORTH", "NCON", "IBAL"]

    if by_country:
        grouped = df.groupby(["COUNTRY", "CANDIDATE"])[criteria].mean().reset_index()
    else:
        grouped = df.groupby("CANDIDATE")[criteria].mean().reset_index()

    logger.debug(f"Summarized main axis data (by_country={by_country})")
    return grouped


def summarize_derived_axis(
    df: pd.DataFrame,
    by_country: bool = False,  # noqa: FBT001, FBT002
) -> pd.DataFrame:
    """
    Summarize derived axis criteria by candidate translation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with computed derived axis criteria (output of
        compute_derived_axis_criteria).
    by_country : bool, optional
        If True, group by both COUNTRY and CANDIDATE.
        If False, group by CANDIDATE only. Default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame with mean values for each criterion, grouped by
        CANDIDATE (and COUNTRY if by_country=True).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'CANDIDATE': ['vibrant', 'vibrant', 'calm', 'calm'],
    ...     'COUNTRY': ['SG', 'MY', 'SG', 'MY'],
    ...     'APPR': [0.8, 0.85, 0.75, 0.8],
    ...     'UNDR': [0.75, 0.8, 0.7, 0.75],
    ...     'CLAR': [0.6, 0.65, 0.55, 0.6],
    ...     'CONN': [0.7, 0.75, 0.65, 0.7],
    ...     'IBAL': [0.85, 0.9, 0.8, 0.85]
    ... })
    >>> result = summarize_derived_axis(df, by_country=False)
    >>> len(result)
    2
    >>> 'CONN' in result.columns
    True

    """
    criteria = ["APPR", "UNDR", "CLAR", "CONN", "IBAL"]

    if by_country:
        grouped = df.groupby(["COUNTRY", "CANDIDATE"])[criteria].mean().reset_index()
    else:
        grouped = df.groupby("CANDIDATE")[criteria].mean().reset_index()

    logger.debug(f"Summarized derived axis data (by_country={by_country})")
    return grouped


def kruskal_wallis_test(
    df: pd.DataFrame,
    axis_type: Literal["main", "derived"],
    independent_var: str = "CANDIDATE"
) -> pd.DataFrame:
    """
    Perform Kruskal-Wallis test for each criterion across groups.

    The Kruskal-Wallis test is a non-parametric test to determine if there
    are significant differences between groups. This is used to test if
    different translation candidates have significantly different quality scores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with computed criteria.
    axis_type : {"main", "derived"}
        Type of axis being tested.
    independent_var : str, optional
        Column name to group by for the test. Default is "CANDIDATE".

    Returns
    -------
    pd.DataFrame
        DataFrame with test results including:
        - CRITERION: Name of the criterion tested
        - statistic: Kruskal-Wallis H statistic
        - pvalue: p-value for the test
        - effect_size: Effect size (epsilon squared)

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     'CANDIDATE': ['A'] * 10 + ['B'] * 10 + ['C'] * 10,
    ...     'APPR': np.random.uniform(0.5, 1.0, 30),
    ...     'UNDR': np.random.uniform(0.5, 1.0, 30),
    ...     'CLAR': np.random.uniform(0.4, 0.9, 30),
    ...     'CONN': np.random.uniform(0.5, 0.9, 30),
    ...     'IBAL': np.random.uniform(0.7, 1.0, 30)
    ... })
    >>> result = kruskal_wallis_test(df, axis_type="derived")
    >>> len(result) == 5  # One row per criterion (derived has 5 criteria)
    True
    >>> 'pvalue' in result.columns
    True

    """
    if axis_type == "main":
        criteria = ["APPR", "UNDR", "CLAR", "ANTO", "ORTH", "NCON", "IBAL"]
    elif axis_type == "derived":
        criteria = ["APPR", "UNDR", "CLAR", "CONN", "IBAL"]
    else:
        msg = "axis_type must be either 'main' or 'derived'"
        raise ValueError(msg)

    results = []

    for criterion in criteria:
        # Group data by independent variable
        groups = [
            group[criterion].to_numpy()
            for name, group in df.groupby(independent_var)
        ]

        # Perform Kruskal-Wallis test
        statistic, pvalue = stats.kruskal(*groups)

        # Calculate effect size (epsilon squared)
        n = len(df)
        k = len(groups)
        effect_size = (statistic - k + 1) / (n - k)

        results.append({
            "CRITERION": criterion,
            "statistic": statistic,
            "pvalue": pvalue,
            "effect_size": effect_size
        })

    result_df = pd.DataFrame(results)
    logger.debug(f"Performed Kruskal-Wallis test for {axis_type} axis")
    return result_df


def mann_whitney_test(
    df: pd.DataFrame,
    criteria: list[str],
    paq_attribute: str,
    group_var: str = "COUNTRY"
) -> pd.DataFrame:
    """
    Perform Mann-Whitney-Wilcoxon test for each criterion and candidate.

    This test compares two groups (e.g., two countries) for each translation
    candidate to determine if there are significant differences in quality scores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with computed criteria.
    criteria : list[str]
        List of criteria column names to test.
    paq_attribute : str
        Name of the PAQ (Perceived Affective Quality) attribute being tested
        (e.g., "pleasant", "eventful").
    group_var : str, optional
        Column name for grouping variable (e.g., "COUNTRY"). Default is "COUNTRY".

    Returns
    -------
    pd.DataFrame
        DataFrame with test results including:
        - PAQ: PAQ attribute name
        - CRITERION: Name of the criterion tested
        - CANDIDATE: Translation candidate
        - statistic: U statistic
        - pvalue: p-value for the test
        - adjusted_pvalue: Bonferroni-adjusted p-value

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     'COUNTRY': ['SG'] * 10 + ['MY'] * 10,
    ...     'CANDIDATE': ['pleasant'] * 20,
    ...     'APPR': np.random.uniform(0.5, 1.0, 20),
    ...     'UNDR': np.random.uniform(0.5, 1.0, 20)
    ... })
    >>> result = mann_whitney_test(df, ['APPR', 'UNDR'], 'pleasant')
    >>> len(result) == 2  # One row per criterion
    True

    """
    results = []

    for criterion in criteria:
        for candidate in df["CANDIDATE"].unique():
            # Filter data for this candidate
            candidate_data = df[df["CANDIDATE"] == candidate]

            # Get groups
            groups = candidate_data[group_var].unique()
            if len(groups) != MANN_WHITNEY_NUM_GROUPS:
                logger.warning(
                    f"Skipping {candidate} for {criterion}: "
                    f"Expected {MANN_WHITNEY_NUM_GROUPS} groups, found {len(groups)}"
                )
                continue

            group1 = candidate_data[candidate_data[group_var] == groups[0]][criterion]
            group2 = candidate_data[candidate_data[group_var] == groups[1]][criterion]

            # Perform Mann-Whitney U test
            statistic, pvalue = stats.mannwhitneyu(
                group1, group2, alternative="two-sided"
            )

            # Bonferroni correction (multiply by number of groups)
            adjusted_pvalue = min(pvalue * MANN_WHITNEY_NUM_GROUPS, 1.0)

            results.append({
                "PAQ": paq_attribute,
                "CRITERION": criterion,
                "CANDIDATE": candidate,
                "statistic": statistic,
                "pvalue": pvalue,
                "adjusted_pvalue": adjusted_pvalue
            })

    result_df = pd.DataFrame(results)
    logger.debug(f"Performed Mann-Whitney test for {paq_attribute}")
    return result_df


if __name__ == "__main__":
    import xdoctest

    xdoctest.doctest_module(__file__)
