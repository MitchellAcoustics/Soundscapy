"""
Tests for the survey_utils module.

This module contains tests for the utility functions and classes in the
soundscapy.surveys.survey_utils module.
"""

import pandas as pd
import pytest

from soundscapy.surveys.survey_utils import (
    EQUAL_ANGLES,
    LANGUAGE_ANGLES,
    LIKERT_SCALES,
    PAQ,
    PAQ_IDS,
    PAQ_LABELS,
    LikertScale,
    mean_responses,
    rename_paqs,
    return_paqs,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with PAQ data."""
    return pd.DataFrame(
        {
            "RecordID": [1, 2, 3, 4],
            "GroupID": ["A", "A", "B", "B"],
            "PAQ1": [4, 3, 5, 2],
            "PAQ2": [2, 5, 3, 4],
            "PAQ3": [1, 2, 4, 5],
            "PAQ4": [3, 4, 2, 1],
            "PAQ5": [5, 1, 3, 4],
            "PAQ6": [2, 3, 5, 2],
            "PAQ7": [4, 5, 1, 3],
            "PAQ8": [1, 2, 4, 5],
            "OtherCol": ["W", "X", "Y", "Z"],
        }
    )


@pytest.fixture
def sample_df_with_paq_names():
    """Create a sample DataFrame with PAQ attribute names as column names."""
    return pd.DataFrame(
        {
            "RecordID": [1, 2, 3, 4],
            "pleasant": [4, 3, 5, 2],
            "vibrant": [2, 5, 3, 4],
            "eventful": [1, 2, 4, 5],
            "chaotic": [3, 4, 2, 1],
            "annoying": [5, 1, 3, 4],
            "monotonous": [2, 3, 5, 2],
            "uneventful": [4, 5, 1, 3],
            "calm": [1, 2, 4, 5],
            "OtherCol": ["W", "X", "Y", "Z"],
        }
    )


@pytest.fixture
def sample_df_with_likert_labels():
    """Create a sample DataFrame with string Likert labels instead of numeric values."""
    # Map numeric values to Likert scale labels
    likert_map = {
        1: "Strongly disagree",
        2: "Somewhat disagree",
        3: "Neutral",
        4: "Somewhat agree",
        5: "Strongly agree",
    }

    # Create DataFrame with numeric values
    df = pd.DataFrame(
        {
            "RecordID": [1, 2, 3, 4],
            "GroupID": ["A", "A", "B", "B"],
            "PAQ1": [4, 3, 5, 2],
            "PAQ2": [2, 5, 3, 4],
            "PAQ3": [1, 2, 4, 5],
            "PAQ4": [3, 4, 2, 1],
            "PAQ5": [5, 1, 3, 4],
            "PAQ6": [2, 3, 5, 2],
            "PAQ7": [4, 5, 1, 3],
            "PAQ8": [1, 2, 4, 5],
            "OtherCol": ["W", "X", "Y", "Z"],
        }
    )

    # Replace numeric values with Likert labels
    for col in PAQ_IDS:
        df[col] = df[col].map(likert_map)

    return df


@pytest.fixture
def sample_df_custom_names():
    """Create a sample DataFrame with custom column names."""
    return pd.DataFrame(
        {
            "RecordID": [1, 2, 3, 4],
            "plsnt": [4, 3, 5, 2],
            "vbrnt": [2, 5, 3, 4],
            "evntfl": [1, 2, 4, 5],
            "chtc": [3, 4, 2, 1],
            "annng": [5, 1, 3, 4],
            "mntnous": [2, 3, 5, 2],
            "unevntfl": [4, 5, 1, 3],
            "clm": [1, 2, 4, 5],
            "OtherCol": ["W", "X", "Y", "Z"],
        }
    )


class TestPAQEnum:
    """Tests for the PAQ enum class."""

    def test_paq_enum_values(self):
        """Test that PAQ enum has the expected values."""
        assert len(PAQ) == 8
        assert PAQ.PLEASANT.label == "pleasant"
        assert PAQ.PLEASANT.id == "PAQ1"
        assert PAQ.VIBRANT.label == "vibrant"
        assert PAQ.VIBRANT.id == "PAQ2"
        assert PAQ.EVENTFUL.label == "eventful"
        assert PAQ.EVENTFUL.id == "PAQ3"
        assert PAQ.CHAOTIC.label == "chaotic"
        assert PAQ.CHAOTIC.id == "PAQ4"
        assert PAQ.ANNOYING.label == "annoying"
        assert PAQ.ANNOYING.id == "PAQ5"
        assert PAQ.MONOTONOUS.label == "monotonous"
        assert PAQ.MONOTONOUS.id == "PAQ6"
        assert PAQ.UNEVENTFUL.label == "uneventful"
        assert PAQ.UNEVENTFUL.id == "PAQ7"
        assert PAQ.CALM.label == "calm"
        assert PAQ.CALM.id == "PAQ8"

    def test_paq_labels_constant(self):
        """Test that PAQ_LABELS contains all PAQ labels in the correct order."""
        assert PAQ_LABELS == [
            "pleasant",
            "vibrant",
            "eventful",
            "chaotic",
            "annoying",
            "monotonous",
            "uneventful",
            "calm",
        ]
        assert len(PAQ_LABELS) == len(PAQ)
        for i, paq in enumerate(PAQ):
            assert PAQ_LABELS[i] == paq.label

    def test_paq_ids_constant(self):
        """Test that PAQ_IDS contains all PAQ IDs in the correct order."""
        assert PAQ_IDS == [
            "PAQ1",
            "PAQ2",
            "PAQ3",
            "PAQ4",
            "PAQ5",
            "PAQ6",
            "PAQ7",
            "PAQ8",
        ]
        assert len(PAQ_IDS) == len(PAQ)
        for i, paq in enumerate(PAQ):
            assert PAQ_IDS[i] == paq.id


class TestLikertScale:
    """Tests for the LikertScale dataclass."""

    def test_likert_scale_default_values(self):
        """Test that LikertScale has the expected default values."""
        likert = LikertScale()

        # Test PAQ scale
        assert likert.paq == [
            "Strongly disagree",
            "Somewhat disagree",
            "Neutral",
            "Somewhat agree",
            "Strongly agree",
        ]

        # Test source scale
        assert likert.source == [
            "Not at all",
            "A little",
            "Moderately",
            "A lot",
            "Dominates completely",
        ]

        # Test overall scale
        assert likert.overall == [
            "Very bad",
            "Bad",
            "Neither bad nor good",
            "Good",
            "Very good",
        ]

        # Test appropriate scale
        assert likert.appropriate == [
            "Not at all",
            "A little",
            "Moderately",
            "A lot",
            "Perfectly",
        ]

        # Test loud scale
        assert likert.loud == [
            "Not at all",
            "A little",
            "Moderately",
            "Very",
            "Extremely",
        ]

        # Test often scale
        assert likert.often == [
            "Never / This is my first time here",
            "Rarely",
            "Sometimes",
            "Often",
            "Very often",
        ]

        # Test visit scale
        assert likert.visit == [
            "Never",
            "Rarely",
            "Sometimes",
            "Often",
            "Very often",
        ]

    def test_likert_scales_constant(self):
        """Test that LIKERT_SCALES is an instance of LikertScale."""
        assert isinstance(LIKERT_SCALES, LikertScale)


class TestConstants:
    """Tests for the constants in survey_utils."""

    def test_equal_angles(self):
        """Test that EQUAL_ANGLES has the expected values."""
        assert EQUAL_ANGLES == (0, 45, 90, 135, 180, 225, 270, 315)
        assert len(EQUAL_ANGLES) == 8

    def test_language_angles(self):
        """Test that LANGUAGE_ANGLES has the expected structure."""
        # Check that LANGUAGE_ANGLES is a dictionary
        assert isinstance(LANGUAGE_ANGLES, dict)

        # Check that it contains the expected languages
        expected_languages = [
            "eng",
            "arb",
            "cmn",
            "hrv",
            "nld",
            "deu",
            "ell",
            "ind",
            "ita",
            "spa",
            "swe",
            "tur",
        ]
        for lang in expected_languages:
            assert lang in LANGUAGE_ANGLES

        # Check that each language has 8 angles
        for lang, angles in LANGUAGE_ANGLES.items():
            assert len(angles) == 8

            # Check that angles are numeric
            for angle in angles:
                assert isinstance(angle, (int, float))


class TestFixtures:
    """Tests for the fixtures."""

    def test_sample_df_with_likert_labels(self, sample_df_with_likert_labels):
        """Test that sample_df_with_likert_labels has string Likert labels instead of numeric values."""
        # Check that the DataFrame has the expected columns
        expected_cols = ["RecordID", "GroupID"] + PAQ_IDS + ["OtherCol"]
        assert list(sample_df_with_likert_labels.columns) == expected_cols

        # Check that PAQ columns contain string values, not numeric
        for col in PAQ_IDS:
            assert (
                sample_df_with_likert_labels[col].dtype == "object"
            )  # String type in pandas

            # Check that values are from the Likert scale
            likert_values = set(LIKERT_SCALES.paq)
            assert set(sample_df_with_likert_labels[col].unique()).issubset(
                likert_values
            )

            # Check that there are no numeric values
            assert not any(
                isinstance(val, (int, float))
                for val in sample_df_with_likert_labels[col]
            )


class TestReturnPAQs:
    """Tests for the return_paqs function."""

    def test_return_paqs_with_ids(self, sample_df):
        """Test return_paqs with default parameters."""
        result = return_paqs(sample_df)

        # Check that result has the expected columns
        expected_cols = ["RecordID", "GroupID"] + PAQ_IDS
        assert list(result.columns) == expected_cols

        # Check that result has the same number of rows as input
        assert len(result) == len(sample_df)

        # Check that the data is preserved
        for col in expected_cols:
            assert (result[col] == sample_df[col]).all()

    def test_return_paqs_without_ids(self, sample_df):
        """Test return_paqs with incl_ids=False."""
        result = return_paqs(sample_df, incl_ids=False)

        # Check that result has only PAQ columns
        assert list(result.columns) == PAQ_IDS

        # Check that result has the same number of rows as input
        assert len(result) == len(sample_df)

        # Check that the data is preserved
        for col in PAQ_IDS:
            assert (result[col] == sample_df[col]).all()

    def test_return_paqs_with_other_cols(self, sample_df):
        """Test return_paqs with other_cols parameter."""
        result = return_paqs(sample_df, other_cols=["OtherCol"])

        # Check that result has the expected columns
        expected_cols = ["RecordID", "GroupID"] + PAQ_IDS + ["OtherCol"]
        assert list(result.columns) == expected_cols

        # Check that result has the same number of rows as input
        assert len(result) == len(sample_df)

        # Check that the data is preserved
        for col in expected_cols:
            assert (result[col] == sample_df[col]).all()

    def test_return_paqs_without_ids_with_other_cols(self, sample_df):
        """Test return_paqs with incl_ids=False and other_cols parameter."""
        result = return_paqs(sample_df, other_cols=["OtherCol"], incl_ids=False)

        # Check that result has the expected columns
        expected_cols = PAQ_IDS + ["OtherCol"]
        assert list(result.columns) == expected_cols

        # Check that result has the same number of rows as input
        assert len(result) == len(sample_df)

        # Check that the data is preserved
        for col in expected_cols:
            assert (result[col] == sample_df[col]).all()


class TestRenamePAQs:
    """Tests for the rename_paqs function."""

    def test_rename_paqs_with_paq_names(self, sample_df_with_paq_names):
        """Test rename_paqs with default parameters on a DataFrame with PAQ attribute names as column names."""
        result = rename_paqs(sample_df_with_paq_names)

        # Check that PAQ labels have been renamed to PAQ IDs
        expected_cols = ["RecordID"] + PAQ_IDS + ["OtherCol"]
        assert list(result.columns) == expected_cols

        # Check that the data is preserved
        assert (result["PAQ1"] == sample_df_with_paq_names["pleasant"]).all()
        assert (result["PAQ2"] == sample_df_with_paq_names["vibrant"]).all()
        assert (result["PAQ3"] == sample_df_with_paq_names["eventful"]).all()
        assert (result["PAQ4"] == sample_df_with_paq_names["chaotic"]).all()
        assert (result["PAQ5"] == sample_df_with_paq_names["annoying"]).all()
        assert (result["PAQ6"] == sample_df_with_paq_names["monotonous"]).all()
        assert (result["PAQ7"] == sample_df_with_paq_names["uneventful"]).all()
        assert (result["PAQ8"] == sample_df_with_paq_names["calm"]).all()

    def test_rename_paqs_with_list(self, sample_df_custom_names):
        """Test rename_paqs with a list of column names."""
        custom_names = [
            "plsnt",
            "vbrnt",
            "evntfl",
            "chtc",
            "annng",
            "mntnous",
            "unevntfl",
            "clm",
        ]
        result = rename_paqs(sample_df_custom_names, paq_aliases=custom_names)

        # Check that custom names have been renamed to PAQ IDs
        expected_cols = ["RecordID"] + PAQ_IDS + ["OtherCol"]
        assert list(result.columns) == expected_cols

        # Check that the data is preserved
        assert (result["PAQ1"] == sample_df_custom_names["plsnt"]).all()
        assert (result["PAQ2"] == sample_df_custom_names["vbrnt"]).all()
        assert (result["PAQ3"] == sample_df_custom_names["evntfl"]).all()
        assert (result["PAQ4"] == sample_df_custom_names["chtc"]).all()
        assert (result["PAQ5"] == sample_df_custom_names["annng"]).all()
        assert (result["PAQ6"] == sample_df_custom_names["mntnous"]).all()
        assert (result["PAQ7"] == sample_df_custom_names["unevntfl"]).all()
        assert (result["PAQ8"] == sample_df_custom_names["clm"]).all()

    def test_rename_paqs_with_dict(self, sample_df_custom_names):
        """Test rename_paqs with a dictionary mapping."""
        custom_mapping = {
            "plsnt": "PAQ1",
            "vbrnt": "PAQ2",
            "evntfl": "PAQ3",
            "chtc": "PAQ4",
            "annng": "PAQ5",
            "mntnous": "PAQ6",
            "unevntfl": "PAQ7",
            "clm": "PAQ8",
        }
        result = rename_paqs(sample_df_custom_names, paq_aliases=custom_mapping)

        # Check that custom names have been renamed to PAQ IDs
        expected_cols = ["RecordID"] + PAQ_IDS + ["OtherCol"]
        assert list(result.columns) == expected_cols

        # Check that the data is preserved
        assert (result["PAQ1"] == sample_df_custom_names["plsnt"]).all()
        assert (result["PAQ2"] == sample_df_custom_names["vbrnt"]).all()
        assert (result["PAQ3"] == sample_df_custom_names["evntfl"]).all()
        assert (result["PAQ4"] == sample_df_custom_names["chtc"]).all()
        assert (result["PAQ5"] == sample_df_custom_names["annng"]).all()
        assert (result["PAQ6"] == sample_df_custom_names["mntnous"]).all()
        assert (result["PAQ7"] == sample_df_custom_names["unevntfl"]).all()
        assert (result["PAQ8"] == sample_df_custom_names["clm"]).all()

    def test_rename_paqs_already_named(self, sample_df):
        """Test rename_paqs on a DataFrame that already has PAQ IDs."""
        result = rename_paqs(sample_df)

        # Check that columns are unchanged
        assert list(result.columns) == list(sample_df.columns)

        # Check that the data is preserved
        for col in sample_df.columns:
            assert (result[col] == sample_df[col]).all()

    def test_rename_paqs_invalid_type(self, sample_df_custom_names):
        """Test rename_paqs with an invalid paq_aliases type."""
        with pytest.raises(TypeError):
            # Pass an integer which is not a valid type for paq_aliases
            # The function expects list, tuple, dict, or None
            rename_paqs(sample_df_custom_names, paq_aliases=123)  # type: ignore


class TestMeanResponses:
    """Tests for the mean_responses function."""

    def test_mean_responses(self, sample_df):
        """Test mean_responses function."""
        result = mean_responses(sample_df, group="GroupID")

        # Check that result has the expected columns
        expected_cols = ["GroupID"] + PAQ_IDS
        assert list(result.columns) == expected_cols

        # Check that result has the expected number of rows (one per group)
        assert len(result) == 2

        # Check that the groups are preserved
        assert set(result["GroupID"]) == set(sample_df["GroupID"])

        # Check that the means are calculated correctly
        group_a = sample_df[sample_df["GroupID"] == "A"]
        group_b = sample_df[sample_df["GroupID"] == "B"]

        result_a = result[result["GroupID"] == "A"]
        result_b = result[result["GroupID"] == "B"]

        for paq_id in PAQ_IDS:
            assert result_a[paq_id].iloc[0] == group_a[paq_id].mean()
            assert result_b[paq_id].iloc[0] == group_b[paq_id].mean()


if __name__ == "__main__":
    pytest.main()
