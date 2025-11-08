"""Tests for the SATP translation testing module."""

import numpy as np
import pandas as pd
import pytest

from soundscapy.databases import satp_testing


class TestComputeMainAxisCriteria:
    """Test suite for compute_main_axis_criteria function."""
    
    def test_basic_computation(self):
        """Test basic computation of main axis criteria."""
        df = pd.DataFrame({
            'APPR': [8.0, 9.0],
            'UNDR': [7.5, 8.5],
            'ANTO': [8.0, 7.0],
            'BIAS': [5.0, 4.5],
            'ASSOCCW': [6.0, 5.5],
            'IMPCCW': [4.0, 3.5],
            'ASSOCW': [7.0, 6.5],
            'IMPCW': [5.0, 4.5],
            'CANDIDATE': ['pleasant', 'pleasant']
        })
        
        result = satp_testing.compute_main_axis_criteria(df)
        
        # Check normalization of 0-10 scale to 0-1
        assert result['APPR'].iloc[0] == 0.8
        assert result['UNDR'].iloc[0] == 0.75
        assert result['ANTO'].iloc[0] == 0.8
        
        # Check computed criteria
        assert 'CLAR' in result.columns
        assert 'ORTH' in result.columns
        assert 'NCON' in result.columns
        assert 'IBAL' in result.columns
        
        # Check that CANDIDATE column is preserved
        assert 'CANDIDATE' in result.columns
    
    def test_clar_computation(self):
        """Test CLAR (clarity) computation formula."""
        df = pd.DataFrame({
            'APPR': [5.0], 'UNDR': [5.0], 'ANTO': [5.0], 'BIAS': [5.0],
            'ASSOCCW': [4.0], 'IMPCCW': [5.0],
            'ASSOCW': [6.0], 'IMPCW': [5.0],
            'CANDIDATE': ['test']
        })
        
        result = satp_testing.compute_main_axis_criteria(df)
        # CLAR = 1 - 0.5*(6.0/10) - 0.5*(4.0/10) = 1 - 0.3 - 0.2 = 0.5
        assert result['CLAR'].iloc[0] == pytest.approx(0.5)
    
    def test_orth_computation(self):
        """Test ORTH (orthogonality) computation formula."""
        df = pd.DataFrame({
            'APPR': [5.0], 'UNDR': [5.0], 'ANTO': [5.0], 'BIAS': [5.0],
            'ASSOCCW': [5.0], 'IMPCCW': [5.0],
            'ASSOCW': [5.0], 'IMPCW': [5.0],
            'CANDIDATE': ['test']
        })
        
        result = satp_testing.compute_main_axis_criteria(df)
        # ORTH = 1 - 2*|5.0/10 - 0.5| = 1 - 2*|0.5 - 0.5| = 1.0
        assert result['ORTH'].iloc[0] == pytest.approx(1.0)
        
        # Test with bias
        df['BIAS'] = [7.0]
        result = satp_testing.compute_main_axis_criteria(df)
        # ORTH = 1 - 2*|7.0/10 - 0.5| = 1 - 2*0.2 = 0.6
        assert result['ORTH'].iloc[0] == pytest.approx(0.6)
    
    def test_ncon_computation(self):
        """Test NCON (non-confusability) computation formula."""
        df = pd.DataFrame({
            'APPR': [5.0], 'UNDR': [5.0], 'ANTO': [5.0], 'BIAS': [5.0],
            'ASSOCCW': [5.0], 'IMPCCW': [4.0],
            'ASSOCW': [5.0], 'IMPCW': [6.0],
            'CANDIDATE': ['test']
        })
        
        result = satp_testing.compute_main_axis_criteria(df)
        # NCON = 1 - 0.5*(6.0/10 + 4.0/10) = 1 - 0.5*1.0 = 0.5
        assert result['NCON'].iloc[0] == pytest.approx(0.5)
    
    def test_ibal_computation(self):
        """Test IBAL (importance balance) computation formula."""
        df = pd.DataFrame({
            'APPR': [5.0], 'UNDR': [5.0], 'ANTO': [5.0], 'BIAS': [5.0],
            'ASSOCCW': [5.0], 'IMPCCW': [6.0],
            'ASSOCW': [5.0], 'IMPCW': [4.0],
            'CANDIDATE': ['test']
        })
        
        result = satp_testing.compute_main_axis_criteria(df)
        # IBAL = 1 - |6.0/10 - 4.0/10| = 1 - 0.2 = 0.8
        assert result['IBAL'].iloc[0] == pytest.approx(0.8)
    
    def test_preserves_other_columns(self):
        """Test that non-computed columns are preserved."""
        df = pd.DataFrame({
            'COUNTRY': ['SG', 'MY'],
            'APPR': [8.0, 9.0],
            'UNDR': [7.5, 8.5],
            'ANTO': [8.0, 7.0],
            'BIAS': [5.0, 4.5],
            'ASSOCCW': [6.0, 5.5],
            'IMPCCW': [4.0, 3.5],
            'ASSOCW': [7.0, 6.5],
            'IMPCW': [5.0, 4.5],
            'CANDIDATE': ['pleasant', 'pleasant']
        })
        
        result = satp_testing.compute_main_axis_criteria(df)
        assert 'COUNTRY' in result.columns
        assert result['COUNTRY'].tolist() == ['SG', 'MY']


class TestComputeDerivedAxisCriteria:
    """Test suite for compute_derived_axis_criteria function."""
    
    def test_basic_computation(self):
        """Test basic computation of derived axis criteria."""
        df = pd.DataFrame({
            'APPR': [8.0, 9.0],
            'UNDR': [7.5, 8.5],
            'ASSOCCW': [6.0, 5.5],
            'IMPCCW': [4.0, 3.5],
            'ASSOCW': [7.0, 6.5],
            'IMPCW': [5.0, 4.5],
            'CANDIDATE': ['vibrant', 'vibrant']
        })
        
        result = satp_testing.compute_derived_axis_criteria(df)
        
        # Check normalization
        assert result['APPR'].iloc[0] == 0.8
        assert result['UNDR'].iloc[0] == 0.75
        
        # Check computed criteria
        assert 'CLAR' in result.columns
        assert 'CONN' in result.columns
        assert 'IBAL' in result.columns
        
        # Should not have main-axis-only criteria
        assert 'ANTO' not in result.columns
        assert 'ORTH' not in result.columns
        assert 'NCON' not in result.columns
    
    def test_conn_computation(self):
        """Test CONN (connectedness) computation formula."""
        df = pd.DataFrame({
            'APPR': [5.0], 'UNDR': [5.0],
            'ASSOCCW': [5.0], 'IMPCCW': [4.0],
            'ASSOCW': [5.0], 'IMPCW': [6.0],
            'CANDIDATE': ['test']
        })
        
        result = satp_testing.compute_derived_axis_criteria(df)
        # CONN = 0.5*(6.0/10 + 4.0/10) = 0.5*1.0 = 0.5
        assert result['CONN'].iloc[0] == pytest.approx(0.5)
    
    def test_clar_computation_derived(self):
        """Test CLAR computation for derived axis (same as main axis)."""
        df = pd.DataFrame({
            'APPR': [5.0], 'UNDR': [5.0],
            'ASSOCCW': [4.0], 'IMPCCW': [5.0],
            'ASSOCW': [6.0], 'IMPCW': [5.0],
            'CANDIDATE': ['test']
        })
        
        result = satp_testing.compute_derived_axis_criteria(df)
        # CLAR = 1 - 0.5*(6.0/10) - 0.5*(4.0/10) = 1 - 0.3 - 0.2 = 0.5
        assert result['CLAR'].iloc[0] == pytest.approx(0.5)


class TestSummarizeMainAxis:
    """Test suite for summarize_main_axis function."""
    
    def test_summary_without_country(self):
        """Test summary aggregation without country grouping."""
        df = pd.DataFrame({
            'CANDIDATE': ['pleasant', 'pleasant', 'annoying', 'annoying'],
            'COUNTRY': ['SG', 'MY', 'SG', 'MY'],
            'APPR': [0.8, 0.85, 0.75, 0.8],
            'UNDR': [0.75, 0.8, 0.7, 0.75],
            'CLAR': [0.6, 0.65, 0.55, 0.6],
            'ANTO': [0.8, 0.85, 0.75, 0.8],
            'ORTH': [0.9, 0.95, 0.85, 0.9],
            'NCON': [0.7, 0.75, 0.65, 0.7],
            'IBAL': [0.85, 0.9, 0.8, 0.85]
        })
        
        result = satp_testing.summarize_main_axis(df, by_country=False)
        
        # Should have one row per candidate
        assert len(result) == 2
        assert set(result['CANDIDATE']) == {'pleasant', 'annoying'}
        
        # Check that means are computed correctly
        pleasant_row = result[result['CANDIDATE'] == 'pleasant'].iloc[0]
        assert pleasant_row['APPR'] == pytest.approx(0.825)
    
    def test_summary_with_country(self):
        """Test summary aggregation with country grouping."""
        df = pd.DataFrame({
            'CANDIDATE': ['pleasant', 'pleasant', 'pleasant', 'pleasant'],
            'COUNTRY': ['SG', 'SG', 'MY', 'MY'],
            'APPR': [0.8, 0.82, 0.85, 0.87],
            'UNDR': [0.75, 0.77, 0.8, 0.82],
            'CLAR': [0.6, 0.62, 0.65, 0.67],
            'ANTO': [0.8, 0.82, 0.85, 0.87],
            'ORTH': [0.9, 0.92, 0.95, 0.97],
            'NCON': [0.7, 0.72, 0.75, 0.77],
            'IBAL': [0.85, 0.87, 0.9, 0.92]
        })
        
        result = satp_testing.summarize_main_axis(df, by_country=True)
        
        # Should have one row per country-candidate combination
        assert len(result) == 2
        assert 'COUNTRY' in result.columns
        
        # Check means for SG
        sg_row = result[result['COUNTRY'] == 'SG'].iloc[0]
        assert sg_row['APPR'] == pytest.approx(0.81)


class TestSummarizeDerivedAxis:
    """Test suite for summarize_derived_axis function."""
    
    def test_summary_without_country(self):
        """Test summary aggregation without country grouping."""
        df = pd.DataFrame({
            'CANDIDATE': ['vibrant', 'vibrant', 'calm', 'calm'],
            'COUNTRY': ['SG', 'MY', 'SG', 'MY'],
            'APPR': [0.8, 0.85, 0.75, 0.8],
            'UNDR': [0.75, 0.8, 0.7, 0.75],
            'CLAR': [0.6, 0.65, 0.55, 0.6],
            'CONN': [0.7, 0.75, 0.65, 0.7],
            'IBAL': [0.85, 0.9, 0.8, 0.85]
        })
        
        result = satp_testing.summarize_derived_axis(df, by_country=False)
        
        # Should have one row per candidate
        assert len(result) == 2
        assert set(result['CANDIDATE']) == {'vibrant', 'calm'}
        
        # Should have CONN but not main-axis-only criteria
        assert 'CONN' in result.columns
        assert 'ANTO' not in result.columns
        assert 'ORTH' not in result.columns
        assert 'NCON' not in result.columns


class TestKruskalWallisTest:
    """Test suite for kruskal_wallis_test function."""
    
    def test_main_axis_test(self):
        """Test Kruskal-Wallis test for main axis."""
        np.random.seed(42)
        df = pd.DataFrame({
            'CANDIDATE': ['A'] * 10 + ['B'] * 10 + ['C'] * 10,
            'APPR': np.random.uniform(0.5, 1.0, 30),
            'UNDR': np.random.uniform(0.5, 1.0, 30),
            'CLAR': np.random.uniform(0.4, 0.9, 30),
            'ANTO': np.random.uniform(0.5, 1.0, 30),
            'ORTH': np.random.uniform(0.6, 1.0, 30),
            'NCON': np.random.uniform(0.4, 0.8, 30),
            'IBAL': np.random.uniform(0.7, 1.0, 30)
        })
        
        result = satp_testing.kruskal_wallis_test(df, axis_type="main")
        
        # Should have one row per criterion
        assert len(result) == 7
        
        # Check columns exist
        assert 'CRITERION' in result.columns
        assert 'statistic' in result.columns
        assert 'pvalue' in result.columns
        assert 'effect_size' in result.columns
        
        # All criteria should be present
        expected_criteria = ['APPR', 'UNDR', 'CLAR', 'ANTO', 'ORTH', 'NCON', 'IBAL']
        assert set(result['CRITERION']) == set(expected_criteria)
    
    def test_derived_axis_test(self):
        """Test Kruskal-Wallis test for derived axis."""
        np.random.seed(42)
        df = pd.DataFrame({
            'CANDIDATE': ['A'] * 10 + ['B'] * 10 + ['C'] * 10,
            'APPR': np.random.uniform(0.5, 1.0, 30),
            'UNDR': np.random.uniform(0.5, 1.0, 30),
            'CLAR': np.random.uniform(0.4, 0.9, 30),
            'CONN': np.random.uniform(0.5, 0.9, 30),
            'IBAL': np.random.uniform(0.7, 1.0, 30)
        })
        
        result = satp_testing.kruskal_wallis_test(df, axis_type="derived")
        
        # Should have one row per criterion (5 for derived axis)
        assert len(result) == 5
        
        # Derived axis criteria
        expected_criteria = ['APPR', 'UNDR', 'CLAR', 'CONN', 'IBAL']
        assert set(result['CRITERION']) == set(expected_criteria)
    
    def test_invalid_axis_type(self):
        """Test that invalid axis type raises ValueError."""
        df = pd.DataFrame({
            'CANDIDATE': ['A', 'B'],
            'APPR': [0.5, 0.6]
        })
        
        with pytest.raises(ValueError, match="axis_type must be either 'main' or 'derived'"):
            satp_testing.kruskal_wallis_test(df, axis_type="invalid")


class TestMannWhitneyTest:
    """Test suite for mann_whitney_test function."""
    
    def test_basic_test(self):
        """Test Mann-Whitney test with two countries."""
        np.random.seed(42)
        df = pd.DataFrame({
            'COUNTRY': ['SG'] * 10 + ['MY'] * 10,
            'CANDIDATE': ['pleasant'] * 20,
            'APPR': np.random.uniform(0.5, 1.0, 20),
            'UNDR': np.random.uniform(0.5, 1.0, 20)
        })
        
        result = satp_testing.mann_whitney_test(
            df, ['APPR', 'UNDR'], 'pleasant'
        )
        
        # Should have one row per criterion
        assert len(result) == 2
        
        # Check columns
        assert 'PAQ' in result.columns
        assert 'CRITERION' in result.columns
        assert 'CANDIDATE' in result.columns
        assert 'statistic' in result.columns
        assert 'pvalue' in result.columns
        assert 'adjusted_pvalue' in result.columns
        
        # Check PAQ attribute
        assert all(result['PAQ'] == 'pleasant')
    
    def test_multiple_candidates(self):
        """Test Mann-Whitney test with multiple candidates."""
        np.random.seed(42)
        df = pd.DataFrame({
            'COUNTRY': ['SG'] * 10 + ['MY'] * 10 + ['SG'] * 10 + ['MY'] * 10,
            'CANDIDATE': ['pleasant'] * 20 + ['annoying'] * 20,
            'APPR': np.random.uniform(0.5, 1.0, 40)
        })
        
        result = satp_testing.mann_whitney_test(df, ['APPR'], 'test')
        
        # Should have one row per candidate
        assert len(result) == 2
        assert set(result['CANDIDATE']) == {'pleasant', 'annoying'}
    
    def test_bonferroni_correction(self):
        """Test that Bonferroni correction is applied."""
        np.random.seed(42)
        df = pd.DataFrame({
            'COUNTRY': ['SG'] * 10 + ['MY'] * 10,
            'CANDIDATE': ['pleasant'] * 20,
            'APPR': np.random.uniform(0.5, 1.0, 20)
        })
        
        result = satp_testing.mann_whitney_test(df, ['APPR'], 'pleasant')
        
        # Adjusted p-value should be 2x original (or capped at 1.0)
        pvalue = result['pvalue'].iloc[0]
        adjusted = result['adjusted_pvalue'].iloc[0]
        assert adjusted == pytest.approx(min(pvalue * 2, 1.0))


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_main_axis_workflow(self):
        """Test complete workflow for main axis."""
        # Create mock survey data with multiple candidates
        np.random.seed(42)
        n_per_candidate = 5
        df = pd.DataFrame({
            'COUNTRY': ['SG', 'MY'] * n_per_candidate + ['SG', 'MY'] * n_per_candidate,
            'APPR': np.random.uniform(7, 10, n_per_candidate * 4),
            'UNDR': np.random.uniform(7, 10, n_per_candidate * 4),
            'ANTO': np.random.uniform(7, 10, n_per_candidate * 4),
            'BIAS': np.random.uniform(4, 6, n_per_candidate * 4),
            'ASSOCCW': np.random.uniform(3, 7, n_per_candidate * 4),
            'IMPCCW': np.random.uniform(3, 7, n_per_candidate * 4),
            'ASSOCW': np.random.uniform(3, 7, n_per_candidate * 4),
            'IMPCW': np.random.uniform(3, 7, n_per_candidate * 4),
            'CANDIDATE': ['pleasant'] * (n_per_candidate * 2) + ['annoying'] * (n_per_candidate * 2)
        })
        
        # Step 1: Compute criteria
        computed = satp_testing.compute_main_axis_criteria(df)
        assert 'CLAR' in computed.columns
        assert 'ORTH' in computed.columns
        
        # Step 2: Summarize
        summary = satp_testing.summarize_main_axis(computed, by_country=False)
        assert len(summary) == 2  # Two candidates
        
        # Step 3: Statistical test
        result = satp_testing.kruskal_wallis_test(computed, axis_type="main")
        assert len(result) == 7
    
    def test_derived_axis_workflow(self):
        """Test complete workflow for derived axis."""
        # Create mock survey data with multiple candidates
        np.random.seed(42)
        n_per_candidate = 5
        df = pd.DataFrame({
            'COUNTRY': ['SG', 'MY'] * n_per_candidate + ['SG', 'MY'] * n_per_candidate,
            'APPR': np.random.uniform(7, 10, n_per_candidate * 4),
            'UNDR': np.random.uniform(7, 10, n_per_candidate * 4),
            'ASSOCCW': np.random.uniform(3, 7, n_per_candidate * 4),
            'IMPCCW': np.random.uniform(3, 7, n_per_candidate * 4),
            'ASSOCW': np.random.uniform(3, 7, n_per_candidate * 4),
            'IMPCW': np.random.uniform(3, 7, n_per_candidate * 4),
            'CANDIDATE': ['vibrant'] * (n_per_candidate * 2) + ['calm'] * (n_per_candidate * 2)
        })
        
        # Step 1: Compute criteria
        computed = satp_testing.compute_derived_axis_criteria(df)
        assert 'CLAR' in computed.columns
        assert 'CONN' in computed.columns
        
        # Step 2: Summarize
        summary = satp_testing.summarize_derived_axis(computed, by_country=True)
        assert len(summary) == 4  # Two countries x two candidates
        
        # Step 3: Statistical test
        result = satp_testing.kruskal_wallis_test(computed, axis_type="derived")
        assert len(result) == 5
