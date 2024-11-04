# Writing a basic test suite for the 'metrics.py' file using the unittest framework

import unittest

import numpy as np
import pytest

from soundscapy.audio.metrics import _stat_calcs


class TestMetricsUnit(unittest.TestCase):
    def test_stat_calcs_avg(self):
        ts = np.array([1, 2, 3, 4, 5])
        res = {}
        updated_res = _stat_calcs("metric", ts, res, ["avg"])
        self.assertEqual(
            updated_res["metric_avg"], 3.0, "Average calculation is incorrect"
        )

    def test_stat_calcs_percentile(self):
        ts = np.array([1, 2, 3, 4, 5])
        res = {}
        updated_res = _stat_calcs("metric", ts, res, [50])
        self.assertEqual(
            updated_res["metric_50"], 3.0, "50th percentile calculation is incorrect"
        )

    def test_stat_calcs_max(self):
        ts = np.array([1, 2, 3, 4, 5])
        res = {}
        updated_res = _stat_calcs("metric", ts, res, ["max"])
        self.assertEqual(
            updated_res["metric_max"], 5, "Max value calculation is incorrect"
        )

    def test_stat_calcs_multiple_stats(self):
        ts = np.array([1, 2, 3, 4, 5])
        res = {}
        updated_res = _stat_calcs("metric", ts, res, [50, "avg", "max"])
        expected_res = {"metric_50": 3.0, "metric_avg": 3.0, "metric_max": 5}
        self.assertEqual(
            updated_res, expected_res, "Multiple statistics calculation is incorrect"
        )

    @pytest.mark.filterwarnings("ignore:Mean of empty slice")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide")
    def test_stat_calcs_empty_ts(self):
        ts = np.array([])
        res = {}
        updated_res = _stat_calcs("metric", ts, res, ["avg"])
        self.assertTrue(
            np.isnan(updated_res.get("metric_avg")),
            "Average for empty array should be np.nan",
        )
