import unittest

import numpy as np
import pandas as pd
import seaborn.objects as so

from soundscapy.plotting.circumplex_plot import CircumplexPlot, CircumplexPlotParams


class TestCircumplexPlot(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.data = pd.DataFrame(
            {
                "ISOPleasant": np.random.uniform(-1, 1, 50),
                "ISOEventful": np.random.uniform(-1, 1, 50),
                "Category": np.random.choice(["A", "B"], 50),
            }
        )
        self.plot = CircumplexPlot(self.data)

    def test_init(self):
        self.assertIsInstance(self.plot, CircumplexPlot)
        self.assertEqual(self.plot.data.shape, (50, 3))

    def test_scatter(self):
        result = self.plot.scatter()
        self.assertIsInstance(result, CircumplexPlot)
        self.assertIsInstance(self.plot._plot, so.Plot)

    def test_scatter_with_hue(self):
        params = CircumplexPlotParams(hue="Category")
        plot = CircumplexPlot(self.data, params)
        result = plot.scatter()
        self.assertIsInstance(result, CircumplexPlot)
        self.assertIsInstance(plot._plot, so.Plot)
        # Check if color aesthetic is set when hue is provided
        self.assertIn("color", plot._plot.layers[-1].aesthetics)

    def test_show_without_plot(self):
        with self.assertRaises(ValueError):
            self.plot.show()

    def test_show_with_plot(self):
        self.plot.scatter()
        try:
            self.plot.show()
        except Exception as e:
            self.fail(f"show() raised {type(e).__name__} unexpectedly!")


if __name__ == "__main__":
    unittest.main()
