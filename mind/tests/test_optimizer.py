import unittest
import numpy as np
import pandas as pd

# Try/Except import so this file doesn't crash the test discovery if cvxpy isn't installed
# but the tests will fail/error out which is correct for verification.
try:
    from mind.optimizer.solver import OptimizerEngine
    from mind.optimizer.objective import MinimumVariance, MeanVariance
    from mind.optimizer.risk_model import CovarianceEstimator

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        if not CVXPY_AVAILABLE:
            self.skipTest("cvxpy not installed")

        self.tickers = ["A", "B"]
        # Simple synthetic data
        # A: Steady growth
        # B: Volatile flat
        self.prices = pd.DataFrame(
            {"A": [100, 101, 102, 103, 104, 105], "B": [100, 90, 110, 90, 110, 100]},
            index=pd.date_range("2023-01-01", periods=6),
        )

        self.risk_model = CovarianceEstimator()
        self.engine = OptimizerEngine()

    def test_covariance_calculation(self):
        cov = self.risk_model.calculate_sample_covariance(self.prices)
        self.assertEqual(cov.shape, (2, 2))
        # B should have much higher variance than A
        self.assertGreater(cov.loc["B", "B"], cov.loc["A", "A"])

    def test_minimum_variance_allocation(self):
        """
        Min Variance should prefer the lower volatility asset (A).
        """
        cov = self.risk_model.calculate_sample_covariance(self.prices)
        mu = pd.Series([0.01, 0.0], index=self.tickers)  # Dummy expected returns

        weights = self.engine.optimize(self.tickers, mu, cov, MinimumVariance())

        # Constraints Check
        self.assertAlmostEqual(weights.sum(), 1.0, places=4)
        self.assertTrue((weights >= 0).all())

        # Logic Check: A has way lower variance, should have majority weight
        self.assertGreater(weights["A"], 0.8)

    def test_mean_variance_allocation(self):
        """
        Mean Variance should balance Return vs Risk.
        """
        cov = self.risk_model.calculate_sample_covariance(self.prices)

        # Case 1: A has return, B has none. A has lower risk.
        # Result: Should be 100% A.
        mu1 = pd.Series([0.1, 0.0], index=self.tickers)
        weights1 = self.engine.optimize(
            self.tickers, mu1, cov, MeanVariance(risk_aversion=1.0)
        )
        self.assertGreater(weights1["A"], 0.95)

        # Case 2: B has massive return to justify its risk.
        # Result: Should allocate some to B.
        mu2 = pd.Series([0.0, 0.5], index=self.tickers)
        weights2 = self.engine.optimize(
            self.tickers, mu2, cov, MeanVariance(risk_aversion=1.0)
        )
        self.assertGreater(weights2["B"], 0.1)


if __name__ == "__main__":
    unittest.main()
