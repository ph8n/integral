import unittest
import numpy as np
import pandas as pd
import pytest

# Try/Except import so this file doesn't crash the test discovery if cvxpy isn't installed
try:
    from mind.optimizer.solver import OptimizerEngine
    from mind.optimizer.objective import MinimumVariance, MeanVariance
    from mind.optimizer.risk_model import CovarianceEstimator
    from mind.optimizer.constraints import WeightConstraint
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        if not CVXPY_AVAILABLE:
            self.skipTest("cvxpy not installed")

        self.tickers = ["A", "B"]
        # Simple synthetic data
        self.prices = pd.DataFrame(
            {"A": [100, 101, 102, 103, 104, 105], "B": [100, 90, 110, 90, 110, 100]},
            index=pd.date_range("2023-01-01", periods=6),
        )

        self.risk_model = CovarianceEstimator()
        self.engine = OptimizerEngine()

    def test_covariance_calculation(self):
        cov = self.risk_model.calculate_sample_covariance(self.prices)
        self.assertEqual(cov.shape, (2, 2))
        self.assertGreater(cov.loc["B", "B"], cov.loc["A", "A"])

    def test_minimum_variance_allocation(self):
        cov = self.risk_model.calculate_sample_covariance(self.prices)
        mu = pd.Series([0.01, 0.0], index=self.tickers)

        weights = self.engine.optimize(self.tickers, mu, cov, MinimumVariance())

        self.assertAlmostEqual(weights.sum(), 1.0, places=4)
        self.assertTrue((weights >= 0).all())
        self.assertGreater(weights["A"], 0.8)

    def test_infeasible_constraints(self):
        """
        Test that the optimizer raises ValueError when constraints cannot be met.
        """
        cov = self.risk_model.calculate_sample_covariance(self.prices)
        mu = pd.Series([0.1, 0.1], index=self.tickers)

        # Constraint: Max weight for A is 0.2, Max weight for B is 0.2.
        # Sum must be 1.0. Impossible.

        con_a = WeightConstraint(max_weight=0.2)

        # Pass manual constraint for B to ensure conflict
        # Or use WeightConstraint globally if it applied to all assets?
        # WeightConstraint applies to *individual* assets.
        # So if we say max_weight=0.2, then A<=0.2 AND B<=0.2.
        # Sum <= 0.4. But Sum==1 required.

        # Note: WeightConstraint implementation applies:
        # weights >= min
        # weights <= max
        # This applies to the vector 'weights'. So ALL elements must be <= max.

        constraints = [WeightConstraint(max_weight=0.2)]

        with self.assertRaises(ValueError):
            self.engine.optimize(
                self.tickers, mu, cov, MinimumVariance(), risk_constraints=constraints
            )


if __name__ == "__main__":
    unittest.main()
