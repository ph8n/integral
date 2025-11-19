import cvxpy as cp
import numpy as np
import pandas as pd
from typing import List, Optional
from .objective import ObjectiveFunction


class OptimizerEngine:
    """
    Solves the optimization problem using cvxpy.
    """

    def optimize(
        self,
        tickers: List[str],
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        objective: ObjectiveFunction,
        constraints: Optional[List] = None,
    ) -> pd.Series:
        """
        Runs the optimization and returns the optimal weights.

        Args:
            tickers: List of ticker symbols in the universe.
            expected_returns: Series of expected returns for each ticker.
            covariance_matrix: DataFrame of covariance for the tickers.
            objective: The ObjectiveFunction strategy to use.
            constraints: Optional list of additional cvxpy constraints.

        Returns:
            pd.Series: Optimal weights indexed by ticker.
        """
        n_assets = len(tickers)

        # Create optimization variables
        weights = cp.Variable(n_assets)

        # Align input data
        # We assume the caller provides aligned data, but reindexing ensures safety
        mu = expected_returns.reindex(tickers).fillna(0.0).values
        sigma = (
            covariance_matrix.reindex(index=tickers, columns=tickers).fillna(0.0).values
        )

        # Ensure Covariance is PSD (Positive Semi-Definite)
        # Small numerical noise can make it non-PSD, usually fixable by symmetrizing
        # but relying on well-formed input for now.

        # Formulate Objective
        obj = objective.formulate(weights, mu, sigma)

        # Default Constraints:
        # 1. Fully Invested (sum(w) = 1)
        # 2. Long Only (w >= 0)
        # These can be overridden if we refactor constraints later, but this is a sensible default.
        base_constraints = [cp.sum(weights) == 1.0, weights >= 0]

        all_constraints = base_constraints
        if constraints:
            # If external constraints are provided, we assume they are ADDITIVE to base constraints
            # or the user handles the logic.
            # For 'add-risk-constraints', we will likely pass more here.
            all_constraints.extend(constraints)

        # Define Problem
        prob = cp.Problem(obj, all_constraints)

        # Solve
        # verbose=False to keep stdout clean
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except cp.SolverError as e:
            raise ValueError(f"Solver failed: {e}")

        # Check Status
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise ValueError(f"Optimization failed. Status: {prob.status}")

        # Return result
        # Use .value to get the numpy array from cvxpy variable
        optimal_weights = weights.value

        # Clean small numerical noise (e.g. 1e-10 -> 0)
        optimal_weights[np.abs(optimal_weights) < 1e-7] = 0.0

        # Re-normalize if we aggressively zeroed out significant weight (unlikely)
        # But mostly just return the series
        return pd.Series(optimal_weights, index=tickers)
