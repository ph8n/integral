from abc import ABC, abstractmethod
import cvxpy as cp
import numpy as np


class ObjectiveFunction(ABC):
    """
    Abstract base class for optimization objectives.
    """

    @abstractmethod
    def formulate(
        self, weights: cp.Variable, expected_returns: np.ndarray, covariance: np.ndarray
    ) -> cp.Objective:
        """
        Constructs the cvxpy Objective object.
        """
        pass


class MinimumVariance(ObjectiveFunction):
    """
    Objective: Minimize Portfolio Variance (Risk).
    """

    def formulate(
        self, weights: cp.Variable, expected_returns: np.ndarray, covariance: np.ndarray
    ) -> cp.Minimize:
        # Min w^T * Sigma * w
        risk = cp.quad_form(weights, covariance)
        return cp.Minimize(risk)


class MeanVariance(ObjectiveFunction):
    """
    Objective: Maximize Utility = Expected Return - Risk Aversion * Variance.
    """

    def __init__(self, risk_aversion: float = 1.0):
        self.risk_aversion = risk_aversion

    def formulate(
        self, weights: cp.Variable, expected_returns: np.ndarray, covariance: np.ndarray
    ) -> cp.Minimize:
        # Maximize Return - Lambda * Risk
        # Equivalent to Minimize Lambda * Risk - Return for the solver

        # w^T * Sigma * w
        risk = cp.quad_form(weights, covariance)

        # mu^T * w
        ret = expected_returns @ weights

        return cp.Minimize(self.risk_aversion * risk - ret)
