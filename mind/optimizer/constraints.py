import cvxpy as cp
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Union


class Constraint(ABC):
    """
    Abstract Base Class for Portfolio Constraints.
    """

    @abstractmethod
    def apply(self, weights: cp.Variable, tickers: List[str]) -> List[cp.Constraint]:
        """
        Generates the cvxpy constraint(s).

        Args:
            weights: The cvxpy variable representing portfolio weights.
            tickers: The list of ticker symbols corresponding to the weights indices.

        Returns:
            A list of cvxpy constraint objects.
        """
        pass


class WeightConstraint(Constraint):
    """
    Constrains individual asset weights to be within [min_weight, max_weight].
    """

    def __init__(self, min_weight: float = 0.0, max_weight: float = 1.0):
        self.min_weight = min_weight
        self.max_weight = max_weight

    def apply(self, weights: cp.Variable, tickers: List[str]) -> List[cp.Constraint]:
        constraints = [weights >= self.min_weight, weights <= self.max_weight]
        return constraints


class LeverageConstraint(Constraint):
    """
    Constrains the Gross Leverage (sum of absolute weights) to be <= limit.
    Typically for Long-Only portfolios, this is sum(w) <= limit.
    For Long-Short, it is sum(abs(w)) <= limit.

    Note: Current default optimizer enforces sum(w)=1 and w>=0 (Long Only).
    This constraint is useful if we relax the default 'fully invested' constraint
    or allow shorting in the future.
    """

    def __init__(self, limit: float = 1.0):
        self.limit = limit

    def apply(self, weights: cp.Variable, tickers: List[str]) -> List[cp.Constraint]:
        # For general case (long/short), use L1 norm
        return [cp.norm(weights, 1) <= self.limit]


class SectorConstraint(Constraint):
    """
    Constrains the sum of weights for a group of assets (sector) to be <= limit.
    """

    def __init__(self, sector_mapping: Dict[str, str], sector_limits: Dict[str, float]):
        """
        Args:
            sector_mapping: Dict mapping ticker -> sector name.
            sector_limits: Dict mapping sector name -> max weight (float).
        """
        self.sector_mapping = sector_mapping
        self.sector_limits = sector_limits

    def apply(self, weights: cp.Variable, tickers: List[str]) -> List[cp.Constraint]:
        constraints = []

        # Pre-compute indices for each sector
        sector_indices = {}
        for i, ticker in enumerate(tickers):
            sector = self.sector_mapping.get(ticker)
            if sector:
                if sector not in sector_indices:
                    sector_indices[sector] = []
                sector_indices[sector].append(i)

        # Create constraint for each limited sector
        for sector, limit in self.sector_limits.items():
            indices = sector_indices.get(sector, [])
            if indices:
                # Sum of weights for this sector <= limit
                # We use cp.sum(weights[indices])
                # Note: weights is a 1D vector (n_assets,)
                # weights[indices] selects the subset
                constraints.append(cp.sum(weights[indices]) <= limit)

        return constraints
