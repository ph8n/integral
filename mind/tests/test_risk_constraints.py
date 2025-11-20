import pytest
import cvxpy as cp
import numpy as np
import pandas as pd
from mind.optimizer.constraints import (
    WeightConstraint,
    LeverageConstraint,
    SectorConstraint,
)


def test_weight_constraint():
    # Problem: Maximize sum(w) subject to w in [0.1, 0.4]
    # n = 3
    weights = cp.Variable(3)
    tickers = ["A", "B", "C"]

    constraint_gen = WeightConstraint(min_weight=0.1, max_weight=0.4)
    constraints = constraint_gen.apply(weights, tickers)
    constraints.append(cp.sum(weights) == 1.0)

    prob = cp.Problem(cp.Maximize(cp.sum(weights)), constraints)
    prob.solve()

    assert prob.status == cp.OPTIMAL
    w_val = weights.value
    assert np.all(w_val >= 0.0999)
    assert np.all(w_val <= 0.4001)
    assert np.isclose(np.sum(w_val), 1.0)


def test_leverage_constraint():
    # Problem: Minimize norm(w) subject to sum(w)=1 and leverage <= 1.5 (not really binding for long only)
    # Let's allow shorting to test leverage.
    weights = cp.Variable(2)
    tickers = ["A", "B"]

    # Allow w to be negative
    # Base constraints: sum(w) == 1
    base = [cp.sum(weights) == 1.0]

    # Leverage <= 1.0 (implies Long Only effectively if sum=1)
    # Actually, if sum=1, leverage=sum(|w|).
    # If w = [1.5, -0.5], sum=1, leverage=2.0.
    # Limit to 1.2
    lev_con = LeverageConstraint(limit=1.2)
    constraints = base + lev_con.apply(weights, tickers)

    # Objective: something that pushes for leverage.
    # Maximize w[0] - w[1].
    # If w[0]=1.5, w[1]=-0.5 -> Obj = 2.0. Leverage=2.0.
    # If constraint works, it should stop at leverage=1.2.
    # w[0] + w[1] = 1 => w[1] = 1 - w[0]
    # |w[0]| + |1 - w[0]| <= 1.2
    # For w[0] > 1, w[0] + (w[0]-1) <= 1.2 => 2w[0] <= 2.2 => w[0] <= 1.1

    prob = cp.Problem(cp.Maximize(weights[0] - weights[1]), constraints)
    prob.solve()

    assert prob.status == cp.OPTIMAL
    w_val = weights.value
    assert np.sum(np.abs(w_val)) <= 1.2001
    assert np.isclose(w_val[0], 1.1, atol=1e-3)


def test_sector_constraint():
    weights = cp.Variable(4)
    tickers = ["T1", "T2", "T3", "T4"]
    # Sectors: T1, T2 -> Tech; T3 -> Energy; T4 -> Tech
    sector_map = {"T1": "Tech", "T2": "Tech", "T3": "Energy", "T4": "Tech"}

    # Limit Tech to 0.5
    sector_con = SectorConstraint(sector_map, {"Tech": 0.5})
    constraints = sector_con.apply(weights, tickers)
    constraints.append(cp.sum(weights) == 1.0)
    constraints.append(weights >= 0)

    # Objective: Maximize sum of Tech weights
    # Tech indices: 0, 1, 3.
    # Maximize w[0]+w[1]+w[3]
    prob = cp.Problem(cp.Maximize(weights[0] + weights[1] + weights[3]), constraints)
    prob.solve()

    assert prob.status == cp.OPTIMAL
    w_val = weights.value
    tech_exposure = w_val[0] + w_val[1] + w_val[3]
    assert tech_exposure <= 0.5001
    assert np.isclose(tech_exposure, 0.5, atol=1e-3)


def test_optimizer_integration():
    # Verify we can pass these to OptimizerEngine
    from mind.optimizer.solver import OptimizerEngine
    from mind.optimizer.objective import ObjectiveFunction

    # Mock Objective
    class MockObjective(ObjectiveFunction):
        def formulate(self, weights, mu, sigma):
            # Maximize first asset
            return cp.Maximize(weights[0])

    engine = OptimizerEngine()
    tickers = ["A", "B"]
    mu = pd.Series([0.1, 0.05], index=tickers)
    sigma = pd.DataFrame(np.eye(2), index=tickers, columns=tickers)

    # Constraint: Max weight 0.8
    w_con = WeightConstraint(max_weight=0.8)

    weights = engine.optimize(
        tickers=tickers,
        expected_returns=mu,
        covariance_matrix=sigma,
        objective=MockObjective(),
        risk_constraints=[w_con],
    )

    # Maximize w[0] (A). Should be capped at 0.8.
    assert np.isclose(weights["A"], 0.8, atol=1e-3)
    assert np.isclose(weights["B"], 0.2, atol=1e-3)
