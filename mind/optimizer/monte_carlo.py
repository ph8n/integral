import numpy as np
import pandas as pd
from typing import List, Dict, Optional


def run_monte_carlo_simulation(
    weights: pd.Series,
    mean_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    initial_equity: float = 100000.0,
    time_horizon_days: int = 252,
    num_simulations: int = 1000,
) -> Dict[str, float]:
    """
    Runs a Monte Carlo simulation to project portfolio performance and estimate risk.

    Args:
        weights: Portfolio weights (sum should ideally be 1.0).
        mean_returns: Expected daily returns for each asset.
        covariance_matrix: Covariance matrix of daily returns.
        initial_equity: Starting portfolio value.
        time_horizon_days: Number of days to simulate.
        num_simulations: Number of simulation paths to generate.

    Returns:
        Dict containing risk metrics like 'expected_final_equity', 'VaR_95', 'CVaR_95'.
    """
    # Align data
    tickers = weights.index.tolist()
    w = weights.values
    mu = mean_returns.reindex(tickers).fillna(0.0).values
    sigma = covariance_matrix.reindex(index=tickers, columns=tickers).fillna(0.0).values

    # Calculate portfolio mean and volatility
    # Portfolio expected return = w.T * mu
    port_mu = np.dot(w, mu)
    # Portfolio volatility = sqrt(w.T * Sigma * w)
    port_sigma = np.sqrt(np.dot(w.T, np.dot(sigma, w)))

    # Simulate paths
    # We assume geometric brownian motion (log-normal) or simple normal returns for the portfolio
    # Using simple normal approximation for portfolio returns: R_p ~ N(port_mu, port_sigma)

    # Generate random returns: shape (time_horizon, num_simulations)
    random_shocks = np.random.normal(0, 1, (time_horizon_days, num_simulations))

    # Daily portfolio returns
    daily_returns = port_mu + port_sigma * random_shocks

    # Cumulative returns (equity curve)
    # Start at initial_equity
    # equity_t = equity_{t-1} * (1 + r_t)
    cumulative_returns = np.cumprod(1 + daily_returns, axis=0)
    final_equity = initial_equity * cumulative_returns[-1, :]

    # Calculate metrics
    mean_final_equity = np.mean(final_equity)

    # Value at Risk (VaR) - 5th percentile (95% confidence)
    # This represents the equity value we expect to exceed 95% of the time.
    # Or the loss we expect to exceed only 5% of the time.
    # Here we return the Equity value at the 5th percentile.
    var_95_equity = np.percentile(final_equity, 5)

    # Conditional VaR (CVaR) - Mean of values below VaR
    cvar_95_equity = np.mean(final_equity[final_equity <= var_95_equity])

    return {
        "initial_equity": initial_equity,
        "expected_final_equity": mean_final_equity,
        "VaR_95_equity": var_95_equity,
        "CVaR_95_equity": cvar_95_equity,
        "sharpe_ratio_simulated": (np.mean(daily_returns) / np.std(daily_returns))
        * np.sqrt(252)
        if np.std(daily_returns) > 0
        else 0.0,
    }
