import pytest
import pandas as pd
import numpy as np
from mind.data.alpha import MovingAverageCrossoverAlpha


def test_ma_crossover_alpha():
    # Create synthetic data
    # Ticker A: Price increasing (Fast > Slow) -> Buy Signal (Positive)
    # Ticker B: Price decreasing (Fast < Slow) -> Sell Signal (Negative)

    dates = pd.date_range(start="2023-01-01", periods=100)

    # Ticker A: Linear increase
    price_a = np.linspace(100, 200, 100)

    # Ticker B: Linear decrease
    price_b = np.linspace(200, 100, 100)

    data = pd.DataFrame({"timestamp": dates, "ticker": ["A"] * 100, "close": price_a})
    data_b = pd.DataFrame({"timestamp": dates, "ticker": ["B"] * 100, "close": price_b})

    # Combine
    market_history = pd.concat([data, data_b])

    # Run Alpha Model
    model = MovingAverageCrossoverAlpha(fast_window=10, slow_window=20)
    expected_returns = model.predict(market_history)

    # Check A (increasing) -> Fast MA > Slow MA -> Positive Return
    assert expected_returns["A"] > 0

    # Check B (decreasing) -> Fast MA < Slow MA -> Negative Return
    assert expected_returns["B"] < 0


def test_ma_crossover_alpha_wide_format():
    # Test with wide format (pivot already done or similar)
    dates = pd.date_range(start="2023-01-01", periods=100)
    price_a = np.linspace(100, 200, 100)

    market_history = pd.DataFrame({"A": price_a}, index=dates)

    # If input is already wide (no ticker column), the current implementation
    # checks for 'ticker' column. If not present, it treats columns as tickers?
    # Looking at implementation:
    # if 'ticker' in df.columns and 'close' in df.columns: pivot...
    # else: use as is.

    model = MovingAverageCrossoverAlpha(fast_window=10, slow_window=20)
    expected_returns = model.predict(market_history)

    assert expected_returns["A"] > 0
