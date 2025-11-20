import pytest
import numpy as np
from mind.portfolio.state import PortfolioState
from mind.portfolio.provider import MockPositionProvider


def test_portfolio_state():
    # Init with cash
    state = PortfolioState(cash=10000.0)
    assert state.calculate_total_equity() == 10000.0

    # Add positions
    state.positions = {"AAPL": 10, "MSFT": 5}

    # Without prices
    assert state.calculate_total_equity() == 10000.0

    # With prices
    prices = {"AAPL": 150.0, "MSFT": 300.0}
    # Equity = 10000 + (10*150) + (5*300) = 10000 + 1500 + 1500 = 13000
    assert state.calculate_total_equity(prices) == 13000.0

    # To Dict
    d = state.to_dict()
    assert d["cash"] == 10000.0
    assert d["positions"]["AAPL"] == 10


def test_mock_position_provider():
    provider = MockPositionProvider(initial_cash=5000.0, initial_positions={"GOOGL": 2})
    state = provider.get_state()

    assert state.cash == 5000.0
    assert state.positions["GOOGL"] == 2

    # Test mutation independence
    state.cash = 0.0
    state2 = provider.get_state()
    assert state2.cash == 5000.0  # Should not have changed

    # Test update helper
    provider.update_state({"GOOGL": 3}, 4000.0)
    state3 = provider.get_state()
    assert state3.cash == 4000.0
    assert state3.positions["GOOGL"] == 3
