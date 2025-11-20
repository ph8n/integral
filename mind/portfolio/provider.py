from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from .state import PortfolioState


class PositionProvider(ABC):
    """
    Interface for fetching the current portfolio state from a source of truth (broker).
    """

    @abstractmethod
    def get_state(self) -> PortfolioState:
        """
        Fetch the current positions and cash.
        """
        pass


class MockPositionProvider(PositionProvider):
    """
    A mock provider for testing or simulation.
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        initial_positions: Optional[Dict[str, float]] = None,
    ):
        self._state = PortfolioState(
            positions=initial_positions or {}, cash=initial_cash
        )

    def get_state(self) -> PortfolioState:
        # Return a copy to prevent mutation of internal state by caller if they modify it
        return PortfolioState(
            positions=self._state.positions.copy(), cash=self._state.cash
        )

    def update_state(self, new_positions: Dict[str, float], new_cash: float):
        """
        Helper for testing to update the "broker" state.
        """
        self._state.positions = new_positions
        self._state.cash = new_cash
