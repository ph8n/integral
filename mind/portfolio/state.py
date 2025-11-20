from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PortfolioState:
    """
    Represents the current state of the portfolio.
    """

    positions: Dict[str, float] = field(default_factory=dict)
    cash: float = 0.0

    def calculate_total_equity(
        self, prices: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate total equity.
        If prices are provided, marks positions to market.
        Otherwise, assumes this is just the sum of cash (useful if we tracked cost basis,
        but here positions are quantity, so we need prices to know value).
        """
        if prices is None:
            # Without prices, we can't value positions.
            # Return cash for now or raise error?
            # For simplicity, return cash, but caller should know better.
            return self.cash

        equity = self.cash
        for ticker, quantity in self.positions.items():
            price = prices.get(ticker, 0.0)
            equity += quantity * price
        return equity

    def to_dict(self) -> Dict:
        """
        Convert to dictionary format expected by other components (e.g. Composer).
        """
        return {"positions": self.positions, "cash": self.cash}
