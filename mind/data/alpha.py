import pandas as pd
from abc import ABC, abstractmethod


class AlphaModel(ABC):
    """
    Interface for generating expected returns (Alpha) from market data.
    """

    @abstractmethod
    def predict(self, market_history: pd.DataFrame) -> pd.Series:
        """
        Generate expected returns for the assets in the market history.

        Args:
            market_history: DataFrame containing historical market data (OHLCV).
                          Can be multi-index (Ticker, Date) or wide format depending on convention.
                          Assuming wide format of Close prices or structured DataFrame from loader.

        Returns:
            pd.Series: Expected returns vector indexed by ticker.
        """
        pass


class MovingAverageCrossoverAlpha(AlphaModel):
    """
    A simple example alpha model based on Moving Average Crossover.
    If Fast MA > Slow MA, expect positive return.
    """

    def __init__(self, fast_window=10, slow_window=50):
        self.fast_window = fast_window
        self.slow_window = slow_window

    def predict(self, market_history: pd.DataFrame) -> pd.Series:
        # basic implementation assuming market_history is a DataFrame of Close prices
        # or has a 'close' column.

        # If input is the detailed OHLVC from DB (Ticker, Timestamp, ...)
        # we need to pivot it first.

        df = market_history.copy()

        # Handle DuckDB/Loader format: columns [ticker, timestamp, close, ...]
        if "ticker" in df.columns and "close" in df.columns:
            df = df.pivot(index="timestamp", columns="ticker", values="close")

        # Calculate MAs
        fast_ma = df.rolling(window=self.fast_window).mean()
        slow_ma = df.rolling(window=self.slow_window).mean()

        # Signal: 1 if Fast > Slow, -1 if Fast < Slow (or 0)
        # For expected returns, we might scale this.
        # Let's just return a dummy expected return of 0.01 (1%) for Buy, -0.01 for Sell

        signals = (fast_ma.iloc[-1] > slow_ma.iloc[-1]).astype(float)
        signals = signals.replace(0, -1)  # simple long/short

        expected_returns = signals * 0.01

        return expected_returns
