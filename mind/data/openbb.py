import pandas as pd
import os
from typing import List, Optional


class OpenBBDataLoader:
    """
    Fetches historical data using OpenBB or yfinance directly.
    """

    def __init__(self, provider: str = "yfinance"):
        self.provider = provider

    def fetch_history(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical daily bars for the given tickers.
        """
        try:
            from openbb import obb

            use_openbb = True
        except ImportError:
            use_openbb = False

        if not tickers:
            return pd.DataFrame()

        # yfinance supports multiple tickers space-separated
        symbols_str = " ".join(tickers)

        try:
            if use_openbb:
                df = obb.equity.price.historical(
                    symbol=symbols_str.replace(" ", ","),
                    start_date=start_date,
                    end_date=end_date,
                    provider=self.provider,
                    interval="1d",
                ).to_df()
                return df
            else:
                # Fallback to yfinance directly
                import yfinance as yf

                print(f"OpenBB not found, using yfinance directly for {symbols_str}...")

                # Download data
                # If single ticker, don't use group_by to get flat DF
                if len(tickers) == 1:
                    data = yf.download(
                        tickers[0],
                        start=start_date,
                        end=end_date,
                        auto_adjust=False,
                        progress=False,
                    )
                else:
                    data = yf.download(
                        tickers,
                        start=start_date,
                        end=end_date,
                        group_by="ticker",
                        auto_adjust=False,
                        progress=False,
                    )

                if data.empty:
                    return pd.DataFrame()

                if len(tickers) == 1:
                    df = data
                    # Handle potential MultiIndex columns if yfinance returns them anyway
                    if isinstance(df.columns, pd.MultiIndex):
                        # Flatten
                        # e.g. ('Close', 'AAPL') -> 'Close'
                        # Or ('Price', 'Close')?
                        # Just keep the level that has 'Close'
                        new_cols = []
                        for col in df.columns:
                            if isinstance(col, tuple):
                                # Find the OHLCV part
                                found = False
                                for part in col:
                                    if str(part).lower() in [
                                        "open",
                                        "high",
                                        "low",
                                        "close",
                                        "volume",
                                        "adj close",
                                    ]:
                                        new_cols.append(str(part))
                                        found = True
                                        break
                                if not found:
                                    new_cols.append("_".join(map(str, col)))
                            else:
                                new_cols.append(str(col))
                        df.columns = new_cols

                    df.columns = [str(c).lower() for c in df.columns]
                    return df
                else:
                    return data

        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def save_to_parquet(self, df: pd.DataFrame, path: str):
        """
        Save the dataframe to parquet for ingestion.
        Expects df to have standard OHLVC columns.
        """
        # Standardize columns to lower case
        df.columns = [str(c).lower() for c in df.columns]

        # Ensure standard Nautilus columns
        required = ["open", "high", "low", "close", "volume"]
        # yfinance might return 'adj close', we can use that as close if close is missing?
        # Or just keep what we have.

        # Rename 'adj close' to 'close' if preferred?
        # Let's stick to 'close' (raw) or 'adj close' if we want adjusted.
        # usually we want adjusted for backtest.
        if "adj close" in df.columns:
            df["close"] = df["adj close"]  # Use adjusted close

        for col in required:
            if col not in df.columns:
                # Maybe it's case sensitive or named differently
                pass

        # Nautilus typically expects 'timestamp' index or column in nanoseconds
        if not isinstance(df.index, pd.DatetimeIndex) and "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"])
            df.set_index("timestamp", inplace=True)

        # Ensure index is named 'timestamp'
        df.index.name = "timestamp"

        df.to_parquet(path)
