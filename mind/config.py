import os
from nautilus_trader.config import TradingNodeConfig, NautilusKernelConfig
from nautilus_trader.live.config import LiveExecEngineConfig


def get_nautilus_config(mode: str = "backtest") -> TradingNodeConfig:
    """
    Returns the configuration for the Nautilus Trading Node.
    """

    if mode == "live":
        try:
            from nautilus_trader.adapters.alpaca.config import (
                AlpacaDataClientConfig,
                AlpacaExecClientConfig,
            )
        except ImportError:
            raise ImportError(
                "Nautilus Alpaca adapter not installed. Install nautilus_trader[alpaca]."
            )

        # Alpaca Credentials
        api_key = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")
        base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        if not api_key or not secret_key:
            raise ValueError("Alpaca credentials not found in environment variables.")

        # Execution Client
        alpaca_exec = AlpacaExecClientConfig(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url,
            account_id="PAPER",  # Or specific ID
        )

        # Data Client (if using Alpaca for live data)
        alpaca_data = AlpacaDataClientConfig(
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url,
        )

        return TradingNodeConfig(
            trader_id="integral-trader",
            exec_clients={"ALPACA": alpaca_exec},
            data_clients={"ALPACA": alpaca_data},
            timeout_connection=30.0,
        )

    else:
        # Backtest Config
        return TradingNodeConfig(
            trader_id="integral-backtester",
            # Backtest engine config is usually default or configured via backtest runner
        )
