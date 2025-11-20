import asyncio
import sys
import os
from decimal import Decimal

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.config import get_nautilus_config
from mind.strategy import IntegralStrategy, IntegralStrategyConfig
# from nautilus_trader.model.identifiers import Venue, InstrumentId
# from nautilus_trader.model.currencies import USD


def main():
    mode = os.environ.get("INTEGRAL_MODE", "backtest")  # "backtest" or "live"
    print(f"Starting Integral in {mode} mode...")

    try:
        from nautilus_trader.live.node import TradingNode
        from nautilus_trader.model.identifiers import Venue, InstrumentId
        from nautilus_trader.model.currencies import USD
    except ImportError:
        print("Nautilus Trader not installed. Please install it to run the strategy.")
        return

    # 1. Configure Node
    config = get_nautilus_config(mode=mode)
    node = TradingNode(config=config)

    # 2. Configure Strategy
    # Define instrument (e.g., SPY on Alpaca)
    venue = Venue("ALPACA")
    instrument_id = InstrumentId.from_str("SPY.ALPACA")

    strategy_config = IntegralStrategyConfig(
        instrument_id=str(instrument_id),
        bar_type="SPY.ALPACA-1-DAY-MID-EXT",  # Example bar type
    )

    strategy = IntegralStrategy(config=strategy_config)

    # 3. Run Node
    node.trader.add_strategy(strategy)

    if mode == "backtest":
        # For backtest, we need to load data into the catalog first
        print("Loading historical data...")
        # Using our OpenBB loader to fetch and save to parquet, then node.catalog.write_data(...)
        # For this script, we assume data is already loaded or we load it here.
        pass

    print("Running Node...")
    # node.run() is blocking or async depending on implementation
    # node.run()


if __name__ == "__main__":
    main()
