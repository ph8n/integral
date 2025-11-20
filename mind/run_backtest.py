import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter(action="ignore", category=FutureWarning)

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.data.openbb import OpenBBDataLoader

# Try importing strategy components, with mock fallback
try:
    from mind.strategy import (
        IntegralStrategy,
        IntegralStrategyConfig,
        RLIntegralStrategy,
        RLIntegralStrategyConfig,
    )

    # Also need nautilus mocks if real nautilus is missing
    try:
        import nautilus_trader

        USE_MOCK = False
    except ImportError:
        USE_MOCK = True
        from mind.nautilus_mock import (
            Strategy,
            Bar,
            StrategyConfig,
            OrderSide,
            TimeInForce,
            Instrument,
            InstrumentId,
            Portfolio,
            Position,
        )
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


def plot_equity_curve(equity_df, title, output_file):
    """
    Plots the equity curve and saves it to a file.
    """
    plt.figure(figsize=(12, 6))

    # Check if index is datetime
    if not isinstance(equity_df.index, pd.DatetimeIndex):
        try:
            equity_df.index = pd.to_datetime(equity_df.index)
        except:
            pass

    plt.plot(equity_df.index, equity_df["equity"], label="Portfolio Equity")
    plt.title(f"Equity Curve: {title}")
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True)

    # Format y-axis as currency
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(["${:,.0f}".format(x) for x in current_values])

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Equity curve plot saved to {output_file}")
    plt.close()


def parse_portfolio_csv(file_path):
    """
    Parses the portfolio CSV to extract tickers and calculate weights.
    Expected format: Fidelity export or similar.
    """
    print(f"Reading portfolio from {file_path}...")

    # Try to find the header row
    header_row = None
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            # Check for common header columns
            if '"Symbol"' in line or "Symbol" in line:
                header_row = i
                break
            if i > 20:  # Give up after 20 lines
                break

    if header_row is None:
        # Fallback: Try reading without skip if it looks like a clean CSV
        header_row = 0

    try:
        df = pd.read_csv(file_path, skiprows=header_row)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return [], {}

    # Clean up column names (strip whitespace/quotes)
    df.columns = [c.strip().replace('"', "") for c in df.columns]

    if "Symbol" not in df.columns:
        print("CSV does not contain 'Symbol' column.")
        return [], {}

    # Filter out summary rows
    df = df[~df["Symbol"].isin(["Cash & Cash Investments", "Account Total"])]
    df = df[df["Symbol"].notna()]

    tickers = df["Symbol"].tolist()

    # Calculate weights based on Market Value if available
    mkt_val_col = next((c for c in df.columns if "Mkt Val" in c), None)

    weights = {}
    if mkt_val_col:
        # Remove '$', ',' and convert to float
        def clean_currency(x):
            if isinstance(x, str):
                return float(
                    x.replace("$", "").replace(",", "").replace("%", "").strip()
                )
            try:
                return float(x)
            except:
                return 0.0

        try:
            df["weight_val"] = df[mkt_val_col].apply(clean_currency)
            total_value = df["weight_val"].sum()
            if total_value > 0:
                for idx, row in df.iterrows():
                    sym = row["Symbol"]
                    w = row["weight_val"] / total_value
                    weights[sym] = w
            else:
                print("Total market value is 0, using equal weights.")
                weights = {t: 1.0 / len(tickers) for t in tickers}
        except Exception as e:
            print(f"Warning: Could not calculate weights from {mkt_val_col}: {e}")
            weights = {t: 1.0 / len(tickers) for t in tickers}
    else:
        print("Warning: 'Mkt Val' column not found. Using equal weights.")
        weights = {t: 1.0 / len(tickers) for t in tickers}

    return tickers, weights


def run_backtest(args):
    print(f"--- Starting Backtest: {args.name} ---")

    # 1. Configuration
    start_date = args.start
    end_date = args.end

    # Portfolio Setup
    if args.csv:
        csv_path = args.csv
        if not os.path.exists(csv_path):
            # Try relative to project root
            root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            possible_path = os.path.join(root_path, csv_path)
            if os.path.exists(possible_path):
                csv_path = possible_path
            else:
                print(f"Error: CSV file not found at {csv_path}")
                return

        tickers, target_weights = parse_portfolio_csv(csv_path)
        if not tickers:
            print("No tickers found in CSV. Exiting.")
            return

        print(f"Loaded {len(tickers)} tickers from {os.path.basename(csv_path)}")
    else:
        # Default
        tickers = ["SPY", "TLT"]
        target_weights = {"SPY": 0.6, "TLT": 0.4}
        print("Using default portfolio (SPY/TLT)")

    # 2. Data Fetching
    loader = OpenBBDataLoader()
    # Data dir relative to this script location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "mind/data/catalog")
    os.makedirs(data_dir, exist_ok=True)

    print(f"Fetching data for {len(tickers)} tickers...")
    valid_tickers = []

    for ticker in tickers:
        try:
            # Normalize ticker string
            ticker = str(ticker).strip().upper()
            if not ticker:
                continue

            # Fetch history
            df = loader.fetch_history([ticker], start_date, end_date)

            if df is None or df.empty:
                print(f"Warning: No data for {ticker}")
                continue

            # Save to Nautilus-compatible Parquet
            filename = f"{ticker}.SIM-1-DAY-MID-EXT.parquet"
            path = os.path.join(data_dir, filename)
            loader.save_to_parquet(df, path)
            valid_tickers.append(ticker)
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")

    tickers = valid_tickers
    if not tickers:
        print("No valid data found for any ticker. Exiting.")
        return

    # Re-normalize weights for valid tickers
    total_weight = sum(target_weights.get(t, 0) for t in tickers)
    if total_weight > 0:
        target_weights = {t: target_weights.get(t, 0) / total_weight for t in tickers}
    else:
        target_weights = {t: 1.0 / len(tickers) for t in tickers}

    # 3. Run Backtest (Vector or RL Loop)
    if args.rl:
        print("Performing RL Strategy Backtest (Event Loop)...")

        # Load all data into memory
        data_map = {}
        all_dates = set()

        for ticker in tickers:
            path = os.path.join(data_dir, f"{ticker}.SIM-1-DAY-MID-EXT.parquet")
            if os.path.exists(path):
                df = pd.read_parquet(path)
                # Ensure timestamp index
                if "timestamp" in df.columns and not isinstance(
                    df.index, pd.DatetimeIndex
                ):
                    df.set_index("timestamp", inplace=True)

                data_map[ticker] = df
                all_dates.update(df.index)

        sorted_dates = sorted(list(all_dates))
        print(f"Loaded data for {len(data_map)} tickers over {len(sorted_dates)} days.")

        # Setup Strategy
        config = RLIntegralStrategyConfig(
            instrument_ids=[f"{t}.SIM" for t in tickers],
            rebalance_interval_bars=20,
            lookback_window=60,
        )

        strategy = RLIntegralStrategy(config)
        # Inject mock components if needed
        if USE_MOCK:
            strategy.order_factory.orders = []  # Reset orders
            strategy.portfolio.account_obj.equity_total = args.capital

        # Event Loop
        equity_curve = []

        # Initialize cache with initial prices (0)
        for ticker in tickers:
            inst_id = f"{ticker}.SIM"
            if USE_MOCK:
                # Mock cache initialization
                pass

        current_equity = args.capital
        holdings = {f"{t}.SIM": 0 for t in tickers}
        cash = args.capital

        print("Running simulation...")
        for date in sorted_dates:
            # Update market data (cache)
            current_prices = {}
            for ticker, df in data_map.items():
                if date in df.index:
                    row = df.loc[date]
                    price = row["close"]
                    inst_id = f"{ticker}.SIM"
                    current_prices[inst_id] = price

                    # Update Mock Cache
                    if USE_MOCK:
                        bar = Bar(
                            instrument_id=InstrumentId(inst_id),
                            close=price,
                            ts_event=date,
                        )
                        strategy.cache.bars[inst_id] = bar

                        # Notify Strategy
                        strategy.on_bar(bar)

            # Process Orders from Strategy
            if USE_MOCK:
                new_orders = strategy.orders
                strategy.orders = []  # Clear processed orders

                for order in new_orders:
                    inst_id = str(order["instrument_id"])
                    qty = int(order["quantity"])
                    side = order["side"]
                    price = current_prices.get(inst_id, 0)

                    if price > 0:
                        cost = qty * price
                        if side == OrderSide.BUY:
                            if cash >= cost:
                                cash -= cost
                                holdings[inst_id] = holdings.get(inst_id, 0) + qty
                        elif side == OrderSide.SELL:
                            current_holdings = holdings.get(inst_id, 0)
                            if current_holdings >= qty:
                                cash += cost
                                holdings[inst_id] -= qty

            # Update Portfolio Value for next step
            portfolio_value = cash
            for inst_id, qty in holdings.items():
                price = current_prices.get(inst_id, 0)
                portfolio_value += qty * price

            if USE_MOCK:
                strategy.portfolio.account_obj.equity_total = portfolio_value
                # Update positions in portfolio mock
                for inst_id, qty in holdings.items():
                    strategy.portfolio.positions[inst_id] = Position(qty)

            equity_curve.append({"date": date, "equity": portfolio_value})

        # Results
        equity_df = pd.DataFrame(equity_curve).set_index("date")
        final_equity = equity_df.iloc[-1]["equity"]
        total_return = (final_equity / args.capital) - 1

        print(f"\n--- RL Backtest Results: {args.name} ---")
        print(f"Initial Equity: ${args.capital:,.2f}")
        print(f"Final Equity:   ${final_equity:,.2f}")
        print(f"Total Return:   {total_return:.2%}")

        # Plotting
        plot_equity_curve(equity_df, args.name, "backtest_results.png")

    else:
        # Vector Backtest (Static)
        print("Performing Vector Backtest...")

        prices = pd.DataFrame()
        for ticker in tickers:
            path = os.path.join(data_dir, f"{ticker}.SIM-1-DAY-MID-EXT.parquet")
            if os.path.exists(path):
                df = pd.read_parquet(path)
                if "close" in df.columns:
                    prices[ticker] = df["close"]

        if prices.empty:
            print("No pricing data found for vector backtest.")
        else:
            # Fill NaNs
            prices.fillna(method="ffill", inplace=True)
            prices.dropna(inplace=True)

            if prices.empty:
                print("Prices empty after dropping NaNs (dates might not overlap).")
                return

            # Calculate Returns
            returns = prices.pct_change().dropna()

            # Portfolio Returns
            valid_cols = [c for c in returns.columns if c in target_weights]
            if not valid_cols:
                print("No matching columns for weights.")
                return

            returns = returns[valid_cols]
            weights_series = pd.Series({k: target_weights[k] for k in valid_cols})

            if weights_series.sum() > 0:
                weights_series = weights_series / weights_series.sum()

            portfolio_daily_ret = (returns * weights_series).sum(axis=1)

            # Equity Curve
            initial_capital = args.capital
            equity_curve = (1 + portfolio_daily_ret).cumprod() * initial_capital

            # Metrics
            total_return = (equity_curve.iloc[-1] / initial_capital) - 1
            sharpe = 0.0
            if portfolio_daily_ret.std() != 0:
                sharpe = (portfolio_daily_ret.mean() / portfolio_daily_ret.std()) * (
                    252**0.5
                )

            # Max Drawdown
            rolling_max = equity_curve.cummax()
            drawdown = (equity_curve - rolling_max) / rolling_max
            max_dd = drawdown.min()

            print(f"\n--- Backtest Results: {args.name} ---")
            print(f"Start Date: {pd.to_datetime(prices.index[0]).date()}")
            print(f"End Date:   {pd.to_datetime(prices.index[-1]).date()}")
            print(f"Initial Equity: ${initial_capital:,.2f}")
            print(f"Final Equity:   ${equity_curve.iloc[-1]:,.2f}")
            print(f"Total Return:   {total_return:.2%}")
            print(f"Sharpe Ratio:   {sharpe:.2f}")
            print(f"Max Drawdown:   {max_dd:.2%}")

            # Plotting
            equity_df = equity_curve.to_frame(name="equity")
            plot_equity_curve(equity_df, args.name, "backtest_results.png")

    print("Backtest completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Integral Backtest")
    parser.add_argument(
        "--name", type=str, default="Custom Portfolio", help="Name of the backtest"
    )
    parser.add_argument(
        "--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, default="2025-01-01", help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to portfolio CSV file (default: auto-detect in root)",
    )
    parser.add_argument(
        "--capital", type=float, default=100000.0, help="Initial capital"
    )
    parser.add_argument(
        "--nautilus", action="store_true", help="Try to configure Nautilus node"
    )
    parser.add_argument(
        "--rl", action="store_true", help="Run RL Strategy (Simulated Event Loop)"
    )

    args = parser.parse_args()

    # Auto-detect CSV in root if not provided
    if not args.csv:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = glob.glob(os.path.join(root_dir, "*.csv"))
        preferred = [c for c in candidates if "Positions" in c or "Portfolio" in c]
        if preferred:
            candidates = preferred

        if candidates:
            candidates.sort(key=os.path.getmtime, reverse=True)
            detected = candidates[0]
            print(f"No CSV specified. Auto-detected: {os.path.basename(detected)}")
            args.csv = detected
        else:
            print("No CSV found in project root. Using default SPY/TLT portfolio.")

    run_backtest(args)
