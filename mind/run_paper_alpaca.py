import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
import time

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Strategy Imports
from mind.data.lstm_alpha import LSTMAlphaModel
from mind.optimizer.solver import OptimizerEngine
from mind.optimizer.objective import MaximizeReturns
from mind.optimizer.risk_model import CovarianceEstimator
from mind.optimizer.monte_carlo import run_monte_carlo_simulation


class AlpacaStrategyExecutor:
    def __init__(self, api_key, api_secret, universe, lookback_days=365):
        self.trading_client = TradingClient(api_key, api_secret, paper=True)
        self.data_client = StockHistoricalDataClient(api_key, api_secret)
        self.universe = universe
        self.lookback_days = lookback_days

        # Initialize Strategy Components
        self.alpha_model = LSTMAlphaModel()  # Random weights if no path
        self.optimizer = OptimizerEngine()
        self.risk_model = CovarianceEstimator()
        self.objective = MaximizeReturns()

    def run_cycle(self):
        print(f"--- Running Strategy Cycle for {self.universe} ---")

        # 1. Fetch Data
        print("Fetching historical data...")
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=self.lookback_days)

        request_params = StockBarsRequest(
            symbol_or_symbols=self.universe,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt,
        )

        bars = self.data_client.get_stock_bars(request_params)
        df = bars.df

        if df.empty:
            print("No data found.")
            return

        # Reset index if it's MultiIndex (Symbol, Timestamp) -> want flattened for strategy?
        # Strategy expects: DataFrame with 'close', 'ticker' column or similar structure
        # Alpaca df index is (symbol, timestamp).
        # LSTMAlphaModel expects pivotable data or already containing 'ticker' column.

        # Transform for Strategy
        df = df.reset_index()
        # Standardize columns
        df.columns = [c.lower() for c in df.columns]
        # rename symbol -> ticker if needed
        if "symbol" in df.columns:
            df = df.rename(columns={"symbol": "ticker"})

        print(f"Data fetched: {len(df)} rows.")

        # 2. Alpha Prediction (LSTM)
        print("Predicting alpha (LSTM)...")
        # LSTM model expects 'timestamp', 'ticker', 'close'
        expected_returns = self.alpha_model.predict(df)
        print("Expected Returns:\n", expected_returns)

        # 3. Risk Model (Covariance)
        print("Calculating Risk...")
        prices_wide = df.pivot(index="timestamp", columns="ticker", values="close")
        prices_wide = prices_wide.ffill().dropna()  # Ensure clean data

        if len(prices_wide) < 30:
            print("Not enough history for covariance.")
            return

        covariance = self.risk_model.calculate_sample_covariance(prices_wide)

        # 4. Optimization
        print("Optimizing Portfolio...")
        # Filter universe to valid data
        valid_tickers = [
            t
            for t in self.universe
            if t in expected_returns.index and t in covariance.index
        ]

        if not valid_tickers:
            print("No valid tickers for optimization.")
            return

        optimal_weights = self.optimizer.optimize(
            tickers=valid_tickers,
            expected_returns=expected_returns,
            covariance_matrix=covariance,
            objective=self.objective,
        )
        print("Optimal Weights:\n", optimal_weights)

        # 5. Monte Carlo Check
        print("Running Monte Carlo Risk Check...")
        account = self.trading_client.get_account()
        equity = float(account.equity)

        risk_metrics = run_monte_carlo_simulation(
            weights=optimal_weights,
            mean_returns=expected_returns,
            covariance_matrix=covariance,
            initial_equity=equity,
            time_horizon_days=20,
            num_simulations=500,
        )
        print(f"Risk Metrics (20d): VaR 95% = ${risk_metrics['VaR_95_equity']:,.2f}")

        # 6. Execution
        self.execute_rebalance(optimal_weights, equity)

    def execute_rebalance(self, target_weights, equity):
        print("Executing Rebalance...")
        positions = self.trading_client.get_all_positions()
        current_positions = {p.symbol: int(p.qty) for p in positions}

        # Get current prices (using last close from data for simplicity, or fetch snapshot)
        # Ideally fetch snapshot
        # For now, assume last close is "close enough" for paper execution or fetch snapshot

        # We'll do a quick snapshot fetch for accuracy
        try:
            # get_stock_latest_quote is available in trading client? No, data client.
            # Or use get_stock_latest_trade
            from alpaca.data.requests import StockLatestTradeRequest

            trade_req = StockLatestTradeRequest(
                symbol_or_symbols=list(target_weights.index)
            )
            trades = self.data_client.get_stock_latest_trade(trade_req)
            current_prices = {t: trades[t].price for t in trades}
        except Exception as e:
            print(f"Could not fetch live prices, skipping execution: {e}")
            return

        for ticker, weight in target_weights.items():
            if ticker not in current_prices:
                continue

            price = current_prices[ticker]
            target_val = equity * weight
            target_qty = int(target_val / price)

            current_qty = current_positions.get(ticker, 0)
            diff = target_qty - current_qty

            if diff == 0:
                continue

            side = OrderSide.BUY if diff > 0 else OrderSide.SELL
            qty = abs(diff)

            print(f"Submitting {side} {qty} {ticker} @ ~{price}")

            try:
                market_order_data = MarketOrderRequest(
                    symbol=ticker, qty=qty, side=side, time_in_force=TimeInForce.DAY
                )
                self.trading_client.submit_order(order_data=market_order_data)
                time.sleep(0.5)  # Rate limit politeness
            except Exception as e:
                print(f"Order failed for {ticker}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, required=True)
    parser.add_argument("--secret", type=str, required=True)
    parser.add_argument("--tickers", type=str, default="SPY,TLT,AAPL,GOOG,NVDA")
    args = parser.parse_args()

    universe = [t.strip() for t in args.tickers.split(",")]

    strategy = AlpacaStrategyExecutor(args.key, args.secret, universe)
    strategy.run_cycle()
