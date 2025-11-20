try:
    from nautilus_trader.trading.strategy import Strategy
    from nautilus_trader.model.data import Bar
    from nautilus_trader.config import StrategyConfig
    from nautilus_trader.model.enums import OrderSide, TimeInForce
    from nautilus_trader.model.instruments import Instrument
    from nautilus_trader.model.identifiers import InstrumentId
except ImportError:
    # Fallback to mock objects if Nautilus is not installed
    from mind.nautilus_mock import (
        Strategy,
        Bar,
        StrategyConfig,
        OrderSide,
        TimeInForce,
        Instrument,
        InstrumentId,
    )

from decimal import Decimal
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Imports for RL Strategy
from mind.data.lstm_alpha import LSTMAlphaModel
from mind.optimizer.solver import OptimizerEngine
from mind.optimizer.objective import MaximizeReturns
from mind.optimizer.risk_model import CovarianceEstimator
from mind.optimizer.monte_carlo import run_monte_carlo_simulation


class IntegralStrategyConfig(StrategyConfig):
    instrument_ids: list[str]
    target_weights: Dict[str, float]
    rebalance_interval_bars: int = 20  # Approx monthly if daily bars


class IntegralStrategy(Strategy):
    def __init__(self, config: IntegralStrategyConfig):
        super().__init__(config)
        self.instrument_ids = [InstrumentId.from_str(i) for i in config.instrument_ids]
        self.target_weights = config.target_weights
        self.bar_count = 0

    def on_start(self):
        self.log.info("Strategy started.")
        for instrument_id in self.instrument_ids:
            self.subscribe_bars(instrument_id)

    def on_bar(self, bar: Bar):
        self.bar_count += 1

        # Simple periodic rebalancing
        if self.bar_count % self.config.rebalance_interval_bars != 0:
            return

        self.log.info(f"Rebalancing at bar {self.bar_count} ({bar.close})")
        self._rebalance()

    def _rebalance(self):
        total_equity = self.portfolio.account(self.venue_id).equity_total
        if total_equity <= 0:
            return

        for instrument_id_str, weight in self.target_weights.items():
            instrument_id = InstrumentId.from_str(instrument_id_str)
            instrument = self.instrument(instrument_id)

            if not instrument:
                self.log.error(f"Instrument {instrument_id} not found.")
                continue

            # Calculate target value and quantity
            target_value = total_equity * Decimal(str(weight))
            last_price = self.cache.bar(instrument_id).close

            if last_price <= 0:
                continue

            target_qty = int(target_value / last_price)
            current_qty = (
                self.portfolio.position(instrument_id).quantity
                if self.portfolio.position(instrument_id)
                else 0
            )

            diff = target_qty - current_qty

            if diff == 0:
                continue

            side = OrderSide.BUY if diff > 0 else OrderSide.SELL
            qty = abs(diff)

            order = self.order_factory.market(
                instrument_id=instrument_id,
                order_side=side,
                quantity=self.instrument(instrument_id).make_qty(qty),
                time_in_force=TimeInForce.GTC,
            )

            self.submit_order(order)

    def on_stop(self):
        self.log.info("Strategy stopped.")
        for instrument_id in self.instrument_ids:
            self.unsubscribe_bars(instrument_id)


class RLIntegralStrategyConfig(StrategyConfig):
    instrument_ids: list[str]
    rebalance_interval_bars: int = 20
    lookback_window: int = 60
    model_path: Optional[str] = None


class RLIntegralStrategy(Strategy):
    """
    Strategy that uses an LSTM-based Alpha Model to predict returns,
    an Optimizer to determine weights, and Monte Carlo simulation for risk checking.
    """

    def __init__(self, config: RLIntegralStrategyConfig):
        super().__init__(config)
        self.instrument_ids = [InstrumentId.from_str(i) for i in config.instrument_ids]
        self.bar_count = 0

        # Initialize components
        self.alpha_model = LSTMAlphaModel(model_path=config.model_path)
        self.optimizer = OptimizerEngine()
        self.risk_model = CovarianceEstimator()
        self.objective = MaximizeReturns()

        # History buffer: Dictionary of lists
        self.history = {str(i): [] for i in config.instrument_ids}

    def on_start(self):
        self.log.info("RL Strategy started.")
        for instrument_id in self.instrument_ids:
            self.subscribe_bars(instrument_id)

    def on_bar(self, bar: Bar):
        self.bar_count += 1

        # Store bar data
        # Assuming bar.close is float or Decimal
        # Convert instrument_id to str to use as key
        inst_key = str(bar.instrument_id)
        if inst_key not in self.history:
            self.history[inst_key] = []

        self.history[inst_key].append(
            {
                "timestamp": bar.ts_event,  # or ts_init
                "close": float(bar.close),
            }
        )

        # Trim history to reasonable size to prevent memory leak, but keep enough for lookback
        # We need at least lookback_window for the model
        max_history = self.config.lookback_window * 2
        if len(self.history[inst_key]) > max_history:
            self.history[inst_key] = self.history[inst_key][-max_history:]

        if self.bar_count % self.config.rebalance_interval_bars != 0:
            return

        self.log.info(f"Rebalancing at bar {self.bar_count}")
        self._rebalance()

    def _rebalance(self):
        # Prepare data for Alpha Model
        # Convert history to DataFrame
        data_frames = []
        for ticker, records in self.history.items():
            if not records:
                continue
            df = pd.DataFrame(records)
            df["ticker"] = ticker
            data_frames.append(df)

        if not data_frames:
            return

        market_history = pd.concat(data_frames)

        # 1. Predict Expected Returns (Alpha)
        # This uses the LSTM model
        expected_returns = self.alpha_model.predict(market_history)

        # 2. Calculate Covariance (Risk)
        # Pivot market history for covariance calculation
        prices_wide = market_history.pivot(
            index="timestamp", columns="ticker", values="close"
        )
        # Handle duplicate timestamps if any
        prices_wide = prices_wide.groupby(prices_wide.index).last()

        # Need enough data for covariance
        if len(prices_wide) < 5:
            return

        covariance_matrix = self.risk_model.calculate_sample_covariance(prices_wide)

        # 3. Optimize Weights
        tickers = list(self.history.keys())
        # Filter tickers that have data in expected_returns and covariance
        valid_tickers = [
            t
            for t in tickers
            if t in expected_returns.index and t in covariance_matrix.index
        ]

        if not valid_tickers:
            return

        try:
            optimal_weights = self.optimizer.optimize(
                tickers=valid_tickers,
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                objective=self.objective,
            )
        except Exception as e:
            self.log.error(f"Optimization failed: {e}")
            return

        self.log.info(f"Optimal Weights: {optimal_weights.to_dict()}")

        # 4. Run Monte Carlo Simulation (for logging/risk check)
        # In mock, equity might be float, in real nautilus, it's Money object, convert to float
        equity_obj = self.portfolio.account(self.venue_id).equity_total
        total_equity = (
            float(equity_obj)
            if hasattr(equity_obj, "__float__")
            else float(str(equity_obj))
        )

        sim_results = run_monte_carlo_simulation(
            weights=optimal_weights,
            mean_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            initial_equity=total_equity,
            time_horizon_days=20,  # Simulate next month
            num_simulations=500,
        )

        self.log.info(
            f"Monte Carlo Risk Analysis (20 days): VaR 95%: {sim_results['VaR_95_equity']:.2f}"
        )

        # 5. Execute Trades
        self._execute_weights(optimal_weights, total_equity)

    def _execute_weights(self, target_weights: pd.Series, total_equity: float):
        if total_equity <= 0:
            return

        for instrument_id_str, weight in target_weights.items():
            instrument_id = InstrumentId.from_str(instrument_id_str)
            instrument = self.instrument(instrument_id)

            if not instrument:
                continue

            target_value = total_equity * weight

            # Using cached bar for execution price check
            # In mock, cache.bar might return None if not set
            cached_bar = self.cache.bar(instrument_id)
            if not cached_bar:
                continue

            last_price = float(cached_bar.close)

            if last_price <= 0:
                continue

            target_qty = int(target_value / last_price)

            # Get current position
            pos = self.portfolio.position(instrument_id)
            current_qty = pos.quantity if pos else 0

            diff = target_qty - current_qty

            if diff == 0:
                continue

            side = OrderSide.BUY if diff > 0 else OrderSide.SELL
            qty = abs(diff)

            # Basic check to avoid tiny dust orders
            if qty == 0:
                continue

            order = self.order_factory.market(
                instrument_id=instrument_id,
                order_side=side,
                quantity=self.instrument(instrument_id).make_qty(qty),
                time_in_force=TimeInForce.GTC,
            )

            self.submit_order(order)

    def on_stop(self):
        self.log.info("RL Strategy stopped.")
        for instrument_id in self.instrument_ids:
            self.unsubscribe_bars(instrument_id)
