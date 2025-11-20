import unittest
from unittest.mock import MagicMock, patch

# Add path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNautilusIntegration(unittest.TestCase):
    def setUp(self):
        # Mock nautilus_trader modules
        self.mock_nautilus = MagicMock()
        self.module_patcher = patch.dict(
            "sys.modules",
            {
                "nautilus_trader": self.mock_nautilus,
                "nautilus_trader.trading.strategy": self.mock_nautilus,
                "nautilus_trader.model.data.bar": self.mock_nautilus,
                "nautilus_trader.config": self.mock_nautilus,
                "nautilus_trader.model.enums": self.mock_nautilus,
                "nautilus_trader.model.instruments": self.mock_nautilus,
                "nautilus_trader.live.node": self.mock_nautilus,
                "nautilus_trader.live.config": self.mock_nautilus,
                "nautilus_trader.adapters.alpaca.config": self.mock_nautilus,
                "nautilus_trader.model.identifiers": self.mock_nautilus,
                "nautilus_trader.model.currencies": self.mock_nautilus,
            },
        )
        self.module_patcher.start()

    def tearDown(self):
        self.module_patcher.stop()
        # Unload mind modules that imported mocks
        for module in list(sys.modules.keys()):
            if (
                module.startswith("mind.strategy")
                or module.startswith("mind.config")
                or module.startswith("mind.run_strategy")
            ):
                del sys.modules[module]

    def test_strategy_instantiation(self):
        from mind.strategy import IntegralStrategy, IntegralStrategyConfig

        # Since IntegralStrategyConfig inherits from a Mock (via StrategyConfig),
        # instantiating it creates a Mock. We can just use it or mock it.
        # Or simpler: just mock the config object passed to strategy.

        config = MagicMock()
        config.instrument_id = "SPY.ALPACA"

        strategy = IntegralStrategy(config)
        self.assertIsNotNone(strategy)

        # Verify on_bar exists
        self.assertTrue(hasattr(strategy, "on_bar"))

    def test_config_generation(self):
        from mind.config import get_nautilus_config

        # Backtest mode
        config = get_nautilus_config("backtest")
        self.assertIsNotNone(config)

        # Live mode (requires env vars, mock them)
        with patch.dict(
            os.environ, {"ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test"}
        ):
            config_live = get_nautilus_config("live")
            self.assertIsNotNone(config_live)


if __name__ == "__main__":
    unittest.main()
