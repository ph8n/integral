import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add the parent directory (mind) to sys.path so we can import the scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock modules BEFORE importing the scripts
sys.modules["quantrocket"] = MagicMock()
sys.modules["quantrocket.master"] = MagicMock()
sys.modules["quantrocket.history"] = MagicMock()
sys.modules["pandas"] = MagicMock()

# Now we can import the scripts
import create_universe
import ingest_daily
import export_history


class TestMarketIngest(unittest.TestCase):
    def setUp(self):
        # Reset mocks before each test
        sys.modules["quantrocket.master"].reset_mock()
        sys.modules["quantrocket.history"].reset_mock()
        sys.modules["pandas"].reset_mock()

    def test_create_universe(self):
        # Setup
        mock_pd = sys.modules["pandas"]
        mock_df = MagicMock()
        mock_df.__getitem__.return_value.tolist.return_value = ["AAPL", "MSFT"]
        mock_pd.read_csv.return_value = mock_df

        # Execute
        create_universe.main()

        # Verify
        mock_pd.read_csv.assert_called_with("data/sp500_tickers.csv")
        sys.modules["quantrocket.master"].create_universe.assert_called_with(
            "sp500", tickers=["AAPL", "MSFT"]
        )

    def test_ingest_daily(self):
        # Execute
        ingest_daily.main()

        # Verify
        sys.modules["quantrocket.history"].create_db.assert_called_with(
            "yahoo-eod", provider="yahoo", bar_size="1 day"
        )
        sys.modules["quantrocket.history"].collect_history.assert_called_with(
            "yahoo-eod", universes=["sp500"]
        )

    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_export_history(self, mock_exists, mock_makedirs):
        # Setup
        mock_exists.return_value = False  # Simulate directory not existing

        # Execute
        export_history.main()

        # Verify
        mock_makedirs.assert_called_with("data/exports")

        expected_output = os.path.join("data/exports", "market_data.csv")
        sys.modules["quantrocket.history"].download_history_file.assert_called_with(
            "yahoo-eod",
            expected_output,
            fields=["Open", "High", "Low", "Close", "Volume"],
        )


if __name__ == "__main__":
    unittest.main()
