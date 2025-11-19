import pandas as pd


class CovarianceEstimator:
    """
    Calculates risk models (covariance matrices) from historical data.
    """

    def calculate_sample_covariance(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the sample covariance matrix of returns from price history.

        Args:
            prices: DataFrame with DatetimeIndex and columns as tickers.
                   Values should be close prices.

        Returns:
            DataFrame representing the covariance matrix of daily returns.
        """
        # Calculate percentage returns
        # dropna() ensures we don't have NaNs from the first row or missing data
        returns = prices.pct_change().dropna()

        # Calculate sample covariance
        # This is the standard unbiased estimator (divided by N-1)
        covariance = returns.cov()

        return covariance
