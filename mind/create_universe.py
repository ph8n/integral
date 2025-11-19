import pandas as pd
from quantrocket.master import create_universe


def main():
    # Read tickers
    df = pd.read_csv("data/sp500_tickers.csv")
    tickers = df["Symbol"].tolist()

    print(f"Creating universe 'sp500' with {len(tickers)} tickers...")

    # Create universe
    # In QuantRocket, usually we ensure tickers exist first, but for Yahoo
    # we often just request the history and it figures it out, or we rely on
    # them being available. However, to create a named universe group,
    # we pass the tickers.
    result = create_universe("sp500", tickers=tickers)
    print("Result:", result)


if __name__ == "__main__":
    main()
