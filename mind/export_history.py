import os
from quantrocket.history import download_history_file

DB_CODE = "yahoo-eod"
EXPORT_DIR = "data/exports"
OUTPUT_FILE = os.path.join(EXPORT_DIR, "market_data.csv")


def main():
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)

    print(f"Exporting data from '{DB_CODE}' to {OUTPUT_FILE}...")

    # Download history to a CSV file
    # Fields: Open,High,Low,Close,Volume are standard.
    # The resulting CSV typically has a MultiIndex (Field, Date, ConId/Symbol) if using get_historical_prices
    # download_history_file usually dumps a CSV matching the internal schema or a flat format.

    download_history_file(
        DB_CODE, OUTPUT_FILE, fields=["Open", "High", "Low", "Close", "Volume"]
    )

    print("Export complete.")


if __name__ == "__main__":
    main()
