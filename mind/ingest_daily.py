from quantrocket.history import create_db, collect_history
from quantrocket.master import list_universes

DB_CODE = "yahoo-eod"
UNIVERSE = "sp500"


def main():
    print(f"Setting up historical database '{DB_CODE}'...")
    # Create the database if it doesn't exist
    # provider='yahoo' is standard for free data in QuantRocket
    try:
        create_db(DB_CODE, provider="yahoo", bar_size="1 day")
        print(f"Database '{DB_CODE}' created/updated.")
    except Exception as e:
        print(f"Note: {e}")

    print(f"Starting ingestion for universe '{UNIVERSE}'...")
    # Collect history
    # This triggers the download. It runs in the background in QuantRocket usually,
    # but the client call might wait or return a task ID.
    collect_history(DB_CODE, universes=[UNIVERSE])
    print("Ingestion triggered.")


if __name__ == "__main__":
    main()
