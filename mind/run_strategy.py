import json
import subprocess
import sys
import os
from typing import Dict, List


def get_mock_prices() -> Dict[str, float]:
    """
    Returns mock prices for a fixed universe of tickers.
    """
    return {"AAPL": 150.00, "MSFT": 300.00, "GOOGL": 2800.00}


def calculate_target_shares(
    total_equity: float, prices: Dict[str, float], weights: Dict[str, float]
) -> Dict[str, int]:
    """
    Calculates target share counts based on total equity and weights.
    """
    target_shares = {}
    for ticker, price in prices.items():
        if ticker in weights:
            target_equity = total_equity * weights[ticker]
            # Floor division to get integer shares
            shares = int(target_equity / price)
            target_shares[ticker] = shares
    return target_shares


def call_composer(
    target_quantities: Dict[str, int], current_portfolio: Dict
) -> List[Dict]:
    """
    Calls the C++ Composer engine to generate orders.
    """
    # Construct input payload
    payload = {"target": target_quantities, "current": current_portfolio}

    input_json = json.dumps(payload)

    # Path to the composer binary
    # Assuming this script is run from the 'mind' directory or we resolve relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    composer_binary = os.path.join(
        script_dir, "..", "composer", "build", "integral_composer"
    )

    if not os.path.exists(composer_binary):
        raise FileNotFoundError(f"Composer binary not found at {composer_binary}")

    stdout = ""
    try:
        process = subprocess.Popen(
            [composer_binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout, stderr = process.communicate(input=input_json)

        if process.returncode != 0:
            print(f"Composer Error: {stderr}", file=sys.stderr)
            sys.exit(process.returncode)

        return json.loads(stdout)

    except subprocess.CalledProcessError as e:
        print(f"Failed to execute composer: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Failed to parse composer output: {e}", file=sys.stderr)
        print(f"Raw Output: {stdout}", file=sys.stderr)
        sys.exit(1)


def main():
    # Configuration
    total_equity = 100000.0
    universe = ["AAPL", "MSFT", "GOOGL"]

    # Equal Weight Allocation (1/N)
    weight = 1.0 / len(universe)
    weights = {ticker: weight for ticker in universe}

    print(f"Running Equal Weight Strategy for {universe}")
    print(f"Total Equity: ${total_equity:,.2f}")

    # 1. Get Prices
    prices = get_mock_prices()
    print("\nMarket Prices:")
    for ticker, price in prices.items():
        print(f"  {ticker}: ${price:,.2f}")

    # 2. Calculate Target Shares
    target_shares = calculate_target_shares(total_equity, prices, weights)
    print("\nTarget Shares:")
    for ticker, shares in target_shares.items():
        print(f"  {ticker}: {shares}")

    # 3. Current Portfolio (Mocked as Empty)
    current_portfolio = {"positions": {}, "cash": total_equity}

    # 4. Call Composer
    print("\nCalling Composer...")
    orders = call_composer(target_shares, current_portfolio)

    # 5. Display Orders
    print(f"\nGenerated {len(orders)} Orders:")
    print(json.dumps(orders, indent=4))


if __name__ == "__main__":
    main()
