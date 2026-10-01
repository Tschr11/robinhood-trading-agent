"""
market_data.py - Where prices come from.

For now this returns made-up (simulated) prices so the rest of the agent
can be built and tested without any internet connection or API keys.
Later we can swap in a real, read-only historical data source.
"""

import random


def get_latest_price(symbol: str) -> float:
    """Return a simulated current price for a symbol."""
    base_prices = {"SPY": 500.00, "QQQ": 430.00}
    base = base_prices.get(symbol, 100.00)
    # Move the price randomly by up to +/- 1% to imitate a live market.
    return round(base * (1 + random.uniform(-0.01, 0.01)), 2)


def get_recent_prices(symbol: str, count: int = 20) -> list[float]:
    """Return a short list of simulated recent prices (oldest first)."""
    return [get_latest_price(symbol) for _ in range(count)]
