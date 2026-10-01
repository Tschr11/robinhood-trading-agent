"""
strategy.py - Decides WHAT the agent would like to do.

A strategy only makes suggestions: "BUY", "SELL", or "HOLD".
It never places trades itself - the risk manager must approve first.
"""


def simple_moving_average(prices: list[float]) -> float:
    """Average of a list of prices."""
    return sum(prices) / len(prices)


def generate_signal(prices: list[float]) -> tuple[str, str]:
    """
    Very simple placeholder strategy:
      - BUY if the latest price is above its recent average (upward momentum)
      - SELL if it is below the average
      - HOLD if there is not enough data

    Returns the signal and a plain-English reason for the dashboard.
    """
    if len(prices) < 5:
        return "HOLD", "Not enough price history yet."

    average = simple_moving_average(prices)
    latest = prices[-1]

    if latest > average:
        return "BUY", f"Price {latest:.2f} is above its average {average:.2f}."
    if latest < average:
        return "SELL", f"Price {latest:.2f} is below its average {average:.2f}."
    return "HOLD", "Price is right at its average."
