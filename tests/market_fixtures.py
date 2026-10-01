"""
Locally generated market data for tests.

Everything here is built from simple, predictable rules in code - no
downloads, no randomness - so every test run sees exactly the same data.
This file is a helper, not a test module (its name doesn't start with test_).
"""

import os
from datetime import datetime, timedelta, timezone

from src.market_data import Candle

# A fixed -05:00 offset (New York standard time) so "which day is it?"
# is unambiguous without needing a time-zone database.
NEW_YORK = timezone(timedelta(hours=-5))
MARKET_OPEN = datetime(2026, 1, 5, 9, 30, tzinfo=NEW_YORK)
FIVE_MINUTES = timedelta(minutes=5)


def candle(timestamp=MARKET_OPEN, open=100.0, high=101.0, low=99.0,
           close=100.5, volume=1000.0) -> Candle:
    """One valid candle; override any field to break it on purpose."""
    return Candle(timestamp, open, high, low, close, volume)


def candles_from_closes(closes, start=MARKET_OPEN, step=FIVE_MINUTES,
                        volume=1000.0, wick=0.5) -> list[Candle]:
    """
    Build valid candles that close at the given prices.
    Each candle opens at the previous close; high/low add a small `wick`.
    `volume` may be one number or a list (one per candle).
    """
    volumes = volume if isinstance(volume, list) else [volume] * len(closes)
    result = []
    previous = closes[0]
    for i, (close, vol) in enumerate(zip(closes, volumes)):
        open_ = previous
        result.append(Candle(start + i * step, open_,
                             max(open_, close) + wick, min(open_, close) - wick,
                             close, vol))
        previous = close
    return result


def rising_closes(count, start=100.0, step=1.0) -> list[float]:
    return [start + i * step for i in range(count)]


def zigzag_closes(count, start=100.0) -> list[float]:
    """Up 2, down 1, up 2, down 1, ... - a gently rising, realistic-ish path."""
    closes, price = [], start
    for i in range(count):
        closes.append(price)
        price += 2.0 if i % 2 == 0 else -1.0
    return closes


def write_csv(folder, symbol, candles) -> str:
    """Save candles to <folder>/<symbol>.csv in the provider's format."""
    rows = ["timestamp,open,high,low,close,volume"]
    for c in candles:
        rows.append(f"{c.timestamp.isoformat()},{c.open},{c.high},{c.low},"
                    f"{c.close},{c.volume}")
    return write_raw_csv(folder, symbol, "\n".join(rows) + "\n")


def write_raw_csv(folder, symbol, text) -> str:
    """Save exact CSV text, e.g. with deliberately broken rows."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{symbol}.csv")
    with open(path, "w", newline="") as f:
        f.write(text)
    return path
