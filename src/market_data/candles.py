"""
candles.py - The basic building blocks of market data.

A "candle" (or "bar") summarizes trading over one period, e.g. 5 minutes:
    timestamp  when the period started (must include a time zone)
    open       first traded price in the period
    high       highest price in the period
    low        lowest price in the period
    close      last traded price in the period
    volume     how many shares traded in the period

Together these five numbers are called OHLCV.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class DataKind(Enum):
    """
    Where in time the data comes from. Every dataset carries one of these,
    so historical data can never be mistaken for live prices.
    """
    HISTORICAL = "historical"  # past candles, e.g. from a saved CSV file
    LIVE = "live"              # current prices from a real-time feed


@dataclass(frozen=True)
class Candle:
    """One OHLCV bar. `frozen=True` means it can't be changed after creation."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataError(ValueError):
    """
    Raised when market data is missing or invalid. It lists EVERY problem
    found, so you can fix the data in one go. The data is never "repaired".
    """
    MAX_SHOWN = 10

    def __init__(self, problems):
        if isinstance(problems, str):
            problems = [problems]
        self.problems = list(problems)
        shown = self.problems[:self.MAX_SHOWN]
        message = " ".join(shown)
        hidden = len(self.problems) - len(shown)
        if hidden:
            message += f" (and {hidden} more problem(s))"
        super().__init__(message)


class InsufficientDataError(MarketDataError):
    """Raised when there are too few candles for a calculation."""
