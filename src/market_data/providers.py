"""
providers.py - WHERE market data comes from, behind one common interface.

A "provider" is any data source: a CSV file, a test fixture, and later
perhaps a real data feed. Every provider looks the same to the rest of the
program, so the source can be swapped without changing anything else.

To add a new provider, subclass MarketDataProvider and fill in:
    name             a short label, e.g. "csv"
    kind             DataKind.HISTORICAL or DataKind.LIVE
    _load_candles()  return the raw candles for one symbol (oldest first)

You do NOT write validation: get_candles() always validates whatever
_load_candles() returns, so no provider can skip the checks.

Providers in this file are all OFFLINE. None of them connect to the
internet, a broker, or Robinhood.
"""

import csv
import os
from abc import ABC, abstractmethod
from datetime import datetime

from config import settings
from src.market_data.candles import (Candle, DataKind, InsufficientDataError,
                                     MarketDataError)
from src.market_data.dataset import MarketDataSet

CSV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class MarketDataProvider(ABC):
    """The interface every data source must follow."""

    name: str = ""
    kind: DataKind | None = None

    @abstractmethod
    def _load_candles(self, symbol: str) -> list[Candle]:
        """Return the raw candles for `symbol`, oldest first."""

    def source_for(self, symbol: str) -> str:
        """Describe where `symbol`'s data comes from (shown on every dataset)."""
        return self.name

    def get_candles(self, symbol: str, limit: int | None = None,
                    min_candles: int = 1) -> MarketDataSet:
        """
        Load, validate and label candles for `symbol`.

        limit        keep only the most recent `limit` candles (None = all)
        min_candles  fail with InsufficientDataError if fewer are available

        Raises MarketDataError if the data is missing or invalid. It never
        returns empty, partial, or substituted data.
        """
        symbol = clean_symbol(symbol)
        if not isinstance(self.kind, DataKind):
            raise MarketDataError(f"Provider '{self.name}' must declare kind as "
                                  "DataKind.HISTORICAL or DataKind.LIVE.")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)
                                  or limit < 1):
            raise MarketDataError(f"limit must be a whole number of at least 1 (got {limit!r}).")

        raw = self._load_candles(symbol)
        if not isinstance(raw, (list, tuple)):
            raise MarketDataError(f"Provider '{self.name}' returned "
                                  f"{type(raw).__name__} instead of a list of candles.")

        # Creating the dataset validates EVERY loaded candle.
        dataset = MarketDataSet(symbol, self.kind, self.source_for(symbol), tuple(raw))
        if limit is not None:
            dataset = dataset.last(limit)
        if len(dataset) < min_candles:
            raise InsufficientDataError(
                f"{symbol}: need at least {min_candles} candles but "
                f"{dataset.source} has {len(dataset)}.")
        return dataset


def clean_symbol(symbol) -> str:
    """' spy ' -> 'SPY'. Only letters, digits, '.' and '-' are allowed."""
    if not isinstance(symbol, str) or not symbol.strip():
        raise MarketDataError(f"Symbol must be a ticker such as 'SPY' (got {symbol!r}).")
    symbol = symbol.strip().upper()
    # This also stops tricks like "../secret" being used as a file name.
    if len(symbol) > 10 or not all(ch.isalnum() or ch in ".-" for ch in symbol):
        raise MarketDataError(f"Symbol {symbol!r} contains invalid characters.")
    return symbol


# --- Provider 1: CSV files on your computer -----------------------------------

class CSVHistoricalProvider(MarketDataProvider):
    """
    Reads HISTORICAL candles from <folder>/<SYMBOL>.csv with columns:

        timestamp,open,high,low,close,volume
        2026-01-05T09:30:00-05:00,500.10,500.80,499.90,500.50,120000

    Timestamps must include a time zone. Rows must be oldest first.
    Blank cells are reported as missing - never filled in.
    """

    name = "csv"
    kind = DataKind.HISTORICAL

    def __init__(self, folder: str = settings.MARKET_DATA_DIR):
        self.folder = folder

    def path_for(self, symbol: str) -> str:
        return os.path.join(self.folder, f"{symbol}.csv")

    def source_for(self, symbol: str) -> str:
        return f"csv:{self.path_for(symbol)}"

    def _load_candles(self, symbol: str) -> list[Candle]:
        path = self.path_for(symbol)
        if not os.path.exists(path):
            raise MarketDataError(f"No historical data file for {symbol} at {path}.")

        problems = []
        candles = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in CSV_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise MarketDataError(f"{path} is missing column(s): {', '.join(missing)}.")
            for row in reader:
                line = reader.line_num
                values = {}
                for column in CSV_COLUMNS:
                    values[column], problem = _parse_cell(column, row.get(column))
                    if problem:
                        problems.append(f"{path} line {line}: {problem}")
                candles.append(Candle(**values))
        if problems:
            raise MarketDataError(problems)
        return candles


def _parse_cell(column: str, text):
    """Turn one CSV cell into a value. Blank -> None (reported later as missing)."""
    if text is None or not text.strip():
        return None, None
    text = text.strip()
    if column == "timestamp":
        try:
            return datetime.fromisoformat(text), None
        except ValueError:
            return None, f"timestamp {text!r} is not a valid date and time."
    try:
        return float(text), None      # "nan"/"inf" parse, then fail validation
    except ValueError:
        return None, f"{column} {text!r} is not a number."


# --- Provider 2: candles already in memory -------------------------------------

class InMemoryProvider(MarketDataProvider):
    """
    Serves candles you hand it directly - useful for tests and, later, for
    backtests. You must say what kind of data it is; nothing is generated.
    """

    def __init__(self, candles_by_symbol: dict, kind: DataKind = DataKind.HISTORICAL,
                 name: str = "in-memory"):
        if not isinstance(candles_by_symbol, dict):
            raise MarketDataError("candles_by_symbol must be a dict like {'SPY': [...]}.")
        self._candles = {clean_symbol(s): list(c) for s, c in candles_by_symbol.items()}
        self.kind = kind
        self.name = name

    def _load_candles(self, symbol: str) -> list[Candle]:
        if symbol not in self._candles:
            raise MarketDataError(f"No {self.name} data for {symbol}.")
        return self._candles[symbol]
