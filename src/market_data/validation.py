"""
validation.py - Checks candles BEFORE anything uses them.

Rules for every candle:
    - timestamp is present, is a datetime, and includes a time zone
    - open/high/low/close/volume are present, real numbers (not text,
      not True/False, not NaN, not infinity)
    - prices are greater than zero; volume is zero or more
    - the bar is possible: high is the highest value, low the lowest
Rules for the whole list:
    - no two candles share a timestamp
    - timestamps go forward in time (oldest first)
    - there are enough candles for what you want to do

Problems are collected and reported together. Nothing is ever fixed,
skipped, or filled in: bad data is rejected, full stop.
"""

import math
from datetime import datetime

from src.market_data.candles import Candle, InsufficientDataError, MarketDataError

PRICE_FIELDS = ("open", "high", "low", "close")


def find_problems(candles) -> list[str]:
    """Return a plain-English description of every problem (empty = all good)."""
    if not isinstance(candles, (list, tuple)):
        return [f"Candles must be a list (got {type(candles).__name__})."]

    problems = []
    seen_timestamps = set()
    latest_timestamp = None

    for number, candle in enumerate(candles, start=1):
        label = f"Candle {number}:"
        if not isinstance(candle, Candle):
            problems.append(f"{label} is not a Candle (got {type(candle).__name__}).")
            continue

        # --- Timestamp -------------------------------------------------------
        ts = candle.timestamp
        timestamp_ok = False
        if ts is None:
            problems.append(f"{label} timestamp is missing.")
        elif not isinstance(ts, datetime):
            problems.append(f"{label} timestamp must be a datetime (got {ts!r}).")
        elif ts.utcoffset() is None:
            problems.append(f"{label} timestamp {ts.isoformat()} has no time zone.")
        else:
            timestamp_ok = True
            label = f"Candle {number} ({ts.isoformat()}):"

        # --- Numbers ---------------------------------------------------------
        numbers_ok = True
        for field in PRICE_FIELDS + ("volume",):
            value = getattr(candle, field)
            problem = _number_problem(field, value)
            if problem:
                problems.append(f"{label} {problem}")
                numbers_ok = False

        # --- Is the bar physically possible? --------------------------------
        if numbers_ok:
            o, h, l, c = candle.open, candle.high, candle.low, candle.close
            if h < l:
                problems.append(f"{label} high ({h}) is below low ({l}).")
            else:
                if h < o:
                    problems.append(f"{label} high ({h}) is below open ({o}).")
                if h < c:
                    problems.append(f"{label} high ({h}) is below close ({c}).")
                if l > o:
                    problems.append(f"{label} low ({l}) is above open ({o}).")
                if l > c:
                    problems.append(f"{label} low ({l}) is above close ({c}).")

        # --- Order and duplicates --------------------------------------------
        if timestamp_ok:
            if ts in seen_timestamps:
                problems.append(f"{label} duplicate timestamp.")
            elif latest_timestamp is not None and ts < latest_timestamp:
                problems.append(f"{label} is out of order (earlier than the "
                                "candle before it; oldest must come first).")
            seen_timestamps.add(ts)
            if latest_timestamp is None or ts > latest_timestamp:
                latest_timestamp = ts

    return problems


def validate_candles(candles, min_candles: int = 1) -> None:
    """Raise MarketDataError (listing every problem) unless the candles are valid."""
    problems = find_problems(candles)
    count = len(candles) if isinstance(candles, (list, tuple)) else 0
    if count < min_candles:
        shortage = f"Need at least {min_candles} candle(s) but got {count}."
        if not problems:
            raise InsufficientDataError(shortage)
        problems.append(shortage)
    if problems:
        raise MarketDataError(problems)


def _number_problem(field: str, value) -> str | None:
    if value is None:
        return f"{field} is missing."
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{field} must be a number (got {value!r})."
    if not math.isfinite(value):
        return f"{field} must be a real, finite number (got {value})."
    if field == "volume":
        if value < 0:
            return f"volume cannot be negative (got {value})."
    elif value <= 0:
        return f"{field} price must be greater than zero (got {value})."
    return None
