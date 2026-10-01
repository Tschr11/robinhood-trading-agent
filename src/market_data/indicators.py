"""
indicators.py - Common technical indicators, calculated from candles.

    SMA  (Simple Moving Average)  average closing price of the last N candles
    RSI  (Relative Strength Index, 0-100)  how strong recent gains are
         compared with recent losses. Above 70 is often called "overbought",
         below 30 "oversold". Uses Wilder's smoothing, the standard method.
    VWAP (Volume-Weighted Average Price)  the average price paid today,
         where busy candles count more:
             sum(typical price x volume) / sum(volume)
             typical price = (high + low + close) / 3
    Average volume  mean volume of the last N candles

If there are not enough candles for a calculation, an
InsufficientDataError is raised. No value is ever guessed or made up.
"""

import math
from dataclasses import dataclass
from datetime import datetime

from src.market_data.candles import (Candle, DataKind, InsufficientDataError,
                                     MarketDataError)
from src.market_data.dataset import MarketDataSet
from src.market_data.validation import validate_candles

# SMA 50 needs the most history, so a full snapshot needs 50 candles.
MIN_CANDLES_FOR_INDICATORS = 50


# --- Individual indicators ----------------------------------------------------

def sma(closes: list[float], period: int) -> float:
    """Simple moving average of the last `period` closing prices."""
    _check_period(period)
    _check_numbers(closes, "closing prices")
    if len(closes) < period:
        raise InsufficientDataError(
            f"SMA {period} needs at least {period} closing prices but got {len(closes)}.")
    return sum(closes[-period:]) / period


def rsi(closes: list[float], period: int = 14) -> float:
    """
    Relative Strength Index using Wilder's smoothing. Needs period + 1
    closes (14 price CHANGES need 15 prices).

    Steps:
      1. changes = each close minus the close before it
      2. first average gain/loss = simple average over the first `period` changes
      3. after that: new average = (previous average x (period - 1) + latest) / period
      4. RS = average gain / average loss;  RSI = 100 - 100 / (1 + RS)
    """
    _check_period(period)
    _check_numbers(closes, "closing prices")
    if len(closes) < period + 1:
        raise InsufficientDataError(
            f"RSI {period} needs at least {period + 1} closing prices but got {len(closes)}.")

    changes = [after - before for before, after in zip(closes, closes[1:])]
    gains = [max(change, 0.0) for change in changes]
    losses = [max(-change, 0.0) for change in changes]

    average_gain = sum(gains[:period]) / period
    average_loss = sum(losses[:period]) / period
    for gain, loss in zip(gains[period:], losses[period:]):
        average_gain = (average_gain * (period - 1) + gain) / period
        average_loss = (average_loss * (period - 1) + loss) / period

    if average_loss == 0:
        # No losses at all: 100 if there were gains, 50 if prices never moved.
        return 100.0 if average_gain > 0 else 50.0
    relative_strength = average_gain / average_loss
    return 100 - 100 / (1 + relative_strength)


def vwap(candles: list[Candle]) -> float:
    """Volume-weighted average price over ALL the candles given."""
    validate_candles(candles)
    total_volume = sum(c.volume for c in candles)
    if total_volume == 0:
        raise MarketDataError("VWAP is undefined because total volume is zero.")
    weighted = sum((c.high + c.low + c.close) / 3 * c.volume for c in candles)
    return weighted / total_volume


def latest_session(candles: list[Candle]) -> list[Candle]:
    """
    The candles from the same calendar day as the most recent candle.
    The day is read in each timestamp's own time zone, so give timestamps
    in the exchange's local time (e.g. -05:00 for New York) for correct days.
    """
    validate_candles(candles)
    session_day = candles[-1].timestamp.date()
    return [c for c in candles if c.timestamp.date() == session_day]


def session_vwap(candles: list[Candle]) -> float:
    """VWAP for the latest trading day only (VWAP resets every day)."""
    return vwap(latest_session(candles))


def average_volume(candles: list[Candle], period: int = 20) -> float:
    """Mean volume of the last `period` candles."""
    _check_period(period)
    validate_candles(candles)
    if len(candles) < period:
        raise InsufficientDataError(
            f"Average volume {period} needs at least {period} candles but got {len(candles)}.")
    return sum(c.volume for c in candles[-period:]) / period


# --- Everything at once -----------------------------------------------------

@dataclass(frozen=True)
class IndicatorSnapshot:
    """All indicators for one symbol at one moment, with labels."""
    symbol: str
    kind: DataKind        # historical or live - carried over from the data
    source: str
    as_of: datetime       # timestamp of the latest candle used
    close: float
    sma_20: float
    sma_50: float
    rsi_14: float
    vwap: float           # latest session only
    average_volume_20: float

    def describe(self) -> str:
        """A short, beginner-friendly summary."""
        return (f"{self.symbol} ({self.kind.value} data from {self.source}, "
                f"as of {self.as_of.isoformat()}): close ${self.close:.2f}, "
                f"SMA 20 ${self.sma_20:.2f}, SMA 50 ${self.sma_50:.2f}, "
                f"RSI 14 {self.rsi_14:.1f}, VWAP ${self.vwap:.2f}, "
                f"average volume {self.average_volume_20:,.0f}")


def compute_indicators(dataset: MarketDataSet) -> IndicatorSnapshot:
    """Calculate every indicator for a validated dataset (needs 50+ candles)."""
    if not isinstance(dataset, MarketDataSet):
        raise MarketDataError(f"Expected a MarketDataSet (got {type(dataset).__name__}).")
    if len(dataset) < MIN_CANDLES_FOR_INDICATORS:
        raise InsufficientDataError(
            f"{dataset.symbol}: need at least {MIN_CANDLES_FOR_INDICATORS} candles "
            f"for SMA 50 but got {len(dataset)}.")

    candles = list(dataset.candles)
    closes = dataset.closes
    return IndicatorSnapshot(
        symbol=dataset.symbol,
        kind=dataset.kind,
        source=dataset.source,
        as_of=dataset.latest.timestamp,
        close=dataset.latest_close,
        sma_20=sma(closes, 20),
        sma_50=sma(closes, 50),
        rsi_14=rsi(closes, 14),
        vwap=session_vwap(candles),
        average_volume_20=average_volume(candles, 20),
    )


# --- Input checks -------------------------------------------------------------

def _check_period(period) -> None:
    if isinstance(period, bool) or not isinstance(period, int) or period < 1:
        raise MarketDataError(f"period must be a whole number of at least 1 (got {period!r}).")


def _check_numbers(values, what: str) -> None:
    if not isinstance(values, (list, tuple)):
        raise MarketDataError(f"{what} must be a list (got {type(values).__name__}).")
    for position, value in enumerate(values, start=1):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise MarketDataError(
                f"{what}: item {position} must be a price above zero (got {value!r}).")
