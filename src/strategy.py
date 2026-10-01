"""
strategy.py - Decides WHAT the agent would like to do: BUY, SELL or HOLD.

This is a deterministic, rules-based strategy ("trend_vwap_v1"). The same
data always produces the same signal, and every signal lists each rule with
the real numbers behind it. There is no machine learning, no AI model and no
randomness - just the explicit rules in config/settings.py:

    Entry (BUY)  - ALL must pass, and no position is open:
        E1  SMA 20 > SMA 50
        E2  close > VWAP
        E3  close > SMA 20
        E4  RSI_ENTRY_MIN <= RSI 14 <= RSI_ENTRY_MAX
        E5  latest volume >= average volume x VOLUME_MULTIPLIER
    Exit (SELL)  - ANY one passes, and a position is open:
        X1  SMA 20 < SMA 50
        X2  close < VWAP
        X3  RSI 14 >= RSI_EXIT
    Otherwise HOLD.

A signal is only a suggestion. This module never places orders, never
touches the paper account, and does not import the paper trader.

Data safety:
    - The caller must say which kind of data it expects (historical or
      live). A mismatch is an error, so historical data can't pose as live.
    - Live data that is too old (or time-stamped in the future) gives HOLD.
    - Too few candles or invalid indicator values give HOLD, never BUY/SELL.

IMPORTANT: these rules are a simple, common starting point for paper
research. Nothing here claims or implies that they make money.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from config import settings
from src.market_data import (MIN_CANDLES_FOR_INDICATORS, DataKind,
                             IndicatorSnapshot, MarketDataSet,
                             compute_indicators)


def _is_real(value) -> bool:
    """True only for ordinary finite numbers (not text, True/False, NaN or inf)."""
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


class StrategyError(ValueError):
    """Raised when the strategy is called incorrectly (a programming mistake)."""


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# --- Configuration ------------------------------------------------------------

@dataclass(frozen=True)
class StrategyConfig:
    """The strategy's thresholds. Defaults come from config/settings.py."""
    name: str = settings.STRATEGY_NAME
    rsi_entry_min: float = settings.RSI_ENTRY_MIN
    rsi_entry_max: float = settings.RSI_ENTRY_MAX
    rsi_exit: float = settings.RSI_EXIT
    volume_multiplier: float = settings.VOLUME_MULTIPLIER
    live_max_age_seconds: float = settings.LIVE_DATA_MAX_AGE_SECONDS

    def __post_init__(self):
        problems = []
        if not isinstance(self.name, str) or not self.name.strip():
            problems.append("name must be non-empty text.")
        numbers = {"rsi_entry_min": self.rsi_entry_min,
                   "rsi_entry_max": self.rsi_entry_max,
                   "rsi_exit": self.rsi_exit,
                   "volume_multiplier": self.volume_multiplier,
                   "live_max_age_seconds": self.live_max_age_seconds}
        for label, value in numbers.items():
            if not _is_real(value):
                problems.append(f"{label} must be a real number (got {value!r}).")
        if not problems:
            if not 0 <= self.rsi_entry_min < self.rsi_entry_max <= 100:
                problems.append("RSI entry range must satisfy "
                                "0 <= rsi_entry_min < rsi_entry_max <= 100.")
            if not 0 < self.rsi_exit <= 100:
                problems.append("rsi_exit must be above 0 and at most 100.")
            if self.volume_multiplier <= 0:
                problems.append("volume_multiplier must be greater than 0.")
            if self.live_max_age_seconds <= 0:
                problems.append("live_max_age_seconds must be greater than 0.")
        if problems:
            raise StrategyError("Invalid strategy settings: " + " ".join(problems))


DEFAULT_CONFIG = StrategyConfig()


# --- The signal ------------------------------------------------------------------

@dataclass(frozen=True)
class RuleCheck:
    """One rule, whether it passed, and the numbers behind it."""
    code: str        # e.g. "E1"
    name: str        # e.g. "Trend up"
    passed: bool
    detail: str      # e.g. "SMA 20 (105.20) > SMA 50 (101.00)"


@dataclass(frozen=True)
class StrategySignal:
    """A fully explained suggestion. It does NOT place an order."""
    symbol: str
    signal: Signal
    timestamp: datetime          # time of the latest candle used
    source: str                  # where the data came from
    kind: DataKind               # historical or live
    strategy: str                # strategy name/version
    has_open_position: bool
    data_ok: bool                # False = data problem, so the answer is HOLD
    summary: str
    rules: tuple[RuleCheck, ...] = field(default_factory=tuple)
    snapshot: IndicatorSnapshot | None = None

    @property
    def is_historical(self) -> bool:
        return self.kind is DataKind.HISTORICAL

    def explain(self) -> str:
        """A readable checklist for logs and the dashboard."""
        lines = [f"{self.symbol}: {self.signal.value} - {self.summary}",
                 f"  Data: {self.kind.value} from {self.source}, "
                 f"as of {self.timestamp.isoformat()} ({self.strategy})"]
        if self.is_historical:
            lines.append("  NOTE: historical data - for research only, "
                         "not a live trading signal.")
        for rule in self.rules:
            mark = "met    " if rule.passed else "not met"
            lines.append(f"  [{mark}] {rule.code} {rule.name}: {rule.detail}")
        return "\n".join(lines)


# --- Main entry point ------------------------------------------------------------

def evaluate(dataset: MarketDataSet, *, expected_kind: DataKind,
             has_open_position: bool, config: StrategyConfig = DEFAULT_CONFIG,
             now: datetime | None = None) -> StrategySignal:
    """
    Produce a signal for one symbol from validated market data.

    expected_kind      DataKind.LIVE or DataKind.HISTORICAL - what the caller
                       intends to use. A mismatch raises StrategyError.
    has_open_position  True if the account currently holds this symbol.
                       (Read it from the account; the strategy can't see it.)
    now                current time, used only to check live data's age.
    """
    if not isinstance(dataset, MarketDataSet):
        raise StrategyError(f"Expected a MarketDataSet (got {type(dataset).__name__}).")
    _check_common_arguments(expected_kind, has_open_position, config)
    if dataset.kind is not expected_kind:
        raise StrategyError(
            f"{dataset.symbol} data from {dataset.source} is {dataset.kind.value}, "
            f"but {expected_kind.value} data was expected. Historical data is "
            "never treated as live.")

    # Enough candles for every indicator?
    if len(dataset) < MIN_CANDLES_FOR_INDICATORS:
        return _data_problem(dataset.symbol, dataset.latest.timestamp, dataset.source,
                             dataset.kind, has_open_position, config,
                             f"Not enough data: need {MIN_CANDLES_FOR_INDICATORS} "
                             f"candles, have {len(dataset)}.")

    # Live data must be fresh.
    if dataset.is_live:
        problem = _live_age_problem(dataset.latest.timestamp, config, now)
        if problem:
            return _data_problem(dataset.symbol, dataset.latest.timestamp,
                                 dataset.source, dataset.kind, has_open_position,
                                 config, problem)

    snapshot = compute_indicators(dataset)
    return evaluate_snapshot(snapshot, dataset.latest.volume,
                             has_open_position=has_open_position, config=config)


def evaluate_snapshot(snapshot: IndicatorSnapshot, latest_volume: float, *,
                      has_open_position: bool,
                      config: StrategyConfig = DEFAULT_CONFIG) -> StrategySignal:
    """
    The pure rule logic, given already-calculated indicators.
    `latest_volume` is the volume of the most recent candle.
    """
    if not isinstance(snapshot, IndicatorSnapshot):
        raise StrategyError(f"Expected an IndicatorSnapshot (got {type(snapshot).__name__}).")
    _check_common_arguments(snapshot.kind, has_open_position, config)

    # Values could be hand-made or corrupted: never decide on bad numbers.
    values = {"close": snapshot.close, "SMA 20": snapshot.sma_20,
              "SMA 50": snapshot.sma_50, "VWAP": snapshot.vwap,
              "average volume": snapshot.average_volume_20}
    bad = [f"{label}={value!r}" for label, value in values.items()
           if not _is_real(value) or value <= 0]
    if not _is_real(snapshot.rsi_14) or not 0 <= snapshot.rsi_14 <= 100:
        bad.append(f"RSI 14={snapshot.rsi_14!r}")
    if not _is_real(latest_volume) or latest_volume < 0:
        bad.append(f"latest volume={latest_volume!r}")
    if bad:
        return _data_problem(snapshot.symbol, snapshot.as_of, snapshot.source,
                             snapshot.kind, has_open_position, config,
                             "Invalid indicator values: " + ", ".join(bad) + ".",
                             snapshot)

    if has_open_position:
        rules = exit_rules(snapshot, config)
        met = [r for r in rules if r.passed]
        if met:
            signal = Signal.SELL
            summary = "exit rule(s) met: " + "; ".join(f"{r.code} {r.name}" for r in met) + "."
        else:
            signal = Signal.HOLD
            summary = "holding a position and no exit rule is met."
    else:
        rules = entry_rules(snapshot, latest_volume, config)
        failed = [r for r in rules if not r.passed]
        if not failed:
            signal = Signal.BUY
            summary = f"all {len(rules)} entry rules met."
        else:
            signal = Signal.HOLD
            summary = (f"{len(failed)} of {len(rules)} entry rules not met: "
                       + "; ".join(f"{r.code} {r.name}" for r in failed) + ".")

    return StrategySignal(
        symbol=snapshot.symbol, signal=signal, timestamp=snapshot.as_of,
        source=snapshot.source, kind=snapshot.kind, strategy=config.name,
        has_open_position=has_open_position, data_ok=True, summary=summary,
        rules=tuple(rules), snapshot=snapshot)


# --- The rules ------------------------------------------------------------------

def entry_rules(s: IndicatorSnapshot, latest_volume: float,
                config: StrategyConfig = DEFAULT_CONFIG) -> list[RuleCheck]:
    """E1-E5. A BUY needs every one of these to pass."""
    needed_volume = s.average_volume_20 * config.volume_multiplier
    return [
        RuleCheck("E1", "Trend up", s.sma_20 > s.sma_50,
                  f"SMA 20 ({s.sma_20:.2f}) > SMA 50 ({s.sma_50:.2f})"),
        RuleCheck("E2", "Above VWAP", s.close > s.vwap,
                  f"close ({s.close:.2f}) > VWAP ({s.vwap:.2f})"),
        RuleCheck("E3", "Above SMA 20", s.close > s.sma_20,
                  f"close ({s.close:.2f}) > SMA 20 ({s.sma_20:.2f})"),
        RuleCheck("E4", "Healthy momentum",
                  config.rsi_entry_min <= s.rsi_14 <= config.rsi_entry_max,
                  f"{config.rsi_entry_min:g} <= RSI 14 ({s.rsi_14:.1f}) "
                  f"<= {config.rsi_entry_max:g}"),
        RuleCheck("E5", "Volume confirms", latest_volume >= needed_volume,
                  f"latest volume ({latest_volume:,.0f}) >= average "
                  f"({s.average_volume_20:,.0f}) x {config.volume_multiplier:g}"),
    ]


def exit_rules(s: IndicatorSnapshot,
               config: StrategyConfig = DEFAULT_CONFIG) -> list[RuleCheck]:
    """X1-X3. A SELL needs any one of these to pass."""
    return [
        RuleCheck("X1", "Trend down", s.sma_20 < s.sma_50,
                  f"SMA 20 ({s.sma_20:.2f}) < SMA 50 ({s.sma_50:.2f})"),
        RuleCheck("X2", "Below VWAP", s.close < s.vwap,
                  f"close ({s.close:.2f}) < VWAP ({s.vwap:.2f})"),
        RuleCheck("X3", "Overbought", s.rsi_14 >= config.rsi_exit,
                  f"RSI 14 ({s.rsi_14:.1f}) >= {config.rsi_exit:g}"),
    ]


# --- Helpers ----------------------------------------------------------------------

def _check_common_arguments(kind, has_open_position, config) -> None:
    if not isinstance(kind, DataKind):
        raise StrategyError("expected_kind must be DataKind.HISTORICAL or DataKind.LIVE.")
    if not isinstance(has_open_position, bool):
        raise StrategyError(f"has_open_position must be True or False (got {has_open_position!r}).")
    if not isinstance(config, StrategyConfig):
        raise StrategyError("config must be a StrategyConfig.")


def _live_age_problem(as_of: datetime, config: StrategyConfig,
                      now: datetime | None) -> str | None:
    now = now or datetime.now(timezone.utc)
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise StrategyError("now must be a datetime with a time zone.")
    age = (now - as_of).total_seconds()
    if age < 0:
        return (f"Live data is time-stamped {-age:.0f}s in the future; "
                "the clock or the data is wrong.")
    if age > config.live_max_age_seconds:
        return (f"Live data is {age:.0f}s old; the limit is "
                f"{config.live_max_age_seconds:g}s.")
    return None


def _data_problem(symbol, timestamp, source, kind, has_open_position, config,
                  reason, snapshot=None) -> StrategySignal:
    """A data problem always means HOLD - never BUY or SELL on bad data."""
    return StrategySignal(
        symbol=symbol, signal=Signal.HOLD, timestamp=timestamp, source=source,
        kind=kind, strategy=config.name, has_open_position=has_open_position,
        data_ok=False, summary=f"data problem - {reason}",
        rules=(RuleCheck("D1", "Usable data", False, reason),), snapshot=snapshot)
