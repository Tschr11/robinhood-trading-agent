"""
Tests for the rules-based Strategy Engine (src/strategy.py).

Run from the project root with:
    python -m unittest discover tests -v

Most tests build an IndicatorSnapshot by hand so each rule can be tested on
its own, right at its boundary. Integration tests then run locally generated
candles (tests/market_fixtures.py) through the full path.

Baseline snapshot used below (every entry rule passes, no exit rule passes):
    close 105, SMA 20 103, SMA 50 100, VWAP 104, RSI 60,
    average volume 1,000, latest volume 1,200
"""

import ast
import dataclasses
import math
import os
import pathlib
import tempfile
import unittest
from datetime import timedelta
from unittest import mock

from config import settings
from src import journal, strategy
from src.market_data import (CSVHistoricalProvider, DataKind,
                             IndicatorSnapshot, MarketDataSet)
from src.paper_trader import PaperTrader
from src.strategy import (DEFAULT_CONFIG, Signal, StrategyConfig, StrategyError,
                          StrategySignal, evaluate, evaluate_snapshot)
from tests.market_fixtures import (FIVE_MINUTES, MARKET_OPEN,
                                   candles_from_closes, write_csv,
                                   zigzag_closes)

NAN, INF = math.nan, math.inf
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

BASE = dict(symbol="SPY", kind=DataKind.HISTORICAL, source="fixture",
            as_of=MARKET_OPEN, close=105.0, sma_20=103.0, sma_50=100.0,
            rsi_14=60.0, vwap=104.0, average_volume_20=1000.0)


def snapshot(**changes) -> IndicatorSnapshot:
    values = dict(BASE)
    values.update(changes)
    return IndicatorSnapshot(**values)


def check(snap=None, latest_volume=1200.0, holding=False, config=DEFAULT_CONFIG):
    return evaluate_snapshot(snap or snapshot(), latest_volume,
                             has_open_position=holding, config=config)


def rule(result: StrategySignal, code: str):
    return next(r for r in result.rules if r.code == code)


def fixture_dataset(closes=None, kind=DataKind.HISTORICAL, count=60):
    closes = closes or zigzag_closes(count)
    return MarketDataSet("SPY", kind, "fixture", tuple(candles_from_closes(closes)))


# --- Entry rules (no open position) ---------------------------------------------

class EntryRuleTests(unittest.TestCase):
    def test_all_entry_rules_met_gives_buy(self):
        result = check()
        self.assertEqual(result.signal, Signal.BUY)
        self.assertEqual([r.code for r in result.rules], ["E1", "E2", "E3", "E4", "E5"])
        self.assertTrue(all(r.passed for r in result.rules))
        self.assertIn("all 5 entry rules met", result.summary)

    def test_each_failed_entry_rule_alone_gives_hold(self):
        cases = {
            "E1": dict(sma_20=99.0, close=105.0),        # SMA 20 below SMA 50
            "E2": dict(vwap=106.0),                      # close below VWAP
            "E3": dict(sma_20=104.0, vwap=103.0, close=103.5),  # close below SMA 20
            "E4": dict(rsi_14=45.0),                     # momentum too weak
        }
        for code, changes in cases.items():
            with self.subTest(rule=code):
                result = check(snapshot(**changes))
                self.assertEqual(result.signal, Signal.HOLD)
                failed = [r.code for r in result.rules if not r.passed]
                self.assertEqual(failed, [code])
                self.assertIn(f"1 of 5 entry rules not met: {code}", result.summary)
        with self.subTest(rule="E5"):
            result = check(latest_volume=999.0)          # volume below average
            self.assertEqual(result.signal, Signal.HOLD)
            self.assertEqual([r.code for r in result.rules if not r.passed], ["E5"])

    def test_rsi_too_high_blocks_entry(self):
        result = check(snapshot(rsi_14=72.0))
        self.assertEqual(result.signal, Signal.HOLD)
        self.assertFalse(rule(result, "E4").passed)

    def test_entry_boundaries(self):
        cases = [  # (changes, latest_volume, expected signal)
            (dict(sma_20=100.0), 1200.0, Signal.HOLD),   # SMA 20 == SMA 50: not "above"
            (dict(vwap=105.0), 1200.0, Signal.HOLD),     # close == VWAP
            (dict(sma_20=105.0), 1200.0, Signal.HOLD),   # close == SMA 20
            (dict(rsi_14=50.0), 1200.0, Signal.BUY),     # RSI exactly at minimum
            (dict(rsi_14=70.0), 1200.0, Signal.BUY),     # RSI exactly at maximum
            (dict(rsi_14=49.99), 1200.0, Signal.HOLD),
            (dict(rsi_14=70.01), 1200.0, Signal.HOLD),
            (dict(), 1000.0, Signal.BUY),                # volume exactly = average
        ]
        for changes, volume, expected in cases:
            with self.subTest(changes=changes, volume=volume):
                self.assertEqual(check(snapshot(**changes), volume).signal, expected)

    def test_several_failures_are_all_listed(self):
        # close 105 is still above SMA 20 (99), so E2 and E3 pass.
        result = check(snapshot(sma_20=99.0, rsi_14=30.0), latest_volume=10.0)
        self.assertEqual(result.signal, Signal.HOLD)
        self.assertIn("3 of 5 entry rules not met: E1 Trend up; "
                      "E4 Healthy momentum; E5 Volume confirms", result.summary)

    def test_exit_conditions_without_a_position_give_hold_not_sell(self):
        result = check(snapshot(sma_20=95.0, close=90.0, rsi_14=80.0))
        self.assertEqual(result.signal, Signal.HOLD)
        self.assertNotIn(Signal.SELL, [result.signal])


# --- Exit rules (position open) --------------------------------------------------

class ExitRuleTests(unittest.TestCase):
    def test_no_exit_rule_met_gives_hold(self):
        result = check(holding=True)
        self.assertEqual(result.signal, Signal.HOLD)
        self.assertEqual([r.code for r in result.rules], ["X1", "X2", "X3"])
        self.assertIn("no exit rule is met", result.summary)

    def test_each_exit_rule_alone_gives_sell(self):
        cases = {
            "X1": dict(sma_20=99.0),                     # trend turned down
            "X2": dict(vwap=106.0),                      # fell below VWAP
            "X3": dict(rsi_14=80.0),                     # overbought
        }
        for code, changes in cases.items():
            with self.subTest(rule=code):
                result = check(snapshot(**changes), holding=True)
                self.assertEqual(result.signal, Signal.SELL)
                self.assertEqual([r.code for r in result.rules if r.passed], [code])
                self.assertIn(f"exit rule(s) met: {code}", result.summary)

    def test_several_exit_rules_are_all_listed(self):
        result = check(snapshot(sma_20=99.0, vwap=106.0), holding=True)
        self.assertEqual(result.signal, Signal.SELL)
        self.assertIn("X1 Trend down; X2 Below VWAP", result.summary)

    def test_exit_boundaries(self):
        cases = [
            (dict(sma_20=100.0), Signal.HOLD),           # SMA 20 == SMA 50: not "below"
            (dict(vwap=105.0), Signal.HOLD),             # close == VWAP
            (dict(rsi_14=75.0), Signal.SELL),            # RSI exactly at exit level
            (dict(rsi_14=74.99), Signal.HOLD),
        ]
        for changes, expected in cases:
            with self.subTest(changes=changes):
                self.assertEqual(check(snapshot(**changes), holding=True).signal, expected)

    def test_entry_conditions_while_holding_give_hold_not_another_buy(self):
        result = check(holding=True)                     # baseline = perfect entry
        self.assertEqual(result.signal, Signal.HOLD)

    def test_low_volume_does_not_force_an_exit(self):
        self.assertEqual(check(latest_volume=0.0, holding=True).signal, Signal.HOLD)


# --- Required fields and explanations ------------------------------------------------

class SignalContentTests(unittest.TestCase):
    def test_every_signal_has_the_required_fields(self):
        for holding in [False, True]:
            with self.subTest(holding=holding):
                result = check(holding=holding)
                self.assertEqual(result.symbol, "SPY")
                self.assertIsInstance(result.signal, Signal)
                self.assertEqual(result.timestamp, MARKET_OPEN)
                self.assertEqual(result.source, "fixture")
                self.assertEqual(result.kind, DataKind.HISTORICAL)
                self.assertEqual(result.strategy, settings.STRATEGY_NAME)
                self.assertTrue(result.rules)
                self.assertTrue(all(r.detail for r in result.rules))

    def test_explanation_shows_numbers_and_met_or_not_met(self):
        text = check(snapshot(rsi_14=45.0)).explain()
        self.assertIn("SPY: HOLD", text)
        self.assertIn("historical from fixture", text)
        self.assertIn("[met    ] E1 Trend up: SMA 20 (103.00) > SMA 50 (100.00)", text)
        self.assertIn("[not met] E4 Healthy momentum: 50 <= RSI 14 (45.0) <= 70", text)

    def test_historical_signals_carry_a_research_only_notice(self):
        self.assertIn("not a live trading signal", check().explain())
        live = check(snapshot(kind=DataKind.LIVE))
        self.assertNotIn("not a live trading signal", live.explain())

    def test_signals_cannot_be_changed(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            check().signal = Signal.BUY

    def test_same_input_always_gives_same_output(self):
        data = fixture_dataset()
        first = evaluate(data, expected_kind=DataKind.HISTORICAL, has_open_position=False)
        for _ in range(5):
            self.assertEqual(evaluate(data, expected_kind=DataKind.HISTORICAL,
                                      has_open_position=False), first)


# --- Missing, insufficient and invalid data ------------------------------------------

class DataSafetyTests(unittest.TestCase):
    def test_insufficient_candles_gives_hold_with_reason(self):
        data = fixture_dataset(count=49)
        result = evaluate(data, expected_kind=DataKind.HISTORICAL, has_open_position=False)
        self.assertEqual(result.signal, Signal.HOLD)
        self.assertFalse(result.data_ok)
        self.assertIn("need 50 candles, have 49", result.summary)
        self.assertEqual(result.timestamp, data.latest.timestamp)
        self.assertEqual(result.source, "fixture")
        self.assertIsNone(result.snapshot)

    def test_insufficient_data_never_sells_either(self):
        result = evaluate(fixture_dataset(count=10), expected_kind=DataKind.HISTORICAL,
                          has_open_position=True)
        self.assertEqual(result.signal, Signal.HOLD)

    def test_invalid_indicator_values_give_hold(self):
        cases = [dict(close=NAN), dict(sma_20=INF), dict(sma_50=-1.0), dict(vwap=0.0),
                 dict(rsi_14=NAN), dict(rsi_14=101.0), dict(rsi_14=-1.0),
                 dict(average_volume_20=None), dict(close="105")]
        for changes in cases:
            for holding in [False, True]:
                with self.subTest(changes=changes, holding=holding):
                    result = check(snapshot(**changes), holding=holding)
                    self.assertEqual(result.signal, Signal.HOLD)
                    self.assertFalse(result.data_ok)
                    self.assertIn("Invalid indicator values", result.summary)

    def test_invalid_latest_volume_gives_hold(self):
        for bad in [NAN, -1.0, None, "1200", True]:
            with self.subTest(latest_volume=bad):
                result = check(latest_volume=bad)
                self.assertEqual(result.signal, Signal.HOLD)
                self.assertFalse(result.data_ok)

    def test_missing_data_is_refused(self):
        for bad in [None, [], {"SPY": 500.0}, candles_from_closes(zigzag_closes(60))]:
            with self.subTest(dataset=type(bad).__name__):
                with self.assertRaises(StrategyError):
                    evaluate(bad, expected_kind=DataKind.HISTORICAL,
                             has_open_position=False)
        with self.assertRaises(StrategyError):
            evaluate_snapshot(None, 1000.0, has_open_position=False)

    def test_has_open_position_must_be_true_or_false(self):
        for bad in [None, 1, "yes"]:
            with self.subTest(has_open_position=bad):
                with self.assertRaises(StrategyError):
                    check(holding=bad)


# --- Historical vs live ---------------------------------------------------------------

class HistoricalVersusLiveTests(unittest.TestCase):
    def test_historical_data_can_never_be_used_as_live(self):
        with self.assertRaises(StrategyError) as caught:
            evaluate(fixture_dataset(), expected_kind=DataKind.LIVE,
                     has_open_position=False)
        self.assertIn("is historical, but live data was expected", str(caught.exception))

    def test_live_data_is_not_accepted_as_historical_either(self):
        with self.assertRaises(StrategyError):
            evaluate(fixture_dataset(kind=DataKind.LIVE),
                     expected_kind=DataKind.HISTORICAL, has_open_position=False)

    def test_expected_kind_is_required(self):
        with self.assertRaises(TypeError):
            evaluate(fixture_dataset(), has_open_position=False)
        with self.assertRaises(StrategyError):
            evaluate(fixture_dataset(), expected_kind="live", has_open_position=False)

    def test_fresh_live_data_is_evaluated(self):
        data = fixture_dataset(kind=DataKind.LIVE)
        now = data.latest.timestamp + timedelta(seconds=30)
        result = evaluate(data, expected_kind=DataKind.LIVE,
                          has_open_position=False, now=now)
        self.assertTrue(result.data_ok)
        self.assertEqual(result.signal, Signal.BUY)
        self.assertEqual(result.kind, DataKind.LIVE)

    def test_stale_live_data_gives_hold(self):
        data = fixture_dataset(kind=DataKind.LIVE)
        now = data.latest.timestamp + timedelta(seconds=121)
        result = evaluate(data, expected_kind=DataKind.LIVE,
                          has_open_position=False, now=now)
        self.assertEqual(result.signal, Signal.HOLD)
        self.assertIn("Live data is 121s old; the limit is 120s", result.summary)

    def test_live_data_from_the_future_gives_hold(self):
        data = fixture_dataset(kind=DataKind.LIVE)
        now = data.latest.timestamp - timedelta(minutes=5)
        result = evaluate(data, expected_kind=DataKind.LIVE,
                          has_open_position=True, now=now)
        self.assertEqual(result.signal, Signal.HOLD)
        self.assertIn("in the future", result.summary)

    def test_now_must_include_a_time_zone(self):
        data = fixture_dataset(kind=DataKind.LIVE)
        with self.assertRaises(StrategyError):
            evaluate(data, expected_kind=DataKind.LIVE, has_open_position=False,
                     now=data.latest.timestamp.replace(tzinfo=None))

    def test_historical_data_is_not_age_checked(self):
        result = evaluate(fixture_dataset(), expected_kind=DataKind.HISTORICAL,
                          has_open_position=False)
        self.assertTrue(result.data_ok)


# --- Configuration ---------------------------------------------------------------------

class ConfigTests(unittest.TestCase):
    def test_defaults_come_from_settings(self):
        self.assertEqual(DEFAULT_CONFIG.rsi_entry_min, settings.RSI_ENTRY_MIN)
        self.assertEqual(DEFAULT_CONFIG.rsi_entry_max, settings.RSI_ENTRY_MAX)
        self.assertEqual(DEFAULT_CONFIG.rsi_exit, settings.RSI_EXIT)
        self.assertEqual(DEFAULT_CONFIG.volume_multiplier, settings.VOLUME_MULTIPLIER)
        self.assertEqual(DEFAULT_CONFIG.live_max_age_seconds,
                         settings.LIVE_DATA_MAX_AGE_SECONDS)

    def test_custom_thresholds_change_the_outcome(self):
        strict = StrategyConfig(volume_multiplier=1.5)   # needs 1,500
        self.assertEqual(check(config=strict).signal, Signal.HOLD)
        early_exit = StrategyConfig(rsi_exit=60.0)
        self.assertEqual(check(holding=True, config=early_exit).signal, Signal.SELL)

    def test_invalid_configuration_is_refused(self):
        bad_configs = [dict(rsi_entry_min=70.0, rsi_entry_max=50.0),
                       dict(rsi_entry_min=-1.0), dict(rsi_entry_max=101.0),
                       dict(rsi_exit=0.0), dict(rsi_exit=150.0),
                       dict(volume_multiplier=0.0), dict(volume_multiplier=NAN),
                       dict(live_max_age_seconds=0), dict(rsi_exit="75"),
                       dict(name="")]
        for kwargs in bad_configs:
            with self.subTest(**{k: repr(v) for k, v in kwargs.items()}):
                with self.assertRaises(StrategyError):
                    StrategyConfig(**kwargs)


# --- Full path with locally generated candles ------------------------------------------

class IntegrationTests(unittest.TestCase):
    def test_rising_fixture_gives_buy(self):
        result = evaluate(fixture_dataset(), expected_kind=DataKind.HISTORICAL,
                          has_open_position=False)
        self.assertEqual(result.signal, Signal.BUY)
        self.assertEqual(result.snapshot.sma_50, 118.0)

    def test_falling_fixture_gives_sell_when_holding(self):
        falling = [300 - c for c in zigzag_closes(60)]
        result = evaluate(fixture_dataset(falling), expected_kind=DataKind.HISTORICAL,
                          has_open_position=True)
        self.assertEqual(result.signal, Signal.SELL)
        self.assertEqual([r.code for r in result.rules if r.passed], ["X1", "X2"])

    def test_main_prints_signals_and_never_trades(self):
        from src import main
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "market")
            write_csv(folder, "SPY", candles_from_closes(zigzag_closes(60)))
            trader = PaperTrader(db_path=os.path.join(tmp, "a.db"),
                                 journal_file=os.path.join(tmp, "j.csv"))
            self.addCleanup(trader.close)
            with mock.patch.object(journal, "log_decision") as log, \
                 mock.patch("builtins.print"):
                signals = main.run_once(trader, provider=CSVHistoricalProvider(folder))
            self.assertEqual([s.signal for s in signals], [Signal.BUY])   # SPY
            actions = [call.args[2] for call in log.call_args_list]
            self.assertEqual(actions, ["SIGNAL ONLY", "SKIPPED"])        # QQQ has no file
            self.assertEqual(trader.cash, 25.00)                          # untouched
            self.assertEqual(trader.positions, {})
            self.assertEqual(trader.store.transactions(), [])


# --- What the strategy is allowed to depend on -----------------------------------------

class IsolationTests(unittest.TestCase):
    def test_strategy_never_imports_trading_or_ai_code(self):
        tree = ast.parse((PROJECT_ROOT / "src" / "strategy.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        self.assertEqual(imported, {"math", "dataclasses", "datetime", "enum",
                                    "config", "src.market_data"})

    def test_strategy_has_no_order_functions(self):
        names = {name for name in dir(strategy) if not name.startswith("_")}
        for forbidden in ["buy", "sell", "order", "execute", "trade"]:
            self.assertFalse(any(name.lower() == forbidden for name in names))


if __name__ == "__main__":
    unittest.main()
