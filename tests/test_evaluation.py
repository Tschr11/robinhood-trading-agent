"""
Tests for the historical evaluation workflow (src/evaluation.py).

Run from the project root with:
    python -m unittest discover tests -v

All data is generated locally and written to temporary folders.
Helper data: 4 trading days of 78 five-minute candles (9:30-15:55, -05:00),
2026-01-05 .. 2026-01-08. Days alternate between rising and falling.
"""

import hashlib
import json
import math
import os
import tempfile
import unittest
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

from config import settings
from src import evaluation, strategy
from src.backtest import (DISCLAIMER, END_OF_DATA, TAKE_PROFIT, BacktestConfig,
                          compute_metrics, first_tradable_index,
                          run_backtest)
from src.evaluation import (IN_SAMPLE, OUT_OF_SAMPLE, DatasetSpec,
                            EvaluationError, EvaluationPlan, buy_and_hold,
                            evaluate, load_plan, plan_from_dict, save_report,
                            split_candles)
from src.market_data import Candle, InsufficientDataError
from src.strategy import Signal, StrategyConfig
from tests.market_fixtures import (MARKET_OPEN, candles_from_closes,
                                   write_csv, zigzag_closes)
from tests.test_backtest import HISTORY, Scripted, bar, data, run

FIXED_NOW = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
SPLIT = date(2026, 1, 7)


def four_days():
    candles = []
    for day in range(4):
        closes = zigzag_closes(78, start=100.0 + day)
        if day % 2:
            closes = [250.0 - c for c in closes]
        candles += candles_from_closes(closes, start=MARKET_OPEN + timedelta(days=day))
    return candles


def _snapshot(path):
    """File contents (or None if missing) - to prove a file was not touched."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def period_history(candles, period):
    """The earlier candles the evaluation uses as warm-up for `period`."""
    return [c for c in candles if c.timestamp < period[0].timestamp][-500:]


class EvaluationTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.market = os.path.join(self.tmp.name, "market")
        self.reports = os.path.join(self.tmp.name, "reports")
        self.log = os.path.join(self.tmp.name, "exposure", "oos_exposure_log.jsonl")
        self.candles = four_days()
        self.csv = write_csv(self.market, "SPY", self.candles)
        # Safety net: no test may create or change the REAL exposure log.
        self.real_log_before = _snapshot(settings.OOS_EXPOSURE_LOG)
        self.addCleanup(lambda: self.assertEqual(
            _snapshot(settings.OOS_EXPOSURE_LOG), self.real_log_before,
            "a test touched the real exposure log"))

    def save(self, report, **kwargs):
        kwargs.setdefault("exposure_log", self.log)
        return save_report(report, kwargs.pop("reports_dir", self.reports), **kwargs)

    def plan(self, **dataset_changes):
        spec = dict(symbol="SPY", folder=self.market, split_date=SPLIT.isoformat())
        spec.update(dataset_changes)
        return plan_from_dict({"name": "test_plan", "datasets": [spec]})

    def run_eval(self, plan=None, **kwargs):
        kwargs.setdefault("exposure_log", self.log)
        kwargs.setdefault("now", FIXED_NOW)
        return evaluate(plan or self.plan(), **kwargs)


# --- Plans --------------------------------------------------------------------------------

class PlanTests(unittest.TestCase):
    base = {"name": "p1", "datasets": [{"symbol": "spy", "split_date": "2026-01-07"}]}

    def with_changes(self, top=None, dataset=None):
        raw = json.loads(json.dumps(self.base))
        raw.update(top or {})
        raw["datasets"][0].update(dataset or {})
        return raw

    def test_valid_plan(self):
        plan = plan_from_dict(self.with_changes(
            {"backtest": {"slippage_pct": 0.001}},
            {"start": "2026-01-05", "end": "2026-01-08", "folder": "x"}))
        spec = plan.datasets[0]
        self.assertEqual(spec.symbol, "SPY")
        self.assertEqual((spec.start, spec.split_date, spec.end),
                         (date(2026, 1, 5), date(2026, 1, 7), date(2026, 1, 8)))
        self.assertEqual(plan.backtest.slippage_pct, 0.001)
        self.assertEqual(plan_from_dict(self.base).datasets[0].folder,
                         settings.MARKET_DATA_DIR)

    def test_plans_cannot_set_strategy_thresholds(self):
        with self.assertRaises(EvaluationError) as caught:
            plan_from_dict(self.with_changes({"strategy": {"rsi_entry_min": 40}}))
        self.assertIn("cannot set strategy thresholds", str(caught.exception))

    def test_invalid_plans_are_rejected(self):
        bad = [
            self.with_changes({"extra": 1}),
            self.with_changes({"name": "../escape"}),
            self.with_changes({"name": ""}),
            {"name": "p", "datasets": []},
            self.with_changes(dataset={"unknown": 1}),
            self.with_changes(dataset={"split_date": "07/01/2026"}),
            self.with_changes(dataset={"split_date": 20260107}),
            self.with_changes(dataset={"symbol": "../etc"}),
            self.with_changes(dataset={"start": "2026-01-08", "end": "2026-01-05"}),
            self.with_changes(dataset={"start": "2026-01-07"}),       # split == start
            self.with_changes(dataset={"end": "2026-01-06"}),         # split after end
            self.with_changes({"backtest": {"no_such_setting": 1}}),
            self.with_changes({"backtest": {"slippage_pct": 0.5}}),
            self.with_changes({"backtest": {"warmup_candles": 10}}),
            self.with_changes({"backtest": {"lookback_candles": 49}}),
            self.with_changes({"backtest": []}),
            {"name": "p", "datasets": [{"symbol": "SPY"}]},           # no split_date
            ["not", "a", "dict"],
        ]
        for raw in bad:
            with self.subTest(raw=str(raw)[:80]):
                with self.assertRaises(EvaluationError):
                    plan_from_dict(raw)

    def test_load_plan_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "plan.json")
            with open(good, "w") as f:
                json.dump(self.base, f)
            self.assertEqual(load_plan(good).name, "p1")
            broken = os.path.join(tmp, "broken.json")
            with open(broken, "w") as f:
                f.write("{not json")
            with self.assertRaises(EvaluationError):
                load_plan(broken)
            with self.assertRaises(EvaluationError):
                load_plan(os.path.join(tmp, "missing.json"))

    def test_example_plan_in_repository_is_valid(self):
        plan = load_plan("plans/example_plan.json")
        self.assertEqual([d.symbol for d in plan.datasets], ["SPY", "QQQ"])

    def test_plan_fingerprint_is_stable_and_sensitive(self):
        a = plan_from_dict(self.base)
        self.assertEqual(a.fingerprint(), plan_from_dict(self.base).fingerprint())
        for changed in [self.with_changes(dataset={"split_date": "2026-01-06"}),
                        self.with_changes({"backtest": {"commission_per_trade": 1.0}}),
                        self.with_changes({"name": "p2"})]:
            self.assertNotEqual(a.fingerprint(), plan_from_dict(changed).fingerprint())


# --- In-sample / out-of-sample separation -----------------------------------------------------

class DateSeparationTests(EvaluationTestCase):
    def test_split_is_clean_and_complete(self):
        spec = DatasetSpec("SPY", self.market, SPLIT)
        inside, outside = split_candles(self.candles, spec)
        self.assertTrue(all(c.timestamp.date() < SPLIT for c in inside))
        self.assertTrue(all(c.timestamp.date() >= SPLIT for c in outside))
        self.assertLess(inside[-1].timestamp, outside[0].timestamp)
        self.assertEqual(inside + outside, self.candles)                 # nothing lost
        self.assertFalse({c.timestamp.date() for c in inside}
                         & {c.timestamp.date() for c in outside})        # no shared day

    def test_start_and_end_are_inclusive(self):
        spec = DatasetSpec("SPY", self.market, SPLIT, start=date(2026, 1, 6),
                           end=date(2026, 1, 7))
        inside, outside = split_candles(self.candles, spec)
        self.assertEqual({c.timestamp.date() for c in inside}, {date(2026, 1, 6)})
        self.assertEqual({c.timestamp.date() for c in outside}, {date(2026, 1, 7)})
        self.assertEqual(len(inside) + len(outside), 156)

    def test_each_backtest_only_receives_its_own_period(self):
        seen = []

        def spy_backtest(dataset, *args, **kwargs):
            seen.append({"dataset": dataset, "trade_start": kwargs.get("trade_start")})
            return run_backtest(dataset, *args, **kwargs)
        with mock.patch.object(evaluation, "run_backtest", side_effect=spy_backtest):
            report = self.run_eval()
        in_call, out_call = seen
        in_data, out_data = in_call["dataset"], out_call["dataset"]
        self.assertTrue(in_data.source.endswith("[in-sample]"))
        self.assertTrue(out_data.source.endswith("[out-of-sample]"))
        # In-sample: nothing on or after the split, not even as history.
        self.assertLess(max(c.timestamp.date() for c in in_data.candles), SPLIT)
        # Out-of-sample: trading starts exactly at the split; anything earlier is
        # in-sample history used only for indicator warm-up.
        first_oos = next(c for c in self.candles if c.timestamp.date() >= SPLIT)
        self.assertEqual(out_call["trade_start"], first_oos.timestamp)
        history = [c for c in out_data.candles if c.timestamp < out_call["trade_start"]]
        self.assertTrue(all(c.timestamp.date() < SPLIT for c in history))
        periods = report.datasets[0].periods
        self.assertEqual([p.period for p in periods], [IN_SAMPLE, OUT_OF_SAMPLE])
        self.assertLess(periods[0].last_candle, periods[1].first_candle)

    def test_in_sample_only_never_touches_out_of_sample_data(self):
        seen = []

        def spy_backtest(dataset, *args, **kwargs):
            seen.append(dataset)
            return run_backtest(dataset, *args, **kwargs)
        with mock.patch.object(evaluation, "run_backtest", side_effect=spy_backtest), \
             mock.patch.object(evaluation, "buy_and_hold",
                               wraps=evaluation.buy_and_hold) as benchmark:
            report = self.run_eval(in_sample_only=True)
        self.assertEqual(len(seen), 1)
        self.assertTrue(all(c.timestamp.date() < SPLIT for c in seen[0].candles))
        for call in benchmark.call_args_list:
            self.assertTrue(all(c.timestamp.date() < SPLIT for c in call.args[0]))
        oos = report.datasets[0].periods[1]
        self.assertEqual(oos.status, "not evaluated")
        self.assertIsNone(oos.strategy)
        self.assertEqual(report.summary()[OUT_OF_SAMPLE]["periods_evaluated"], 0)

    def test_too_short_period_is_skipped_with_reason(self):
        # In-sample is day 1 only (78 candles) with no earlier history, but 80
        # warm-up candles are required. Out-of-sample can borrow day 1 as history.
        report = self.run_eval(plan_from_dict({"name": "short", "datasets": [{
            "symbol": "SPY", "folder": self.market, "split_date": "2026-01-06"}],
            "backtest": {"warmup_candles": 80, "lookback_candles": 500}}))
        in_s, out_s = report.datasets[0].periods
        self.assertEqual(in_s.status, "skipped")
        self.assertIn("no tradable candle", in_s.reason)
        self.assertIn("has 0 earlier candle(s) and 78 in the period", in_s.reason)
        self.assertEqual(out_s.status, "ok")
        self.assertEqual(out_s.history_candles, 78)

    def test_dataset_errors_are_reported_and_others_still_run(self):
        os.makedirs(os.path.join(self.tmp.name, "bad"))
        with open(os.path.join(self.tmp.name, "bad", "QQQ.csv"), "w") as f:
            f.write("timestamp,open,high,low,close,volume\n"
                    "2026-01-05T09:30:00-05:00,100,99,98,100,1000\n")       # high < open
        plan = plan_from_dict({"name": "mixed", "datasets": [
            {"symbol": "QQQ", "folder": os.path.join(self.tmp.name, "bad"),
             "split_date": "2026-01-07"},
            {"symbol": "IWM", "folder": self.market, "split_date": "2026-01-07"},
            {"symbol": "SPY", "folder": self.market, "split_date": "2026-01-07"}]})
        report = self.run_eval(plan)
        qqq, iwm, spy = report.datasets
        self.assertEqual(qqq.status, "error")
        self.assertIn("high (99.0) is below open", qqq.error)
        self.assertIsNotNone(qqq.sha256)
        self.assertEqual(iwm.status, "error")
        self.assertIsNone(iwm.sha256)
        self.assertEqual(spy.status, "ok")
        self.assertIn("ERROR:", report.to_text())


# --- The strategy is evaluated as-is ------------------------------------------------------------

class NoOptimizationTests(EvaluationTestCase):
    def test_current_thresholds_are_used_and_left_unchanged(self):
        before = strategy.DEFAULT_CONFIG
        report = self.run_eval()
        self.assertIs(strategy.DEFAULT_CONFIG, before)
        self.assertEqual(report.strategy_config, asdict(StrategyConfig()))
        self.assertEqual(report.strategy_name, settings.STRATEGY_NAME)

    def test_every_backtest_uses_the_default_strategy_config(self):
        with mock.patch.object(evaluation, "run_backtest", wraps=run_backtest) as spy:
            self.run_eval()
        for call in spy.call_args_list:
            self.assertIs(call.kwargs["strategy_config"], strategy.DEFAULT_CONFIG)

    def test_no_parameter_search_exists(self):
        names = [n.lower() for n in dir(evaluation)]
        for word in ["optimi", "tune", "grid", "search", "best_params"]:
            self.assertFalse(any(word in n for n in names), word)


# --- Reproducibility --------------------------------------------------------------------------

class ReproducibilityTests(EvaluationTestCase):
    def test_same_plan_and_data_give_identical_reports(self):
        first = self.run_eval()
        second = self.run_eval()
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.to_text(), second.to_text())

    def test_only_the_generation_time_differs_between_runs(self):
        first = self.run_eval().to_dict()
        later = self.run_eval(now=FIXED_NOW + timedelta(days=3)).to_dict()
        diff = {k for k in first if first[k] != later[k]}
        self.assertEqual(diff, {"generated_at"})

    def test_file_fingerprints_are_recorded(self):
        report = self.run_eval()
        with open(self.csv, "rb") as f:
            expected = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(report.datasets[0].sha256, expected)
        self.assertEqual(report.plan_fingerprint, self.plan().fingerprint())
        record = report.to_dict()
        self.assertEqual(record["plan"]["backtest"], asdict(BacktestConfig()))

    def test_changing_the_data_changes_the_fingerprint(self):
        before = self.run_eval()
        changed = list(self.candles)
        changed[-1] = Candle(changed[-1].timestamp, changed[-1].open,
                             changed[-1].high + 0.01, changed[-1].low,
                             changed[-1].close, changed[-1].volume)
        write_csv(self.market, "SPY", changed)
        after = self.run_eval()
        self.assertNotEqual(before.datasets[0].sha256, after.datasets[0].sha256)
        self.assertNotEqual(before.data_fingerprint, after.data_fingerprint)


# --- Buy-and-hold benchmark ---------------------------------------------------------------------

class BenchmarkTests(unittest.TestCase):
    # 50 history candles, then the period: the first decision is at the close
    # of period candle 0, so buy-and-hold buys at candle 1's open ($100).
    period = [bar(0), bar(1, o=100.0, c=120.0), bar(2, o=120.0, c=90.0),
              bar(3, o=90.0, c=110.0)]
    candles = HISTORY + period
    start = period[0].timestamp

    def config(self, **changes):
        values = dict(starting_capital=1000.0, slippage_pct=0.0, commission_per_trade=0.0)
        values.update(changes)
        return BacktestConfig(**values)

    def bh(self, config, candles=None):
        return buy_and_hold(candles or self.candles, config, trade_start=self.start)

    def test_hand_calculated_without_costs(self):
        b = self.bh(self.config())
        self.assertEqual(b.entry_time, self.period[1].timestamp)
        self.assertEqual(b.exit_time, self.period[3].timestamp)
        self.assertEqual(b.shares, 10)
        self.assertAlmostEqual(b.final_equity, 1100.0)
        self.assertAlmostEqual(b.total_return_pct, 0.10)
        # equity 1200 (close 120) -> 900 (close 90) -> 1100
        self.assertAlmostEqual(b.max_drawdown, 300.0)
        self.assertAlmostEqual(b.max_drawdown_pct, 0.25)
        self.assertAlmostEqual(b.largest_pct_drawdown, 0.25)
        self.assertAlmostEqual(b.largest_pct_drawdown_dollars, 300.0)
        self.assertEqual((b.total_commission, b.total_slippage), (0.0, 0.0))

    def test_hand_calculated_with_costs(self):
        b = self.bh(self.config(slippage_pct=0.01, commission_per_trade=1.0))
        shares = 9.891                       # (1000 - 1) / 101 = 9.89108.. -> 4 decimals
        self.assertAlmostEqual(b.entry_price, 101.0)
        self.assertAlmostEqual(b.exit_price, 108.9)
        self.assertAlmostEqual(b.shares, shares)
        cash = 1000 - shares * 101.0 - 1.0
        self.assertAlmostEqual(b.final_equity, cash + shares * 108.9 - 1.0)
        self.assertEqual(b.total_commission, 2.0)
        self.assertAlmostEqual(b.total_slippage, shares * 1.0 + shares * 1.1)

    def test_whole_shares_and_too_little_capital(self):
        whole = self.bh(self.config(starting_capital=150.0, allow_fractional_shares=False))
        self.assertEqual(whole.shares, 1)
        nothing = self.bh(self.config(starting_capital=50.0, allow_fractional_shares=False,
                                      commission_per_trade=1.0))
        self.assertEqual(nothing.shares, 0)
        self.assertEqual(nothing.total_return_pct, 0.0)
        self.assertEqual(nothing.total_commission, 0.0)

    def test_needs_candles_after_warmup(self):
        with self.assertRaises(InsufficientDataError):
            self.bh(self.config(), HISTORY + self.period[:1])      # no candle to fill on
        with self.assertRaises(InsufficientDataError):
            buy_and_hold([bar(i) for i in range(50)], self.config())


class SameDatesTests(EvaluationTestCase):
    def test_benchmark_and_strategy_cover_the_same_tradable_window(self):
        report = self.run_eval()
        spec = report.plan.datasets[0]
        for period, candles in zip(report.datasets[0].periods,
                                   split_candles(self.candles, spec)):
            combined = period_history(self.candles, candles) + candles
            first = first_tradable_index(combined, report.plan.backtest.warmup_candles,
                                         candles[0].timestamp)
            self.assertEqual(period.benchmark.entry_time, combined[first].timestamp)
            self.assertEqual(period.benchmark.exit_time, candles[-1].timestamp)
            self.assertEqual(period.first_candle, candles[0].timestamp)
            self.assertEqual(period.last_candle, candles[-1].timestamp)

    def test_strategy_trades_fall_inside_the_benchmark_window(self):
        spec = DatasetSpec("SPY", self.market, SPLIT)
        for candles in split_candles(self.candles, spec):
            combined = period_history(self.candles, candles) + candles
            start = candles[0].timestamp
            result = run_backtest(data(combined), trade_start=start)
            window = buy_and_hold(combined, BacktestConfig(), trade_start=start)
            for trade in result.trades:
                self.assertGreaterEqual(trade.entry_time, window.entry_time)
                self.assertLessEqual(trade.exit_time, window.exit_time)


# --- Commission and slippage totals in the backtester --------------------------------------------

class CostTotalTests(unittest.TestCase):
    def config(self, slippage):
        return BacktestConfig(starting_capital=1000.0, slippage_pct=slippage,
                              commission_per_trade=1.0, flatten_end_of_day=False)

    def test_market_exit_slippage_is_counted_on_both_sides(self):
        candles = [bar(0), bar(1, o=100.0), bar(2, c=101.0)]
        result, _ = run(candles, {0: Signal.BUY}, self.config(0.01))
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, END_OF_DATA)
        expected = trade.shares * 100.0 * 0.01 + trade.shares * 101.0 * 0.01
        self.assertAlmostEqual(trade.slippage_cost, expected)
        self.assertAlmostEqual(result.metrics.total_slippage, expected)
        self.assertEqual(result.metrics.total_commission, 2.0)

    def test_take_profit_adds_no_exit_slippage(self):
        candles = [bar(0), bar(1), bar(2, o=100.5, h=110.0, c=105.0), bar(3)]
        result, _ = run(candles, {0: Signal.BUY}, self.config(0.01))
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, TAKE_PROFIT)
        self.assertAlmostEqual(trade.slippage_cost, trade.shares * 100.0 * 0.01)

    def test_no_slippage_means_zero_cost(self):
        candles = [bar(0), bar(1), bar(2)]
        result, _ = run(candles, {0: Signal.BUY}, self.config(0.0))
        self.assertEqual(result.metrics.total_slippage, 0.0)


# --- Reports -------------------------------------------------------------------------------------

class ReportTests(EvaluationTestCase):
    REQUIRED = ["Total return", "Max $ drawdown", "Max % drawdown", "Win rate",
                "Profit factor",
                "Number of trades", "Average win", "Average loss", "Commissions",
                "Slippage", "Buy-and-hold"]

    def test_json_and_text_reports_are_written(self):
        json_path, txt_path = self.save(self.run_eval())
        self.assertTrue(json_path.startswith(self.reports))
        with open(json_path) as f:
            record = json.load(f)
        with open(txt_path) as f:
            text = f.read()
        self.assertEqual(record["kind"], evaluation.REPORT_KIND)
        self.assertEqual(record["disclaimer"], DISCLAIMER)
        self.assertTrue(record["assumptions"])
        self.assertTrue(record["limitations"])
        period = record["datasets"][0]["periods"][1]
        for key in ["total_return_pct", "max_drawdown", "win_rate", "profit_factor",
                    "number_of_trades", "average_win", "average_loss",
                    "total_commission", "total_slippage"]:
            self.assertIn(key, period["strategy"])
        for key in ["total_return_pct", "max_drawdown", "total_commission", "total_slippage"]:
            self.assertIn(key, period["benchmark"])
        for label in self.REQUIRED + ["Assumptions:", "Limitations:", DISCLAIMER]:
            self.assertIn(label, text)
        self.assertTrue(text.startswith("EVALUATION - historical backtests"))

    def test_report_numbers_match_the_backtest(self):
        report = self.run_eval()
        spec = report.plan.datasets[0]
        _, oos = split_candles(self.candles, spec)
        direct = run_backtest(data(period_history(self.candles, oos) + oos),
                              trade_start=oos[0].timestamp).metrics
        stored = report.datasets[0].periods[1].strategy
        self.assertEqual(stored.total_return_pct, direct.total_return_pct)
        self.assertEqual(stored.number_of_trades, direct.number_of_trades)

    def test_reports_are_never_overwritten(self):
        report = self.run_eval()
        first = self.save(report)
        second = self.save(report)
        self.assertNotEqual(first, second)
        self.assertEqual(len(os.listdir(self.reports)), 4)

    def test_reports_cannot_go_next_to_the_paper_account_or_journal(self):
        report = self.run_eval()
        # Point the protected folders at temporary ones, so that even if the
        # guard ever broke, this test could not write into the real data/ or logs/.
        data_dir = os.path.join(self.tmp.name, "data")
        log_dir = os.path.join(self.tmp.name, "logs")
        with mock.patch.multiple(settings, DATA_DIR=data_dir, LOG_DIR=log_dir,
                                 DATABASE_FILE=os.path.join(data_dir, "paper_account.db"),
                                 JOURNAL_FILE=os.path.join(log_dir, "trade_journal.csv")):
            for folder in [data_dir, log_dir, os.path.join(data_dir, "market"),
                           os.path.join(data_dir, "reports")]:
                with self.subTest(folder=folder):
                    with self.assertRaises(EvaluationError):
                        self.save(report, reports_dir=folder)
                    self.assertFalse(os.path.exists(folder))

    def test_real_project_folders_are_protected_too(self):
        report = self.run_eval()
        for folder in [settings.DATA_DIR, settings.LOG_DIR]:
            with self.subTest(folder=folder):
                with mock.patch("os.makedirs", side_effect=AssertionError("must not create")), \
                     mock.patch("builtins.open", side_effect=AssertionError("must not write")):
                    with self.assertRaises(EvaluationError):
                        self.save(report, reports_dir=folder)

    def test_evaluation_never_touches_the_paper_account_or_journal(self):
        boom = AssertionError("evaluation must not touch this")
        with mock.patch("src.storage.AccountStore.__init__", side_effect=boom), \
             mock.patch("src.paper_trader.PaperTrader.__init__", side_effect=boom), \
             mock.patch("src.journal.log_decision", side_effect=boom):
            self.save(self.run_eval())

    def test_in_sample_only_report_says_so(self):
        report = self.run_eval(in_sample_only=True)
        json_path, _ = self.save(report)
        self.assertTrue(json_path.endswith("_in-sample-only.json"))
        self.assertIn("NOT evaluated", report.to_text())

    def test_report_makes_no_profitability_claims(self):
        text = self.run_eval().to_text().lower()
        for phrase in ["guarantee", "will make money", "proven", "risk-free"]:
            self.assertNotIn(phrase, text)
        self.assertIn("not evidence that the strategy will be profitable", text)

    def test_undefined_metrics_show_as_n_a(self):
        report = self.run_eval()
        text = report.to_text()
        self.assertIn("n/a", text)                    # e.g. buy-and-hold win rate
        record = report.to_dict()
        self.assertFalse(any(isinstance(v, float) and math.isnan(v)
                             for v in record["datasets"][0]["periods"][0]["strategy"].values()))

    def test_command_line_runs_a_plan(self):
        plan_path = os.path.join(self.tmp.name, "plan.json")
        with open(plan_path, "w") as f:
            json.dump({"name": "cli", "datasets": [
                {"symbol": "SPY", "folder": self.market, "split_date": "2026-01-07"}]}, f)
        with mock.patch("builtins.print"):
            json_path, txt_path = evaluation.main(
                [plan_path, "--reports-dir", self.reports, "--in-sample-only",
                 "--exposure-log", self.log])
        self.assertTrue(os.path.exists(json_path) and os.path.exists(txt_path))



# --- Out-of-sample exposure tracking --------------------------------------------------------------

class ExposureTrackingTests(EvaluationTestCase):
    def count(self, report, symbol="SPY"):
        return next(d for d in report.datasets if d.spec.symbol == symbol).prior_oos_evaluations

    def test_key_identifies_the_dataset_period(self):
        report = self.run_eval()
        with open(self.csv, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(report.datasets[0].oos_key, {
            "symbol": "SPY", "sha256": sha, "split_date": "2026-01-07",
            "end_date": "2026-01-08"})                  # last candle date (no plan end)
        explicit = self.run_eval(self.plan(end="2026-01-07"))
        self.assertEqual(explicit.datasets[0].oos_key["end_date"], "2026-01-07")

    def test_saved_out_of_sample_evaluations_are_counted(self):
        self.assertEqual(self.count(self.run_eval()), 0)
        self.save(self.run_eval())
        self.save(self.run_eval())
        report = self.run_eval()
        self.assertEqual(self.count(report), 2)
        self.assertIn("Earlier SAVED out-of-sample evaluations of this exact dataset "
                      "period (2026-01-07 to 2026-01-08): 2", report.to_text())
        self.assertIn("each extra look weakens", report.to_text())

    def test_count_survives_name_cost_dataset_and_folder_changes(self):
        self.save(self.run_eval())
        renamed = plan_from_dict({"name": "renamed", "datasets": [
            {"symbol": "SPY", "folder": self.market, "split_date": "2026-01-07"}]})
        costly = plan_from_dict({"name": "test_plan", "datasets": [
            {"symbol": "SPY", "folder": self.market, "split_date": "2026-01-07"}],
            "backtest": {"commission_per_trade": 0.01}})
        write_csv(self.market, "QQQ", self.candles[::-1][::-1])
        bigger = plan_from_dict({"name": "test_plan", "datasets": [
            {"symbol": "SPY", "folder": self.market, "split_date": "2026-01-07"},
            {"symbol": "QQQ", "folder": self.market, "split_date": "2026-01-07"}]})
        explicit_end = self.plan(end="2026-01-08")       # same period, end spelled out
        for label, plan in [("renamed", renamed), ("commission", costly),
                            ("extra dataset", bigger), ("explicit end", explicit_end)]:
            with self.subTest(label):
                self.assertEqual(self.count(self.run_eval(plan)), 1)
        self.assertEqual(self.count(self.run_eval(bigger), "QQQ"), 0)
        # A different report folder does not matter: the log is separate.
        other_reports = os.path.join(self.tmp.name, "elsewhere")
        self.save(self.run_eval(), reports_dir=other_reports)
        self.assertEqual(self.count(self.run_eval()), 2)

    def test_different_dataset_periods_are_counted_separately(self):
        self.save(self.run_eval())
        self.assertEqual(self.count(self.run_eval(self.plan(split_date="2026-01-06"))), 0)
        self.assertEqual(self.count(self.run_eval(self.plan(end="2026-01-07"))), 0)
        changed = list(self.candles)
        changed[-1] = Candle(changed[-1].timestamp, changed[-1].open, changed[-1].high + 0.01,
                             changed[-1].low, changed[-1].close, changed[-1].volume)
        write_csv(self.market, "SPY", changed)                           # new fingerprint
        self.assertEqual(self.count(self.run_eval()), 0)

    def test_unsaved_and_in_sample_only_runs_are_not_counted(self):
        self.run_eval()                                   # evaluated but never saved
        self.save(self.run_eval(in_sample_only=True))     # saved, but no OOS look
        self.assertEqual(self.count(self.run_eval()), 0)
        self.assertFalse(os.path.exists(self.log))

    def test_in_sample_only_still_shows_earlier_looks(self):
        self.save(self.run_eval())
        report = self.run_eval(in_sample_only=True)
        self.assertEqual(self.count(report), 1)

    def test_skipped_or_failed_datasets_are_not_logged(self):
        plan = plan_from_dict({"name": "skips", "datasets": [
            {"symbol": "SPY", "folder": self.market, "split_date": "2026-01-08"},
            {"symbol": "IWM", "folder": self.market, "split_date": "2026-01-07"}],
            "backtest": {"warmup_candles": 400, "lookback_candles": 500}})
        report = self.run_eval(plan)
        self.assertEqual(report.datasets[0].periods[1].status, "skipped")
        self.assertEqual(report.datasets[1].status, "error")
        self.save(report)
        self.assertFalse(os.path.exists(self.log))

    def test_log_is_written_only_after_the_report_files(self):
        with mock.patch.object(evaluation.json, "dump", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.save(self.run_eval())
        self.assertFalse(os.path.exists(self.log))
        self.assertEqual(self.count(self.run_eval()), 0)

    def test_log_failure_after_saving_is_reported_not_hidden(self):
        report = self.run_eval()
        os.makedirs(self.log)                        # a folder where the file should be
        with self.assertRaises(EvaluationError) as caught:
            self.save(report)
        self.assertIn("The report was saved", str(caught.exception))
        with self.assertRaises(EvaluationError) as caught:
            self.run_eval()                          # reading it is refused clearly too
        self.assertIn("is not a file", str(caught.exception))

    def test_damaged_log_is_an_error_not_a_reset(self):
        os.makedirs(os.path.dirname(self.log))
        with open(self.log, "w") as f:
            f.write("{not json\n")
        with self.assertRaises(EvaluationError) as caught:
            self.run_eval()
        self.assertIn("line 1 is unreadable", str(caught.exception))

    def test_log_records_are_complete(self):
        json_path, _ = self.save(self.run_eval())
        with open(self.log) as f:
            records = [json.loads(line) for line in f]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["report"], json_path)
        self.assertEqual(record["plan_name"], "test_plan")
        for field in ["symbol", "sha256", "split_date", "end_date", "recorded_at",
                      "code_fingerprint"]:
            self.assertIn(field, record)

    def test_exposure_log_cannot_go_next_to_the_paper_account_or_journal(self):
        data_dir = os.path.join(self.tmp.name, "data")
        log_dir = os.path.join(self.tmp.name, "logs")
        report = self.run_eval()
        with mock.patch.multiple(settings, DATA_DIR=data_dir, LOG_DIR=log_dir,
                                 DATABASE_FILE=os.path.join(data_dir, "paper_account.db"),
                                 JOURNAL_FILE=os.path.join(log_dir, "trade_journal.csv")):
            for path in [os.path.join(data_dir, "oos.jsonl"),
                         os.path.join(log_dir, "sub", "oos.jsonl")]:
                with self.subTest(path=path):
                    with self.assertRaises(EvaluationError):
                        self.save(report, exposure_log=path)
                    self.assertFalse(os.path.exists(path))
        self.assertFalse(os.path.exists(self.reports))       # refused before writing

    def test_report_explains_what_the_count_can_and_cannot_detect(self):
        report = self.run_eval()
        text = report.to_text()
        for note in evaluation.OOS_COUNT_NOTES:
            self.assertIn(note, text)
        self.assertIn("SAVED", text)
        self.assertIn("cannot detect", text)
        self.assertEqual(report.to_dict()["oos_count_notes"], list(evaluation.OOS_COUNT_NOTES))


# --- Warm-up history in the evaluation ------------------------------------------------------------

class EvaluationWarmupTests(EvaluationTestCase):
    def test_out_of_sample_uses_in_sample_candles_only_as_history(self):
        report = self.run_eval()
        in_s, out_s = report.datasets[0].periods
        self.assertEqual(in_s.history_candles, 0)          # nothing before the file starts
        self.assertEqual(out_s.history_candles, 156)       # the two in-sample days
        self.assertEqual(out_s.first_candle.date(), SPLIT)
        self.assertEqual(out_s.candles, 156)
        _, oos = split_candles(self.candles, report.plan.datasets[0])
        # The benchmark (and the strategy) may first trade the 2nd out-of-sample candle.
        self.assertEqual(out_s.benchmark.entry_time, oos[1].timestamp)

    def test_history_is_limited_to_the_lookback(self):
        plan = plan_from_dict({"name": "short_lookback", "datasets": [
            {"symbol": "SPY", "folder": self.market, "split_date": "2026-01-07"}],
            "backtest": {"lookback_candles": 100, "warmup_candles": 50}})
        self.assertEqual(self.run_eval(plan).datasets[0].periods[1].history_candles, 100)

    def test_no_out_of_sample_trade_or_equity_before_the_split(self):
        results = []

        def spy_backtest(dataset, *args, **kwargs):
            result = run_backtest(dataset, *args, **kwargs)
            results.append(result)
            return result
        with mock.patch.object(evaluation, "run_backtest", side_effect=spy_backtest):
            self.run_eval()
        oos_result = results[1]
        self.assertEqual(oos_result.first_candle.date(), SPLIT)
        self.assertTrue(all(ts.date() >= SPLIT for ts, _ in oos_result.equity_curve))
        self.assertTrue(all(t.signal_time.date() >= SPLIT for t in oos_result.trades))


# --- Benchmark alignment ----------------------------------------------------------------------------

class BenchmarkAlignmentTests(unittest.TestCase):
    def first_fill(self, candles, config, trade_start=None):
        always = lambda view, **kw: SimpleNamespace(signal=Signal.BUY)
        result = run_backtest(data(candles), config, strategy_fn=always,
                              trade_start=trade_start)
        return result.trades[0].entry_time

    def test_benchmark_enters_where_the_strategy_first_can(self):
        candles = [bar(i) for i in range(70)] + [bar(i, day=1) for i in range(70)]
        no_flatten = dict(flatten_end_of_day=False)
        cases = [(BacktestConfig(**no_flatten), None),
                 (BacktestConfig(warmup_candles=60, **no_flatten), None),
                 (BacktestConfig(**no_flatten), candles[70].timestamp),
                 (BacktestConfig(warmup_candles=120, **no_flatten), candles[70].timestamp)]
        for config, start in cases:
            with self.subTest(warmup=config.warmup_candles, trade_start=start):
                self.assertEqual(buy_and_hold(candles, config, trade_start=start).entry_time,
                                 self.first_fill(candles, config, start))

    def test_warmup_below_50_cannot_misalign_the_benchmark(self):
        with self.assertRaises(EvaluationError):
            plan_from_dict({"name": "p", "datasets": [
                {"symbol": "SPY", "split_date": "2026-01-07"}],
                "backtest": {"warmup_candles": 10}})
        # Even a config that bypassed validation enters at candle 50, not 10.
        config = BacktestConfig()
        object.__setattr__(config, "warmup_candles", 10)
        candles = [bar(i) for i in range(70)]
        self.assertEqual(buy_and_hold(candles, config).entry_time, candles[50].timestamp)


# --- Code fingerprint --------------------------------------------------------------------------------

class CodeFingerprintTests(EvaluationTestCase):
    def test_fingerprint_covers_the_evaluation_source_files(self):
        fingerprint = self.run_eval().code_fingerprint
        files = fingerprint["files"]
        for path in ["src/strategy.py", "src/backtest.py", "src/evaluation.py",
                     "src/risk_manager.py", "src/market_data/indicators.py",
                     "src/market_data/validation.py", "src/market_data/providers.py"]:
            with self.subTest(path=path):
                with open(path, "rb") as f:
                    self.assertEqual(files[path], hashlib.sha256(f.read()).hexdigest())
        self.assertEqual(len(fingerprint["combined"]), 64)
        self.assertIn("source files only", fingerprint["scope"])

    def test_changing_a_source_file_changes_the_fingerprint(self):
        copy_root = os.path.join(self.tmp.name, "copy")
        for rel in evaluation.code_fingerprint()["files"]:
            os.makedirs(os.path.join(copy_root, os.path.dirname(rel)), exist_ok=True)
            with open(rel, "rb") as src, open(os.path.join(copy_root, rel), "wb") as dst:
                dst.write(src.read())
        with mock.patch.object(evaluation, "PROJECT_ROOT", copy_root):
            same = evaluation.code_fingerprint()["combined"]
            with open(os.path.join(copy_root, "src", "strategy.py"), "a") as f:
                f.write("\n# changed\n")
            changed = evaluation.code_fingerprint()["combined"]
        self.assertEqual(same, evaluation.code_fingerprint()["combined"])
        self.assertNotEqual(same, changed)

    def test_report_states_the_fingerprint_scope(self):
        report = self.run_eval()
        self.assertIn("source files only - not the Python version", report.to_text())
        self.assertIn("code_fingerprint", report.to_dict())


# --- Documentation ----------------------------------------------------------------------------------

class DocumentationTests(unittest.TestCase):
    def setUp(self):
        with open("README.md") as f:
            self.readme = f.read()

    def test_reproducibility_statement_mentions_changing_counts(self):
        self.assertNotIn("only the generation time differs.", self.readme)
        self.assertIn("out-of-sample counts", self.readme)

    def test_costs_are_described_as_already_included(self):
        self.assertIn("already included", self.readme)
        self.assertIn("must not be subtracted again", self.readme)

    def test_cost_rows_in_reports_are_marked_as_included(self):
        period = SimpleNamespace(strategy=compute_metrics([], [25.0], 25.0),
                                 benchmark=SimpleNamespace(
                                     total_return_pct=0.0, max_drawdown=0.0,
                                     max_drawdown_pct=0.0, largest_pct_drawdown=0.0,
                                     largest_pct_drawdown_dollars=0.0, shares=0,
                                     total_commission=0.0, total_slippage=0.0))
        text = "\n".join(evaluation._comparison_table(period))
        self.assertIn("Commissions *", text)
        self.assertIn("Already included in Total return", text)

if __name__ == "__main__":
    unittest.main()
