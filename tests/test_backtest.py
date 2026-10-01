"""
Tests for the historical Backtesting Engine (src/backtest.py).

Run from the project root with:
    python -m unittest discover tests -v

Most tests use a SCRIPTED strategy that says BUY/SELL on chosen candles, so
fills, costs and exits can be checked to the cent. Look-ahead tests use the
real strategy. All candles are generated locally.

"Clean" settings used by many tests: $1,000 start, no costs, no end-of-day
flattening, decisions from the first candle. At $100 with a 1% stop, risk
sizing allows 20 shares but cash allows 10, so a buy is 10 shares.
"""

import ast
import math
import pathlib
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

from src import strategy
from src.backtest import (DISCLAIMER, END_OF_DATA, LABEL, SESSION_END,
                          STOP_LOSS, STRATEGY_EXIT, TAKE_PROFIT, BacktestConfig,
                          BacktestError, BacktestTrade, compute_metrics,
                          run_backtest)
from src.market_data import (Candle, DataKind, InsufficientDataError,
                             MarketDataError, MarketDataSet)
from src.strategy import Signal
from tests.market_fixtures import (FIVE_MINUTES, MARKET_OPEN,
                                   candles_from_closes, zigzag_closes)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
BUY, SELL = Signal.BUY, Signal.SELL

CLEAN = BacktestConfig(starting_capital=1000.0, slippage_pct=0.0,
                       commission_per_trade=0.0, flatten_end_of_day=False,
                       warmup_candles=1)


def bar(i, o=100.0, h=None, l=None, c=None, volume=1000.0, day=0):
    """Candle number i (5 minutes apart) on day `day`. Defaults: flat at $100."""
    c = o if c is None else c
    h = max(o, c) if h is None else h
    l = min(o, c) if l is None else l
    ts = MARKET_OPEN + timedelta(days=day) + i * FIVE_MINUTES
    return Candle(ts, o, h, l, c, volume)


def data(candles, kind=DataKind.HISTORICAL):
    return MarketDataSet("SPY", kind, "fixture", tuple(candles))


class Scripted:
    """A fake strategy: says `plan[i]` at candle i, HOLD otherwise. Records calls."""
    name = "scripted"

    def __init__(self, candles, plan):
        self.index = {c.timestamp: i for i, c in enumerate(candles)}
        self.plan = plan
        self.calls = []

    def __call__(self, view, *, expected_kind, has_open_position):
        i = self.index[view.latest.timestamp]
        self.calls.append(SimpleNamespace(i=i, size=len(view), view=view,
                                          kind=expected_kind, holding=has_open_position))
        return SimpleNamespace(signal=self.plan.get(i, Signal.HOLD))


def run(candles, plan, config=CLEAN):
    script = Scripted(candles, plan)
    return run_backtest(data(candles), config, strategy_fn=script), script


def two_day_zigzag():
    day1 = candles_from_closes(zigzag_closes(78))
    day2 = candles_from_closes([300 - c for c in zigzag_closes(78)],
                               start=MARKET_OPEN + timedelta(days=1))
    return day1 + day2


# --- Chronological execution ---------------------------------------------------------

class ChronologyTests(unittest.TestCase):
    def test_strategy_is_asked_once_per_candle_in_time_order(self):
        candles = [bar(i) for i in range(6)]
        _, script = run(candles, {})
        self.assertEqual([call.i for call in script.calls], [0, 1, 2, 3, 4, 5])

    def test_strategy_only_ever_sees_candles_up_to_the_current_one(self):
        candles = [bar(i) for i in range(6)]
        _, script = run(candles, {})
        for call in script.calls:
            self.assertEqual(call.size, call.i + 1)
            self.assertEqual(call.view.latest.timestamp, candles[call.i].timestamp)
            self.assertTrue(all(c.timestamp <= candles[call.i].timestamp
                                for c in call.view.candles))
            self.assertEqual(call.kind, DataKind.HISTORICAL)

    def test_warmup_and_lookback_limit_what_the_strategy_sees(self):
        candles = [bar(i) for i in range(10)]
        config = BacktestConfig(warmup_candles=4, lookback_candles=5,
                                flatten_end_of_day=False)
        _, script = run(candles, {}, config)
        self.assertEqual(script.calls[0].i, 3)                 # 4th candle
        self.assertEqual(max(call.size for call in script.calls), 5)

    def test_buy_fills_at_the_next_open_not_any_close(self):
        # Signal close 100, next open 101, next close 101.8 (all different).
        candles = [bar(0, c=100.0), bar(1, o=101.0, c=101.8), bar(2, o=101.8)]
        result, _ = run(candles, {0: BUY})
        trade = result.trades[0]
        self.assertEqual(trade.signal_time, candles[0].timestamp)
        self.assertEqual(trade.entry_time, candles[1].timestamp)
        self.assertEqual(trade.entry_price, 101.0)

    def test_sell_fills_at_the_next_open_not_any_close(self):
        candles = [bar(0), bar(1), bar(2, c=100.5), bar(3, o=100.4, c=100.9), bar(4)]
        result, _ = run(candles, {0: BUY, 2: SELL})
        self.assertEqual(result.trades[0].exit_price, 100.4)

    def test_strategy_is_told_whether_a_position_is_open(self):
        candles = [bar(i) for i in range(4)]
        _, script = run(candles, {0: BUY})
        self.assertEqual([call.holding for call in script.calls],
                         [False, True, True, True])

    def test_trades_are_in_order_and_never_overlap(self):
        result = run_backtest(data(two_day_zigzag()))
        self.assertGreater(len(result.trades), 1)
        for trade in result.trades:
            self.assertLess(trade.signal_time, trade.entry_time)
            self.assertLessEqual(trade.entry_time, trade.exit_time)
        for earlier, later in zip(result.trades, result.trades[1:]):
            self.assertLessEqual(earlier.exit_time, later.entry_time)

    def test_only_one_position_at_a_time(self):
        candles = [bar(i) for i in range(5)]
        result, _ = run(candles, {0: BUY, 1: BUY, 2: BUY})
        self.assertEqual(len(result.trades), 1)

    def test_signal_on_the_last_candle_is_never_filled(self):
        candles = [bar(i) for i in range(3)]
        result, _ = run(candles, {2: BUY})
        self.assertEqual(result.trades, ())
        self.assertEqual(result.expired_orders[0][1], "BUY")


# --- No look-ahead bias ------------------------------------------------------------------

class NoLookAheadTests(unittest.TestCase):
    def record(self, candles):
        """Run the REAL strategy, recording every decision it makes."""
        decisions = []

        def recording(view, **kwargs):
            result = strategy.evaluate(view, **kwargs)
            decisions.append((view.latest.timestamp, result.signal))
            return result
        result = run_backtest(data(candles), strategy_fn=recording)
        return result, decisions

    def test_changing_the_future_does_not_change_the_past(self):
        original = two_day_zigzag()
        cut = 55                       # 14:05 on day 1, while trades are happening
        crash = [Candle(c.timestamp, 50.0, 50.0, 40.0, 41.0, 9_999_999.0)
                 for c in original[cut + 1:]]
        changed = original[:cut + 1] + crash
        result_a, decisions_a = self.record(original)
        result_b, decisions_b = self.record(changed)
        cutoff = original[cut].timestamp

        self.assertEqual([d for d in decisions_a if d[0] <= cutoff],
                         [d for d in decisions_b if d[0] <= cutoff])
        self.assertEqual([t for t in result_a.trades if t.exit_time <= cutoff],
                         [t for t in result_b.trades if t.exit_time <= cutoff])
        self.assertEqual(result_a.equity_curve[:cut + 1], result_b.equity_curve[:cut + 1])
        # Make sure the test means something: the future change DID matter.
        self.assertNotEqual(result_a.equity_curve, result_b.equity_curve)
        self.assertNotEqual(result_a.trades, result_b.trades)
        self.assertNotEqual(decisions_a, decisions_b)

    def test_end_of_day_exit_ignores_next_day_prices(self):
        day1 = [bar(i) for i in range(3)]
        calm = [bar(i, day=1) for i in range(2)]
        wild = [bar(i, o=10.0, h=500.0, l=5.0, c=400.0, day=1) for i in range(2)]
        config = BacktestConfig(starting_capital=1000.0, slippage_pct=0.0,
                                commission_per_trade=0.0, warmup_candles=1)
        result_calm, _ = run(day1 + calm, {0: BUY}, config)
        result_wild, _ = run(day1 + wild, {0: BUY}, config)
        self.assertEqual(result_calm.trades[0], result_wild.trades[0])
        self.assertEqual(result_calm.trades[0].exit_reason, SESSION_END)

    def test_stop_is_not_checked_against_the_signal_candle(self):
        # The signal candle itself dips to 50, but we weren't in yet.
        candles = [bar(0, o=100.0, l=50.0, c=100.0), bar(1), bar(2)]
        result, _ = run(candles, {0: BUY})
        self.assertEqual(result.trades[0].exit_reason, END_OF_DATA)


# --- Stop-loss, take-profit and other exits ----------------------------------------------

class ExitTests(unittest.TestCase):
    # Entry: 10 shares at $100 -> stop $99, target $102.

    def exit_with(self, exit_bar, plan=None):
        candles = [bar(0), bar(1), exit_bar, bar(3)]
        result, _ = run(candles, plan or {0: BUY})
        return result.trades[0]

    def test_stop_loss_inside_the_candle(self):
        trade = self.exit_with(bar(2, o=100.0, l=98.5, c=99.5))
        self.assertEqual(trade.exit_reason, STOP_LOSS)
        self.assertAlmostEqual(trade.exit_price, 99.0)
        self.assertAlmostEqual(trade.realized_pnl, -10.0)

    def test_gap_below_stop_fills_at_the_worse_open(self):
        trade = self.exit_with(bar(2, o=97.0, c=97.0))
        self.assertEqual(trade.exit_reason, STOP_LOSS)
        self.assertAlmostEqual(trade.exit_price, 97.0)
        self.assertAlmostEqual(trade.realized_pnl, -30.0)

    def test_take_profit_inside_the_candle(self):
        trade = self.exit_with(bar(2, o=100.0, h=102.5, c=101.0))
        self.assertEqual(trade.exit_reason, TAKE_PROFIT)
        self.assertAlmostEqual(trade.exit_price, 102.0)
        self.assertAlmostEqual(trade.realized_pnl, 20.0)

    def test_gap_above_target_fills_at_the_open(self):
        trade = self.exit_with(bar(2, o=103.0, c=103.0))
        self.assertEqual(trade.exit_reason, TAKE_PROFIT)
        self.assertAlmostEqual(trade.exit_price, 103.0)

    def test_stop_and_target_in_one_candle_assumes_the_stop(self):
        trade = self.exit_with(bar(2, o=100.0, h=103.0, l=98.0, c=101.0))
        self.assertEqual(trade.exit_reason, STOP_LOSS)
        self.assertAlmostEqual(trade.realized_pnl, -10.0)

    def test_stop_can_trigger_on_the_entry_candle(self):
        candles = [bar(0), bar(1, o=100.0, l=98.0, c=100.0), bar(2)]
        result, _ = run(candles, {0: BUY})
        self.assertEqual(result.trades[0].exit_reason, STOP_LOSS)
        self.assertEqual(result.trades[0].exit_time, candles[1].timestamp)

    def test_strategy_sell_exits_at_the_next_open(self):
        candles = [bar(0), bar(1), bar(2, c=100.5), bar(3, o=100.5), bar(4)]
        result, _ = run(candles, {0: BUY, 2: SELL})
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, STRATEGY_EXIT)
        self.assertEqual(trade.exit_time, candles[3].timestamp)
        self.assertAlmostEqual(trade.exit_price, 100.5)

    def test_position_closed_at_end_of_data(self):
        candles = [bar(0), bar(1), bar(2, c=101.0)]
        result, _ = run(candles, {0: BUY})
        self.assertEqual(result.trades[0].exit_reason, END_OF_DATA)
        self.assertAlmostEqual(result.trades[0].exit_price, 101.0)
        self.assertAlmostEqual(result.metrics.final_equity, 1010.0)

    def test_day_trading_flattens_at_the_session_close(self):
        candles = [bar(0), bar(1), bar(2, c=100.7), bar(0, day=1), bar(1, day=1)]
        config = BacktestConfig(starting_capital=1000.0, slippage_pct=0.0,
                                commission_per_trade=0.0, warmup_candles=1)
        result, _ = run(candles, {0: BUY}, config)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, SESSION_END)
        self.assertEqual(trade.exit_time, candles[2].timestamp)
        self.assertAlmostEqual(trade.exit_price, 100.7)

    def test_without_flattening_a_position_can_be_held_overnight(self):
        candles = [bar(0), bar(1), bar(2), bar(0, day=1), bar(1, day=1, c=101.0)]
        result, _ = run(candles, {0: BUY})                   # CLEAN: no flattening
        self.assertEqual(result.trades[0].exit_reason, END_OF_DATA)
        self.assertEqual(result.trades[0].exit_time, candles[4].timestamp)

    def test_buy_at_the_last_close_of_a_day_is_not_filled_next_morning(self):
        candles = [bar(0), bar(1), bar(0, day=1), bar(1, day=1)]
        config = BacktestConfig(starting_capital=1000.0, slippage_pct=0.0,
                                commission_per_trade=0.0, warmup_candles=1)
        result, _ = run(candles, {1: BUY}, config)
        self.assertEqual(result.trades, ())
        self.assertEqual(result.expired_orders,
                         ((candles[1].timestamp, "BUY", "day ended; no overnight orders"),))


# --- Transaction costs ------------------------------------------------------------------

class CostTests(unittest.TestCase):
    candles = [bar(0), bar(1, o=100.0), bar(2, c=101.0)]

    def test_no_costs_baseline(self):
        result, _ = run(self.candles, {0: BUY})
        trade = result.trades[0]
        self.assertEqual(trade.shares, 10)
        self.assertAlmostEqual(trade.realized_pnl, 10 * (101.0 - 100.0))
        self.assertEqual(trade.commission, 0.0)

    def test_slippage_worsens_both_entry_and_exit(self):
        config = BacktestConfig(starting_capital=1000.0, slippage_pct=0.01,
                                commission_per_trade=0.0, flatten_end_of_day=False,
                                warmup_candles=1)
        trade = run(self.candles, {0: BUY}, config)[0].trades[0]
        self.assertAlmostEqual(trade.entry_price, 101.0)      # 100 x 1.01
        self.assertAlmostEqual(trade.exit_price, 99.99)       # 101 x 0.99
        self.assertAlmostEqual(trade.realized_pnl,
                               trade.shares * (99.99 - 101.0))

    def test_commission_is_charged_on_entry_and_exit(self):
        config = BacktestConfig(starting_capital=1000.0, slippage_pct=0.0,
                                commission_per_trade=1.0, flatten_end_of_day=False,
                                warmup_candles=1)
        result = run(self.candles, {0: BUY}, config)[0]
        trade = result.trades[0]
        self.assertAlmostEqual(trade.shares, 9.99)            # (1000 - 1) / 100
        self.assertEqual(trade.commission, 2.0)
        self.assertAlmostEqual(trade.realized_pnl, 9.99 * 1.0 - 2.0)
        self.assertAlmostEqual(result.metrics.final_equity, 1000 + trade.realized_pnl)
        self.assertEqual(result.metrics.total_commission, 2.0)

    def test_higher_costs_mean_lower_return(self):
        returns = []
        for slippage, commission in [(0.0, 0.0), (0.001, 0.0), (0.001, 1.0)]:
            config = BacktestConfig(starting_capital=1000.0, slippage_pct=slippage,
                                    commission_per_trade=commission,
                                    flatten_end_of_day=False, warmup_candles=1)
            returns.append(run(self.candles, {0: BUY}, config)[0].metrics.total_return_pct)
        self.assertGreater(returns[0], returns[1])
        self.assertGreater(returns[1], returns[2])

    def test_take_profit_is_a_limit_order_without_slippage(self):
        config = BacktestConfig(starting_capital=1000.0, slippage_pct=0.001,
                                commission_per_trade=0.0, flatten_end_of_day=False,
                                warmup_candles=1)
        candles = [bar(0), bar(1), bar(2, o=100.5, h=110.0, c=105.0), bar(3)]
        trade = run(candles, {0: BUY}, config)[0].trades[0]
        self.assertEqual(trade.exit_reason, TAKE_PROFIT)
        self.assertAlmostEqual(trade.exit_price, trade.entry_price * 1.02)

    def test_stop_loss_exit_pays_slippage(self):
        config = BacktestConfig(starting_capital=1000.0, slippage_pct=0.001,
                                commission_per_trade=0.0, flatten_end_of_day=False,
                                warmup_candles=1)
        candles = [bar(0), bar(1), bar(2, o=100.0, l=90.0, c=95.0), bar(3)]
        trade = run(candles, {0: BUY}, config)[0].trades[0]
        stop = trade.entry_price * 0.99
        self.assertAlmostEqual(trade.exit_price, stop * 0.999)


# --- Position sizing and risk -------------------------------------------------------------

class SizingTests(unittest.TestCase):
    candles = [bar(0), bar(1), bar(2)]

    def test_risk_based_sizing(self):
        config = BacktestConfig(starting_capital=1000.0, stop_loss_pct=0.05,
                                slippage_pct=0.0, flatten_end_of_day=False,
                                warmup_candles=1)
        trade = run(self.candles, {0: BUY}, config)[0].trades[0]
        self.assertAlmostEqual(trade.shares, 4.0)              # $20 risk / $5 stop
        risk = trade.shares * trade.entry_price * 0.05
        self.assertLessEqual(risk, 0.02 * 1000 + 1e-9)

    def test_max_position_pct_caps_size(self):
        config = BacktestConfig(starting_capital=1000.0, max_position_pct=0.5,
                                slippage_pct=0.0, flatten_end_of_day=False,
                                warmup_candles=1)
        trade = run(self.candles, {0: BUY}, config)[0].trades[0]
        self.assertAlmostEqual(trade.shares, 5.0)

    def test_whole_shares_only(self):
        config = BacktestConfig(starting_capital=150.0, risk_per_trade_pct=1.0,
                                allow_fractional_shares=False, slippage_pct=0.0,
                                flatten_end_of_day=False, warmup_candles=1)
        self.assertEqual(run(self.candles, {0: BUY}, config)[0].trades[0].shares, 1)

    def test_too_small_to_buy_is_rejected_and_recorded(self):
        config = BacktestConfig(starting_capital=50.0, allow_fractional_shares=False,
                                slippage_pct=0.0, flatten_end_of_day=False,
                                warmup_candles=1)
        result = run(self.candles, {0: BUY}, config)[0]
        self.assertEqual(result.trades, ())
        self.assertIn("rounds to zero", result.rejected_entries[0][1])
        self.assertEqual(result.metrics.final_equity, 50.0)

    def test_daily_loss_limit_blocks_entries_until_the_next_day(self):
        # Daily limit 1% of $1,000 = $10. A $10 stop-loss uses it all up.
        config = BacktestConfig(starting_capital=1000.0, daily_loss_pct=0.01,
                                slippage_pct=0.0, flatten_end_of_day=False,
                                warmup_candles=1)
        candles = [bar(0), bar(1), bar(2, l=98.0, c=99.0), bar(3, o=99.0), bar(4, o=99.0),
                   bar(0, o=99.0, day=1), bar(1, o=99.0, day=1), bar(2, o=99.0, day=1)]
        result, _ = run(candles, {0: BUY, 3: BUY, 5: BUY}, config)
        self.assertEqual(len(result.trades), 2)
        self.assertIn("Daily loss limit", result.rejected_entries[0][1])
        self.assertEqual(result.trades[1].entry_time, candles[6].timestamp)  # next day


# --- Metrics ----------------------------------------------------------------------------------

def fake_trade(pnl, commission=0.0):
    return BacktestTrade("SPY", MARKET_OPEN, MARKET_OPEN, 100.0, MARKET_OPEN, 100.0,
                         1.0, commission, pnl, STRATEGY_EXIT)


class MetricTests(unittest.TestCase):
    def test_hand_calculated_metrics(self):
        trades = [fake_trade(p) for p in [10.0, -5.0, 20.0, -10.0, 0.0]]
        m = compute_metrics(trades, [100, 120, 90, 130, 117], 100.0)
        self.assertEqual(m.number_of_trades, 5)
        self.assertEqual((m.wins, m.losses), (2, 2))          # $0 is neither
        self.assertAlmostEqual(m.win_rate, 0.4)
        self.assertAlmostEqual(m.average_win, 15.0)
        self.assertAlmostEqual(m.average_loss, -7.5)
        self.assertAlmostEqual(m.profit_factor, 2.0)          # 30 / 15
        self.assertAlmostEqual(m.total_return_pct, 0.17)      # 100 -> 117
        self.assertAlmostEqual(m.max_drawdown, 30.0)          # 120 -> 90
        self.assertAlmostEqual(m.max_drawdown_pct, 0.25)

    def test_drawdown_counts_from_the_starting_capital(self):
        m = compute_metrics([], [80.0, 90.0], 100.0)
        self.assertAlmostEqual(m.max_drawdown, 20.0)
        self.assertAlmostEqual(m.max_drawdown_pct, 0.2)

    def test_no_trades_gives_undefined_not_made_up_values(self):
        m = compute_metrics([], [100.0, 100.0], 100.0)
        self.assertEqual(m.number_of_trades, 0)
        self.assertIsNone(m.win_rate)
        self.assertIsNone(m.average_win)
        self.assertIsNone(m.average_loss)
        self.assertIsNone(m.profit_factor)
        self.assertEqual(m.total_return_pct, 0.0)

    def test_profit_factor_undefined_without_losses(self):
        m = compute_metrics([fake_trade(5.0)], [105.0], 100.0)
        self.assertIsNone(m.profit_factor)
        self.assertIsNone(m.average_loss)
        self.assertEqual(m.win_rate, 1.0)

    def test_metrics_match_the_trades_of_a_real_run(self):
        result = run_backtest(data(two_day_zigzag()))
        m = result.metrics
        pnls = [t.realized_pnl for t in result.trades]
        self.assertEqual(m.number_of_trades, len(pnls))
        self.assertAlmostEqual(m.final_equity, 25.0 + sum(pnls))
        self.assertAlmostEqual(m.final_equity, result.equity_curve[-1][1])


# --- Invalid and insufficient data ------------------------------------------------------------

class DataSafetyTests(unittest.TestCase):
    def test_live_data_is_refused(self):
        with self.assertRaises(MarketDataError) as caught:
            run_backtest(data([bar(i) for i in range(60)], kind=DataKind.LIVE))
        self.assertIn("historical data is required", str(caught.exception))

    def test_insufficient_data(self):
        with self.assertRaises(InsufficientDataError) as caught:
            run_backtest(data([bar(i) for i in range(50)]))
        self.assertIn("needs at least 51 candles", str(caught.exception))

    def test_raw_or_invalid_candles_are_refused(self):
        good = [bar(i) for i in range(60)]
        with self.assertRaises(MarketDataError):
            run_backtest(good)                                  # not a dataset
        with self.assertRaises(MarketDataError):
            run_backtest(data(good[:30] + [bar(30, o=math.nan)] + good[31:]))
        with self.assertRaises(MarketDataError):
            run_backtest(data(good + [good[-1]]))               # duplicate timestamp

    def test_invalid_settings_are_refused(self):
        bad = [dict(starting_capital=0), dict(starting_capital=math.nan),
               dict(risk_per_trade_pct=0), dict(max_position_pct=1.5),
               dict(stop_loss_pct=1.0), dict(take_profit_pct=-0.01),
               dict(commission_per_trade=-1), dict(slippage_pct=0.5),
               dict(flatten_end_of_day="yes"), dict(warmup_candles=0),
               dict(warmup_candles=60, lookback_candles=50)]
        for kwargs in bad:
            with self.subTest(**{k: repr(v) for k, v in kwargs.items()}):
                with self.assertRaises(BacktestError):
                    BacktestConfig(**kwargs)

    def test_strategy_must_return_a_signal(self):
        with self.assertRaises(BacktestError):
            run_backtest(data([bar(i) for i in range(3)]), CLEAN,
                         strategy_fn=lambda view, **kw: "BUY")


# --- Separation from paper trading, labels and honesty ----------------------------------------

class SeparationTests(unittest.TestCase):
    def test_backtester_does_not_import_the_paper_trading_engine(self):
        tree = ast.parse((PROJECT_ROOT / "src" / "backtest.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        for forbidden in ["src.paper_trader", "src.storage", "src.journal", "sqlite3"]:
            self.assertNotIn(forbidden, imported)

    def test_running_a_backtest_never_touches_the_account_or_journal(self):
        boom = AssertionError("the backtest must not touch this")
        with mock.patch("src.storage.AccountStore.__init__", side_effect=boom), \
             mock.patch("src.paper_trader.PaperTrader.__init__", side_effect=boom), \
             mock.patch("src.journal.log_decision", side_effect=boom):
            run_backtest(data(two_day_zigzag()))

    def test_results_are_labelled_as_a_backtest_with_a_disclaimer(self):
        result = run_backtest(data(two_day_zigzag()))
        self.assertEqual(result.label, LABEL)
        self.assertEqual(result.strategy, "trend_vwap_v1")
        report = result.report()
        self.assertTrue(report.startswith("BACKTEST - historical simulation"))
        self.assertIn(DISCLAIMER, report)
        self.assertIn("NOT evidence that the strategy will be profitable", report)
        self.assertNotIn("guarantee", report.lower())

    def test_report_shows_undefined_metrics_honestly(self):
        result = run_backtest(data([bar(i) for i in range(60)]))
        report = result.report()
        self.assertEqual(result.metrics.number_of_trades, 0)
        self.assertIn("Win rate          n/a", report)
        self.assertIn("Profit factor     n/a", report)

    def test_same_data_same_result(self):
        candles = two_day_zigzag()
        self.assertEqual(run_backtest(data(candles)), run_backtest(data(candles)))


if __name__ == "__main__":
    unittest.main()
