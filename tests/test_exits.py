"""
Tests for automatic stop-loss and take-profit exits (PaperTrader.check_exits).

Run from the project root with:
    python -m unittest discover tests -v

"Standard position" used below: 0.04 SPY bought at $500.00
    stop-loss   = $495.00
    take-profit = $510.00 (default: 2% above entry)
    cost $20.00, so cash left = $5.00
"""

import math
import os
import tempfile
import unittest
from datetime import date
from unittest import mock

from config import settings
from src import journal
from src.paper_trader import PaperTrader
from src.risk_manager import RiskManager

NAN = math.nan
INF = math.inf


class ExitTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.journal_file = os.path.join(self.tmp.name, "journal.csv")
        self.trader = self.make_trader()

    def tearDown(self):
        self.tmp.cleanup()

    def make_trader(self, **kwargs):
        kwargs.setdefault("journal_file", self.journal_file)
        kwargs.setdefault("today", lambda: date(2026, 1, 5))
        # A fresh, throwaway database per trader (never data/paper_account.db)
        self.db_count = getattr(self, "db_count", 0) + 1
        kwargs.setdefault("db_path",
                          os.path.join(self.tmp.name, f"account{self.db_count}.db"))
        trader = PaperTrader(**kwargs)
        self.addCleanup(trader.close)
        return trader

    def open_standard(self, trader=None, **kwargs):
        result = (trader or self.trader).buy("SPY", 0.04, 500.00, 495.00, **kwargs)
        self.assertTrue(result.success, result.reason)
        return result

    def last_row(self):
        return journal.read_journal(self.journal_file)[-1]

    def books(self):
        t = self.trader
        return (t.cash, {s: (p.shares, p.entry_price) for s, p in t.positions.items()},
                t.realized_pnl, t.realized_loss_today)


# --- Take-profit setup when buying -------------------------------------------

class TakeProfitSetupTests(ExitTestCase):
    def test_default_take_profit_is_two_percent_above_entry(self):
        self.open_standard()
        self.assertAlmostEqual(self.trader.positions["SPY"].take_profit_price, 510.00)

    def test_custom_take_profit(self):
        self.open_standard(take_profit_price=520.00)
        self.assertEqual(self.trader.positions["SPY"].take_profit_price, 520.00)

    def test_bad_take_profit_rejects_the_buy(self):
        before = self.books()
        for bad in [500.00, 499.00, 0, -1, NAN, INF, "520", True]:
            with self.subTest(take_profit_price=bad):
                result = self.trader.buy("SPY", 0.04, 500.00, 495.00,
                                         take_profit_price=bad)
                self.assertFalse(result.success)
                self.assertIn("Take-profit price must be", result.reason)
        self.assertEqual(self.books(), before)

    def test_averaging_recalculates_default_take_profit(self):
        trader = self.make_trader(risk_manager=RiskManager(max_open_positions=2))
        trader.buy("SPY", 0.02, 500.00, 495.00)
        trader.buy("SPY", 0.02, 510.00, 505.00)          # average entry $505
        self.assertAlmostEqual(trader.positions["SPY"].take_profit_price, 515.10)


# --- Stop-loss triggers ------------------------------------------------------

class StopLossTests(ExitTestCase):
    def test_price_exactly_at_stop_closes_position(self):
        self.open_standard()
        results = self.trader.check_exits({"SPY": 495.00})
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].action, "STOP-LOSS SELL")
        self.assertEqual(self.trader.positions, {})
        self.assertAlmostEqual(self.trader.cash, 24.80)          # 5 + 0.04 x 495
        self.assertAlmostEqual(self.trader.realized_pnl, -0.20)
        self.assertAlmostEqual(self.trader.realized_loss_today, 0.20)

    def test_gap_below_stop_fills_at_the_supplied_price(self):
        """Price jumped past the stop: we get the WORSE price, not the stop."""
        self.open_standard()
        result = self.trader.check_exits({"SPY": 480.00})[0]
        self.assertEqual(result.price, 480.00)
        self.assertAlmostEqual(result.realized_pnl, -0.80)       # 0.04 x -20
        self.assertAlmostEqual(self.trader.cash, 24.20)
        self.assertAlmostEqual(self.trader.realized_loss_today, 0.80)

    def test_stop_loss_reason_is_journaled(self):
        self.open_standard()
        self.trader.check_exits({"SPY": 494.00})
        row = self.last_row()
        self.assertEqual(row["action"], "STOP-LOSS SELL")
        self.assertIn("Stop-loss hit", row["reason"])
        self.assertIn("$495.00", row["reason"])
        self.assertAlmostEqual(float(row["realized_pnl"]), -0.24)
        self.assertAlmostEqual(float(row["cash_after"]), 24.76)

    def test_stop_loss_works_after_daily_loss_limit_is_reached(self):
        trader = self.make_trader(risk_manager=RiskManager(max_open_positions=2))
        self.assertTrue(trader.buy("SPY", 0.02, 500.00, 495.00).success)  # $10
        self.assertTrue(trader.buy("QQQ", 0.1, 100.00, 95.00).success)    # $10
        self.assertTrue(trader.sell("QQQ", 0.1, 87.50).success)  # -$1.25: limit hit
        blocked = trader.buy("QQQ", 0.01, 100.00, 99.00)
        self.assertFalse(blocked.success)
        self.assertIn("Daily loss limit reached", blocked.reason)
        results = trader.check_exits({"SPY": 490.00})
        self.assertEqual(results[0].action, "STOP-LOSS SELL")
        self.assertEqual(trader.positions, {})
        self.assertAlmostEqual(trader.realized_loss_today, 1.25 + 0.20)


# --- Take-profit triggers ----------------------------------------------------

class TakeProfitTests(ExitTestCase):
    def test_price_exactly_at_target_closes_position(self):
        self.open_standard()
        results = self.trader.check_exits({"SPY": 510.00})
        self.assertEqual(results[0].action, "TAKE-PROFIT SELL")
        self.assertEqual(self.trader.positions, {})
        self.assertAlmostEqual(self.trader.cash, 25.40)
        self.assertAlmostEqual(self.trader.realized_pnl, 0.40)
        self.assertEqual(self.trader.realized_loss_today, 0.0)

    def test_price_above_target_fills_at_the_supplied_price(self):
        self.open_standard()
        result = self.trader.check_exits({"SPY": 530.00})[0]
        self.assertAlmostEqual(result.realized_pnl, 1.20)
        self.assertAlmostEqual(self.trader.cash, 26.20)

    def test_custom_target_is_used(self):
        self.open_standard(take_profit_price=520.00)
        self.assertEqual(self.trader.check_exits({"SPY": 515.00}), [])
        self.assertEqual(self.trader.check_exits({"SPY": 520.00})[0].action,
                         "TAKE-PROFIT SELL")

    def test_take_profit_reason_is_journaled(self):
        self.open_standard()
        self.trader.check_exits({"SPY": 512.00})
        row = self.last_row()
        self.assertEqual(row["action"], "TAKE-PROFIT SELL")
        self.assertIn("Take-profit hit", row["reason"])
        self.assertAlmostEqual(float(row["realized_pnl"]), 0.48)


# --- Prices between the stop and the target ---------------------------------

class NoTriggerTests(ExitTestCase):
    def test_prices_between_stop_and_target_do_nothing(self):
        self.open_standard()
        before = self.books()
        rows_before = len(journal.read_journal(self.journal_file))
        for price in [495.01, 500.00, 505.00, 509.99]:
            with self.subTest(price=price):
                self.assertEqual(self.trader.check_exits({"SPY": price}), [])
        self.assertEqual(self.books(), before)
        self.assertEqual(len(journal.read_journal(self.journal_file)), rows_before)

    def test_no_positions_means_nothing_to_do(self):
        self.assertEqual(self.trader.check_exits({"SPY": 1.00}), [])

    def test_prices_for_symbols_not_held_are_ignored(self):
        self.open_standard()
        self.assertEqual(self.trader.check_exits({"SPY": 500.00, "QQQ": 1.00}), [])

    def test_lowercase_price_keys_are_accepted(self):
        self.open_standard()
        self.assertEqual(self.trader.check_exits({" spy ": 494.00})[0].action,
                         "STOP-LOSS SELL")


# --- Never sell more than held -----------------------------------------------

class NoOversellTests(ExitTestCase):
    def test_exit_sells_exactly_the_shares_held(self):
        self.open_standard()
        self.trader.sell("SPY", 0.01, 500.00)                  # 0.03 left
        result = self.trader.check_exits({"SPY": 494.00})[0]
        self.assertAlmostEqual(result.shares, 0.03)
        self.assertEqual(self.trader.positions, {})

    def test_second_check_after_exit_does_nothing(self):
        self.open_standard()
        self.trader.check_exits({"SPY": 494.00})
        cash = self.trader.cash
        self.assertEqual(self.trader.check_exits({"SPY": 494.00}), [])
        self.assertEqual(self.trader.cash, cash)

    def test_only_the_triggered_position_is_closed(self):
        trader = self.make_trader(risk_manager=RiskManager(max_open_positions=2))
        self.open_standard(trader)
        self.assertTrue(trader.buy("QQQ", 0.01, 400.00, 396.00).success)                # target $408
        results = trader.check_exits({"SPY": 500.00, "QQQ": 410.00})
        self.assertEqual([r.action for r in results], ["TAKE-PROFIT SELL"])
        self.assertEqual(list(trader.positions), ["SPY"])

    def test_two_positions_can_exit_in_one_check(self):
        trader = self.make_trader(risk_manager=RiskManager(max_open_positions=2))
        self.open_standard(trader)
        self.assertTrue(trader.buy("QQQ", 0.01, 400.00, 396.00).success)
        results = trader.check_exits({"SPY": 490.00, "QQQ": 410.00})
        self.assertEqual(sorted(r.action for r in results),
                         ["STOP-LOSS SELL", "TAKE-PROFIT SELL"])
        self.assertEqual(trader.positions, {})


# --- Invalid or missing price data ------------------------------------------

class InvalidPriceDataTests(ExitTestCase):
    def test_bad_prices_skip_the_position_and_change_nothing(self):
        self.open_standard()
        before = self.books()
        bad_inputs = [{}, {"SPY": None}, {"SPY": NAN}, {"SPY": INF},
                      {"SPY": -INF}, {"SPY": 0}, {"SPY": -494.00},
                      {"SPY": "494"}, {"SPY": True}, {"QQQ": 494.00}]
        for prices in bad_inputs:
            with self.subTest(prices=prices):
                results = self.trader.check_exits(prices)
                self.assertEqual(len(results), 1)
                self.assertFalse(results[0].success)
                self.assertEqual(results[0].action, "EXIT CHECK SKIPPED")
                self.assertIn("No valid price for SPY", results[0].reason)
        self.assertEqual(self.books(), before)

    def test_prices_that_are_not_a_dict_are_handled(self):
        self.open_standard()
        before = self.books()
        for bad in [None, 494.00, "SPY=494", [("SPY", 494.00)]]:
            with self.subTest(market_prices=bad):
                results = self.trader.check_exits(bad)
                self.assertEqual(results[0].action, "EXIT CHECK SKIPPED")
        self.assertEqual(self.books(), before)

    def test_skip_is_journaled(self):
        self.open_standard()
        self.trader.check_exits({"SPY": NAN})
        row = self.last_row()
        self.assertEqual(row["action"], "EXIT CHECK SKIPPED")
        self.assertIn("were not checked", row["reason"])

    def test_one_bad_price_does_not_block_other_positions(self):
        trader = self.make_trader(risk_manager=RiskManager(max_open_positions=2))
        self.open_standard(trader)
        self.assertTrue(trader.buy("QQQ", 0.01, 400.00, 396.00).success)
        results = trader.check_exits({"SPY": NAN, "QQQ": 395.00})
        self.assertEqual(sorted(r.action for r in results),
                         ["EXIT CHECK SKIPPED", "STOP-LOSS SELL"])
        self.assertEqual(list(trader.positions), ["SPY"])

    def test_nothing_happens_if_paper_trading_switch_is_off(self):
        self.open_standard()
        before = self.books()
        with mock.patch.object(settings, "PAPER_TRADING", False):
            results = self.trader.check_exits({"SPY": 494.00})
        self.assertEqual(results[0].action, "EXIT CHECK SKIPPED")
        self.assertEqual(self.books(), before)


if __name__ == "__main__":
    unittest.main()
