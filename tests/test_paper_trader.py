"""
Tests for src/paper_trader.py.

Run from the project root with:
    python -m unittest discover tests -v

Each test gets its own temporary journal file, so the real
logs/trade_journal.csv is never touched.

Handy numbers (default settings, $25 account):
    per-trade risk limit = 2% of $25 = $0.50
    daily loss limit     = 5% of account equity
    "standard trade" below: 0.04 SPY at $500, stop $495
        cost = $20.00, risk = $0.20
"""

import math
import os
import tempfile
import unittest
from datetime import date, timedelta
from unittest import mock

from config import settings
from src import journal
from src.paper_trader import PaperTrader
from src.risk_manager import RiskDecision, RiskManager

NAN = math.nan
INF = math.inf


class FakeClock:
    """Pretend calendar so tests can jump to 'tomorrow'."""
    def __init__(self):
        self.day = date(2026, 1, 5)

    def __call__(self):
        return self.day

    def next_day(self):
        self.day += timedelta(days=1)


class PaperTraderTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.journal_file = os.path.join(self.tmp.name, "journal.csv")
        self.clock = FakeClock()
        self.trader = self.make_trader()

    def tearDown(self):
        self.tmp.cleanup()

    def make_trader(self, **kwargs):
        kwargs.setdefault("journal_file", self.journal_file)
        kwargs.setdefault("today", self.clock)
        return PaperTrader(**kwargs)

    def journal_rows(self):
        return journal.read_journal(self.journal_file)

    def books(self, trader=None):
        """Snapshot of everything that must NOT change on a rejected order."""
        t = trader or self.trader
        return (t.cash, {s: (p.shares, p.entry_price) for s, p in t.positions.items()},
                t.realized_pnl, t.realized_loss_today)

    def buy_standard(self):
        return self.trader.buy("SPY", 0.04, 500.00, 495.00)


# --- Starting state ----------------------------------------------------------

class StartingAccountTests(PaperTraderTestCase):
    def test_starts_with_25_dollars_and_nothing_else(self):
        self.assertEqual(self.trader.cash, 25.00)
        self.assertEqual(self.trader.positions, {})
        self.assertEqual(self.trader.realized_pnl, 0.0)
        self.assertEqual(self.trader.realized_loss_today, 0.0)

    def test_rejects_bad_starting_cash(self):
        for bad in [-1, NAN, INF, "25", None]:
            with self.subTest(starting_cash=bad):
                with self.assertRaises(ValueError):
                    self.make_trader(starting_cash=bad)


# --- Successful purchases ----------------------------------------------------

class SuccessfulPurchaseTests(PaperTraderTestCase):
    def test_buy_updates_cash_and_position(self):
        result = self.buy_standard()
        self.assertTrue(result.success, result.reason)
        self.assertAlmostEqual(self.trader.cash, 5.00)
        position = self.trader.positions["SPY"]
        self.assertEqual(position.shares, 0.04)
        self.assertEqual(position.entry_price, 500.00)
        self.assertEqual(position.stop_loss_price, 495.00)

    def test_symbol_is_cleaned_up(self):
        self.assertTrue(self.trader.buy("  spy ", 0.04, 500.00, 495.00).success)
        self.assertIn("SPY", self.trader.positions)

    def test_buy_is_journaled(self):
        self.buy_standard()
        rows = self.journal_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["action"], "BUY")
        self.assertEqual(rows[0]["symbol"], "SPY")
        self.assertAlmostEqual(float(rows[0]["cash_after"]), 5.00)

    def test_spending_exactly_all_cash(self):
        # 0.05 x $500 = $25.00, risk 0.05 x $10 = $0.50 (exactly on both limits)
        self.assertTrue(self.trader.buy("SPY", 0.05, 500.00, 490.00).success)
        self.assertEqual(self.trader.cash, 0.0)

    def test_buying_more_averages_the_entry_price(self):
        trader = self.make_trader(risk_manager=RiskManager(max_open_positions=2))
        self.assertTrue(trader.buy("SPY", 0.02, 500.00, 495.00).success)
        self.assertTrue(trader.buy("SPY", 0.02, 510.00, 505.00).success)
        position = trader.positions["SPY"]
        self.assertAlmostEqual(position.shares, 0.04)
        self.assertAlmostEqual(position.entry_price, 505.00)
        self.assertEqual(position.stop_loss_price, 505.00)  # latest stop


# --- Risk Manager approval is required ---------------------------------------

class RiskApprovalTests(PaperTraderTestCase):
    def test_too_risky_buy_is_rejected_and_books_unchanged(self):
        before = self.books()
        result = self.trader.buy("SPY", 0.04, 500.00, 450.00)  # risk $2.00
        self.assertFalse(result.success)
        self.assertIn("per-trade limit", result.reason)
        self.assertEqual(self.books(), before)

    def test_rejection_is_journaled_with_reason(self):
        self.trader.buy("SPY", 0.04, 500.00, 450.00)
        rows = self.journal_rows()
        self.assertEqual(rows[-1]["action"], "BUY REJECTED")
        self.assertIn("per-trade limit", rows[-1]["reason"])

    def test_second_position_is_rejected(self):
        self.buy_standard()
        result = self.trader.buy("QQQ", 0.01, 400.00, 396.00)
        self.assertFalse(result.success)
        self.assertIn("the limit is 1", result.reason)

    def test_risk_manager_is_actually_consulted(self):
        """A Risk Manager that says no to everything must block every buy."""
        class AlwaysNo(RiskManager):
            def evaluate(self, trade, account):
                return RiskDecision(False, ["Test says no."])
        trader = self.make_trader(risk_manager=AlwaysNo())
        result = trader.buy("SPY", 0.04, 500.00, 495.00)
        self.assertFalse(result.success)
        self.assertIn("Test says no.", result.reason)
        self.assertEqual(trader.positions, {})

    def test_risk_manager_sees_real_account_numbers(self):
        seen = []
        class Spy(RiskManager):
            def evaluate(self, trade, account):
                seen.append(account)
                return super().evaluate(trade, account)
        trader = self.make_trader(risk_manager=Spy())
        trader.buy("SPY", 0.04, 500.00, 495.00)
        trader.sell("SPY", 0.04, 490.00)          # lose $0.40
        trader.buy("SPY", 0.01, 500.00, 495.00)
        latest = seen[-1]
        self.assertAlmostEqual(latest.cash, 24.60)
        self.assertEqual(latest.open_positions, 0)
        self.assertAlmostEqual(latest.realized_loss_today, 0.40)

    def test_buy_rejected_when_paper_trading_switch_is_off(self):
        with mock.patch.object(settings, "PAPER_TRADING", False):
            result = self.buy_standard()
        self.assertFalse(result.success)
        self.assertIn("Only paper trading is allowed", result.reason)


# --- Insufficient cash -------------------------------------------------------

class InsufficientCashTests(PaperTraderTestCase):
    def test_buy_costing_more_than_cash_is_rejected(self):
        before = self.books()
        result = self.trader.buy("SPY", 0.06, 500.00, 495.00)  # $30 > $25
        self.assertFalse(result.success)
        self.assertIn("cash is available", result.reason)
        self.assertEqual(self.books(), before)

    def test_cash_check_works_even_if_risk_manager_approves_everything(self):
        """Second line of defense: the trader checks cash on its own too."""
        class AlwaysYes(RiskManager):
            def evaluate(self, trade, account):
                return RiskDecision(True, [])
        trader = self.make_trader(risk_manager=AlwaysYes())
        result = trader.buy("SPY", 1, 500.00, 495.00)  # $500 > $25
        self.assertFalse(result.success)
        self.assertIn("cash is available", result.reason)
        self.assertEqual(trader.cash, 25.00)

    def test_deposit_adds_cash(self):
        self.assertTrue(self.trader.deposit(25.00).success)
        self.assertEqual(self.trader.cash, 50.00)
        self.assertEqual(self.journal_rows()[-1]["action"], "DEPOSIT")

    def test_bad_deposits_are_rejected(self):
        for bad in [0, -25, NAN, INF, "25", None]:
            with self.subTest(amount=bad):
                self.assertFalse(self.trader.deposit(bad).success)
        self.assertEqual(self.trader.cash, 25.00)


# --- Successful sales --------------------------------------------------------

class SuccessfulSaleTests(PaperTraderTestCase):
    def test_selling_everything_at_a_profit(self):
        self.buy_standard()
        result = self.trader.sell("SPY", 0.04, 510.00)
        self.assertTrue(result.success, result.reason)
        self.assertAlmostEqual(result.realized_pnl, 0.40)    # 0.04 x $10
        self.assertAlmostEqual(self.trader.cash, 25.40)
        self.assertEqual(self.trader.positions, {})

    def test_partial_sale_keeps_the_rest(self):
        self.buy_standard()
        self.assertTrue(self.trader.sell("SPY", 0.01, 500.00).success)
        position = self.trader.positions["SPY"]
        self.assertAlmostEqual(position.shares, 0.03)
        self.assertEqual(position.entry_price, 500.00)
        self.assertAlmostEqual(self.trader.cash, 10.00)

    def test_close_position_sells_everything(self):
        self.buy_standard()
        result = self.trader.close_position("spy", 505.00)
        self.assertTrue(result.success)
        self.assertEqual(self.trader.positions, {})
        self.assertAlmostEqual(self.trader.cash, 25.20)

    def test_sale_is_journaled_with_pnl(self):
        self.buy_standard()
        self.trader.sell("SPY", 0.04, 510.00)
        row = self.journal_rows()[-1]
        self.assertEqual(row["action"], "SELL")
        self.assertAlmostEqual(float(row["realized_pnl"]), 0.40)
        self.assertAlmostEqual(float(row["cash_after"]), 25.40)


# --- Selling more than held --------------------------------------------------

class OversellTests(PaperTraderTestCase):
    def test_cannot_sell_more_than_held(self):
        self.buy_standard()
        before = self.books()
        result = self.trader.sell("SPY", 0.05, 500.00)
        self.assertFalse(result.success)
        self.assertIn("only 0.04 are held", result.reason)
        self.assertEqual(self.books(), before)

    def test_cannot_sell_something_not_held(self):
        result = self.trader.sell("SPY", 0.01, 500.00)
        self.assertFalse(result.success)
        self.assertIn("don't hold any SPY", result.reason)

    def test_close_position_when_nothing_held(self):
        result = self.trader.close_position("QQQ", 400.00)
        self.assertFalse(result.success)
        self.assertIn("don't hold any QQQ", result.reason)

    def test_cannot_sell_twice(self):
        self.buy_standard()
        self.assertTrue(self.trader.sell("SPY", 0.04, 500.00).success)
        self.assertFalse(self.trader.sell("SPY", 0.04, 500.00).success)


# --- Invalid orders ----------------------------------------------------------

class InvalidOrderTests(PaperTraderTestCase):
    BAD_NUMBERS = [0, -1, NAN, INF, -INF, "10", None, True]
    BAD_SYMBOLS = ["", "   ", None, 123, ["SPY"]]

    def test_invalid_buys_are_rejected_without_crashing(self):
        before = self.books()
        cases = ([dict(shares=v) for v in self.BAD_NUMBERS]
                 + [dict(price=v) for v in self.BAD_NUMBERS]
                 + [dict(stop_loss_price=v) for v in self.BAD_NUMBERS]
                 + [dict(symbol=v) for v in self.BAD_SYMBOLS])
        for change in cases:
            order = dict(symbol="SPY", shares=0.04, price=500.00,
                         stop_loss_price=495.00)
            order.update(change)
            with self.subTest(**{k: repr(v) for k, v in change.items()}):
                result = self.trader.buy(**order)
                self.assertFalse(result.success)
                self.assertTrue(result.reason.startswith("Rejected:"))
        self.assertEqual(self.books(), before)

    def test_invalid_sells_are_rejected_without_crashing(self):
        self.buy_standard()
        before = self.books()
        cases = ([dict(shares=v) for v in self.BAD_NUMBERS]
                 + [dict(price=v) for v in self.BAD_NUMBERS]
                 + [dict(symbol=v) for v in self.BAD_SYMBOLS])
        for change in cases:
            order = dict(symbol="SPY", shares=0.01, price=500.00)
            order.update(change)
            with self.subTest(**{k: repr(v) for k, v in change.items()}):
                result = self.trader.sell(**order)
                self.assertFalse(result.success)
                self.assertTrue(result.reason.startswith("Rejected:"))
        self.assertEqual(self.books(), before)

    def test_every_rejection_is_journaled(self):
        self.trader.buy("SPY", NAN, 500.00, 495.00)
        self.trader.sell("SPY", 1, 500.00)
        self.trader.deposit(-5)
        actions = [row["action"] for row in self.journal_rows()]
        self.assertEqual(actions, ["BUY REJECTED", "SELL REJECTED",
                                   "DEPOSIT REJECTED"])


# --- P&L calculations --------------------------------------------------------

class ProfitAndLossTests(PaperTraderTestCase):
    def test_no_positions_means_zero_unrealized(self):
        self.assertEqual(self.trader.unrealized_pnl({}), 0.0)
        self.assertEqual(self.trader.total_equity({}), 25.00)

    def test_unrealized_gain_and_loss(self):
        self.buy_standard()
        self.assertAlmostEqual(self.trader.unrealized_pnl({"SPY": 502.00}), 0.08)
        self.assertAlmostEqual(self.trader.unrealized_pnl({"SPY": 495.00}), -0.20)

    def test_total_equity_uses_market_prices(self):
        self.buy_standard()  # $5 cash + 0.04 shares
        self.assertAlmostEqual(self.trader.total_equity({"SPY": 510.00}), 25.40)

    def test_unrealized_does_not_change_the_books(self):
        self.buy_standard()
        before = self.books()
        self.trader.unrealized_pnl({"SPY": 1000.00})
        self.assertEqual(self.books(), before)

    def test_missing_or_bad_market_price_raises_clear_error(self):
        self.buy_standard()
        for prices in [{}, {"SPY": NAN}, {"SPY": -1}, {"SPY": "500"}, None]:
            with self.subTest(prices=prices):
                with self.assertRaises(ValueError):
                    self.trader.unrealized_pnl(prices)

    def test_realized_pnl_adds_up_across_trades(self):
        self.buy_standard()
        self.trader.sell("SPY", 0.04, 510.00)  # +0.40
        self.buy_standard()
        self.trader.sell("SPY", 0.04, 497.50)  # -0.10
        self.assertAlmostEqual(self.trader.realized_pnl, 0.30)
        self.assertAlmostEqual(self.trader.cash, 25.30)

    def test_partial_sale_splits_realized_and_unrealized(self):
        self.buy_standard()
        self.trader.sell("SPY", 0.02, 510.00)   # realize +0.20
        self.assertAlmostEqual(self.trader.realized_pnl, 0.20)
        self.assertAlmostEqual(self.trader.unrealized_pnl({"SPY": 510.00}), 0.20)


# --- Daily loss tracking -----------------------------------------------------

class DailyLossTrackingTests(PaperTraderTestCase):
    def test_losing_sale_is_tracked(self):
        self.buy_standard()
        self.trader.sell("SPY", 0.04, 495.00)   # -0.20
        self.assertAlmostEqual(self.trader.realized_loss_today, 0.20)
        self.assertAlmostEqual(self.trader.account_state().realized_loss_today, 0.20)

    def test_winning_sale_adds_no_loss(self):
        self.buy_standard()
        self.trader.sell("SPY", 0.04, 510.00)
        self.assertEqual(self.trader.realized_loss_today, 0.0)

    def test_wins_do_not_erase_losses(self):
        self.buy_standard()
        self.trader.sell("SPY", 0.04, 495.00)   # -0.20
        self.buy_standard()
        self.trader.sell("SPY", 0.04, 510.00)   # +0.40
        self.assertAlmostEqual(self.trader.realized_loss_today, 0.20)
        self.assertAlmostEqual(self.trader.realized_pnl_today, 0.20)

    def test_buying_is_blocked_after_daily_limit(self):
        # Price gaps far below the stop: buy 0.1 @ $100, sell @ $87.50
        self.assertTrue(self.trader.buy("SPY", 0.1, 100.00, 95.00).success)
        self.trader.sell("SPY", 0.1, 87.50)      # -$1.25 (5% of $25)
        result = self.trader.buy("SPY", 0.01, 100.00, 99.00)
        self.assertFalse(result.success)
        self.assertIn("Daily loss limit reached", result.reason)

    def test_new_day_resets_daily_loss(self):
        self.trader.buy("SPY", 0.1, 100.00, 95.00)
        self.trader.sell("SPY", 0.1, 87.50)
        self.clock.next_day()
        self.assertEqual(self.trader.account_state().realized_loss_today, 0.0)
        self.assertTrue(self.trader.buy("SPY", 0.01, 100.00, 99.00).success)
        # All-time P&L is NOT reset
        self.assertAlmostEqual(self.trader.realized_pnl, -1.25)

    def test_selling_is_still_allowed_after_daily_limit(self):
        """Hitting the limit must never trap us in a position."""
        trader = self.make_trader(risk_manager=RiskManager(max_open_positions=2))
        trader.buy("SPY", 0.04, 100.00, 95.00)
        trader.buy("QQQ", 0.1, 100.00, 95.00)
        trader.sell("QQQ", 0.1, 87.50)          # -$1.25
        self.assertTrue(trader.sell("SPY", 0.04, 100.00).success)


# --- Every transaction is journaled ------------------------------------------

class JournalTests(PaperTraderTestCase):
    def test_full_sequence_is_recorded_in_order(self):
        self.buy_standard()
        self.trader.sell("SPY", 1, 500.00)       # rejected: oversell
        self.trader.sell("SPY", 0.04, 505.00)
        self.trader.deposit()
        actions = [row["action"] for row in self.journal_rows()]
        self.assertEqual(actions, ["BUY", "SELL REJECTED", "SELL", "DEPOSIT"])

    def test_real_journal_file_is_not_touched_by_tests(self):
        self.buy_standard()
        self.assertTrue(self.trader.journal_file.startswith(self.tmp.name))


if __name__ == "__main__":
    unittest.main()
