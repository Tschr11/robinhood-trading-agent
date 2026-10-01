"""
Tests for saving the paper account in SQLite (src/storage.py + PaperTrader).

Run from the project root with:
    python -m unittest discover tests -v

A "restart" in these tests means: close the trader, then create a new
PaperTrader pointing at the same database file - exactly what happens when
the program is stopped and started again.

Every test uses its own temporary folder, so data/paper_account.db and
logs/trade_journal.csv are never touched.
"""

import math
import os
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
from unittest import mock

from config import settings
from src import journal
from src.paper_trader import PaperTrader
from src.risk_manager import RiskManager


class FakeClock:
    def __init__(self):
        self.day = date(2026, 1, 5)

    def __call__(self):
        return self.day


class PersistenceTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = os.path.join(self.tmp.name, "account.db")
        self.journal_file = os.path.join(self.tmp.name, "journal.csv")
        self.clock = FakeClock()
        self.trader = self.open()

    def open(self, **kwargs):
        """Start (or restart) a trader on this test's database file."""
        kwargs.setdefault("db_path", self.db_path)
        kwargs.setdefault("journal_file", self.journal_file)
        kwargs.setdefault("today", self.clock)
        trader = PaperTrader(**kwargs)
        self.addCleanup(trader.close)
        return trader

    def restart(self, **kwargs):
        self.trader.close()
        self.trader = self.open(**kwargs)
        return self.trader

    def books(self, trader=None):
        t = trader or self.trader
        return (t.cash, {s: (p.shares, p.entry_price, p.stop_loss_price,
                             p.take_profit_price) for s, p in t.positions.items()},
                t.realized_pnl, t.realized_pnl_today, t.realized_loss_today)

    def saved_books(self):
        """Books as a brand-new trader reads them straight from disk."""
        other = self.open()
        return self.books(other)

    def journal_actions(self):
        return [row["action"] for row in journal.read_journal(self.journal_file)]

    def ledger_actions(self):
        return [row["action"] for row in self.trader.store.transactions()]

    def buy_standard(self, trader=None, **kwargs):
        result = (trader or self.trader).buy("SPY", 0.04, 500.00, 495.00, **kwargs)
        self.assertTrue(result.success, result.reason)
        return result


# --- New accounts ------------------------------------------------------------

class NewAccountTests(PersistenceTestCase):
    def test_new_account_uses_configured_starting_capital(self):
        self.assertFalse(self.trader.restored)
        self.assertEqual(self.trader.cash, settings.STARTING_CAPITAL)
        self.assertEqual(self.trader.starting_capital, 25.00)
        self.assertEqual(self.trader.positions, {})

    def test_database_file_is_created(self):
        self.assertTrue(os.path.exists(self.db_path))

    def test_custom_starting_cash_for_a_new_account(self):
        trader = self.open(db_path=os.path.join(self.tmp.name, "other.db"),
                           starting_cash=100.00)
        self.assertEqual(trader.cash, 100.00)


# --- Restarting with cash ----------------------------------------------------

class RestartWithCashTests(PersistenceTestCase):
    def test_restart_restores_instead_of_resetting(self):
        self.trader.deposit(25.00)
        trader = self.restart()
        self.assertTrue(trader.restored)
        self.assertEqual(trader.cash, 50.00)

    def test_starting_cash_is_ignored_for_an_existing_account(self):
        self.trader.deposit(10.00)
        trader = self.restart(starting_cash=999.00)
        self.assertEqual(trader.cash, 35.00)
        self.assertEqual(trader.starting_capital, 25.00)

    def test_several_restarts_keep_adding_up(self):
        for _ in range(3):
            self.trader.deposit(25.00)
            self.restart()
        self.assertEqual(self.trader.cash, 100.00)


# --- Restarting with open positions ------------------------------------------

class RestartWithPositionsTests(PersistenceTestCase):
    def test_every_position_field_survives_a_restart(self):
        self.buy_standard(take_profit_price=512.34)
        before = self.books()
        trader = self.restart()
        position = trader.positions["SPY"]
        self.assertEqual(position.symbol, "SPY")
        self.assertEqual(position.shares, 0.04)
        self.assertEqual(position.entry_price, 500.00)
        self.assertEqual(position.stop_loss_price, 495.00)
        self.assertEqual(position.take_profit_price, 512.34)
        self.assertEqual(self.books(), before)

    def test_averaged_entry_price_survives_a_restart(self):
        trader = self.restart(risk_manager=RiskManager(max_open_positions=2))
        trader.buy("SPY", 0.02, 500.00, 495.00)
        trader.buy("SPY", 0.02, 510.00, 505.00)
        trader = self.restart()
        self.assertAlmostEqual(trader.positions["SPY"].entry_price, 505.00)
        self.assertAlmostEqual(trader.positions["SPY"].shares, 0.04)

    def test_position_limit_still_applies_after_restart(self):
        self.buy_standard()
        trader = self.restart()
        result = trader.buy("QQQ", 0.01, 400.00, 396.00)
        self.assertFalse(result.success)
        self.assertIn("the limit is 1", result.reason)

    def test_restored_position_can_be_sold(self):
        self.buy_standard()
        trader = self.restart()
        self.assertTrue(trader.sell("SPY", 0.04, 510.00).success)
        self.assertAlmostEqual(trader.cash, 25.40)

    def test_restored_position_still_has_stop_loss_protection(self):
        self.buy_standard()
        trader = self.restart()
        results = trader.check_exits({"SPY": 494.00})
        self.assertEqual(results[0].action, "STOP-LOSS SELL")


# --- Persistence after buys, sells, exits and deposits ----------------------

class PersistAfterTradesTests(PersistenceTestCase):
    def test_buy_is_saved_immediately(self):
        self.buy_standard()
        self.assertEqual(self.saved_books(), self.books())
        self.assertAlmostEqual(self.saved_books()[0], 5.00)

    def test_full_sale_is_saved(self):
        self.buy_standard()
        self.trader.sell("SPY", 0.04, 510.00)
        cash, positions, realized, realized_today, loss_today = self.saved_books()
        self.assertAlmostEqual(cash, 25.40)
        self.assertEqual(positions, {})
        self.assertAlmostEqual(realized, 0.40)
        self.assertAlmostEqual(realized_today, 0.40)
        self.assertEqual(loss_today, 0.0)

    def test_partial_sale_is_saved(self):
        self.buy_standard()
        self.trader.sell("SPY", 0.01, 490.00)
        cash, positions, realized, _, loss_today = self.saved_books()
        self.assertAlmostEqual(positions["SPY"][0], 0.03)
        self.assertAlmostEqual(cash, 9.90)
        self.assertAlmostEqual(realized, -0.10)
        self.assertAlmostEqual(loss_today, 0.10)

    def test_close_position_is_saved(self):
        self.buy_standard()
        self.trader.close_position("SPY", 505.00)
        self.assertEqual(self.saved_books()[1], {})

    def test_automatic_exits_are_saved(self):
        self.buy_standard()
        self.trader.check_exits({"SPY": 512.00})          # take-profit
        self.assertEqual(self.saved_books(), self.books())
        self.assertEqual(self.saved_books()[1], {})
        self.buy_standard()
        self.trader.check_exits({"SPY": 494.00})          # stop-loss
        self.assertEqual(self.saved_books(), self.books())

    def test_deposit_is_saved(self):
        self.trader.deposit(25.00)
        self.assertEqual(self.saved_books()[0], 50.00)

    def test_rejected_orders_change_nothing_on_disk(self):
        before = self.saved_books()
        self.trader.buy("SPY", 1, 500.00, 495.00)          # too expensive
        self.trader.sell("SPY", 0.01, 500.00)              # nothing held
        self.trader.deposit(-5)                            # invalid
        self.assertEqual(self.saved_books(), before)

    def test_ledger_records_money_movements_only(self):
        self.buy_standard()
        self.trader.sell("SPY", 1, 500.00)                 # rejected
        self.trader.sell("SPY", 0.04, 505.00)
        self.trader.deposit()
        self.buy_standard()
        self.trader.check_exits({"SPY": 494.00})
        self.assertEqual(self.ledger_actions(),
                         ["BUY", "SELL", "DEPOSIT", "BUY", "STOP-LOSS SELL"])
        last = self.trader.store.transactions()[-1]
        self.assertAlmostEqual(last["realized_pnl"], -0.24)
        self.assertAlmostEqual(last["cash_after"], self.trader.cash)

    def test_csv_journal_still_works(self):
        self.buy_standard()
        self.trader.sell("SPY", 1, 500.00)
        self.trader.sell("SPY", 0.04, 505.00)
        self.trader.deposit()
        self.assertEqual(self.journal_actions(),
                         ["BUY", "SELL REJECTED", "SELL", "DEPOSIT"])


# --- Daily loss tracking across restarts ------------------------------------

class DailyLossPersistenceTests(PersistenceTestCase):
    def take_big_loss(self):
        self.assertTrue(self.trader.buy("SPY", 0.1, 100.00, 95.00).success)
        self.assertTrue(self.trader.sell("SPY", 0.1, 87.50).success)  # -$1.25

    def test_daily_loss_survives_restart(self):
        self.take_big_loss()
        trader = self.restart()
        self.assertAlmostEqual(trader.realized_loss_today, 1.25)
        self.assertEqual(trader.current_day, date(2026, 1, 5))

    def test_restart_cannot_dodge_the_daily_loss_limit(self):
        self.take_big_loss()
        trader = self.restart()
        result = trader.buy("SPY", 0.01, 100.00, 99.00)
        self.assertFalse(result.success)
        self.assertIn("Daily loss limit reached", result.reason)

    def test_new_day_after_restart_resets_daily_loss_only(self):
        self.take_big_loss()
        self.trader.close()
        self.clock.day += timedelta(days=1)
        trader = self.open()
        self.assertEqual(trader.account_state().realized_loss_today, 0.0)
        self.assertTrue(trader.buy("SPY", 0.01, 100.00, 99.00).success)
        self.assertAlmostEqual(trader.realized_pnl, -1.25)       # all-time kept
        self.assertEqual(self.saved_books()[4], 0.0)             # reset saved


# --- Duplicate prevention ----------------------------------------------------

class DuplicateOrderTests(PersistenceTestCase):
    def test_same_order_id_is_applied_only_once(self):
        self.assertTrue(self.trader.deposit(10.00, order_id="dep-1").success)
        repeat = self.trader.deposit(10.00, order_id="dep-1")
        self.assertFalse(repeat.success)
        self.assertIn("Duplicate order", repeat.reason)
        self.assertEqual(self.trader.cash, 35.00)

    def test_duplicates_are_caught_after_a_restart(self):
        self.buy_standard(order_id="buy-1")
        trader = self.restart(risk_manager=RiskManager(max_open_positions=2))
        repeat = trader.buy("SPY", 0.01, 500.00, 495.00, order_id="buy-1")
        self.assertFalse(repeat.success)
        self.assertIn("Duplicate order", repeat.reason)
        self.assertEqual(trader.positions["SPY"].shares, 0.04)

    def test_duplicate_sell_is_blocked(self):
        trader = self.restart(risk_manager=RiskManager(max_open_positions=2))
        trader.buy("SPY", 0.04, 500.00, 495.00)
        self.assertTrue(trader.sell("SPY", 0.01, 505.00, order_id="s-1").success)
        self.assertFalse(trader.sell("SPY", 0.01, 505.00, order_id="s-1").success)
        self.assertAlmostEqual(trader.positions["SPY"].shares, 0.03)

    def test_rejected_order_does_not_use_up_its_id(self):
        self.assertFalse(self.trader.buy("SPY", 1, 500.00, 495.00,
                                         order_id="try-1").success)
        self.assertTrue(self.buy_standard(order_id="try-1").success)

    def test_bad_order_ids_are_rejected(self):
        for bad in ["", "   ", 123, True]:
            with self.subTest(order_id=bad):
                self.assertFalse(self.trader.deposit(5, order_id=bad).success)
        self.assertEqual(self.trader.cash, 25.00)


# --- Failures leave the account consistent ----------------------------------

class FailureSafetyTests(PersistenceTestCase):
    def assertUnchangedEverywhere(self, before, ledger_before, journal_before):
        self.assertEqual(self.books(), before)            # memory
        self.assertEqual(self.saved_books(), before)      # disk
        self.assertEqual(self.ledger_actions(), ledger_before)
        self.assertEqual(self.journal_actions(), journal_before)

    def test_crash_while_saving_a_buy_rolls_everything_back(self):
        before, ledger, csv_rows = self.books(), self.ledger_actions(), self.journal_actions()
        with mock.patch.object(self.trader.store, "save_positions",
                               side_effect=RuntimeError("disk exploded")):
            with self.assertRaises(RuntimeError):
                self.buy_standard()
        self.assertUnchangedEverywhere(before, ledger, csv_rows)

    def test_crash_while_saving_a_sell_rolls_everything_back(self):
        self.buy_standard()
        before, ledger, csv_rows = self.books(), self.ledger_actions(), self.journal_actions()
        # save_account runs first and succeeds; save_positions then fails.
        with mock.patch.object(self.trader.store, "save_positions",
                               side_effect=RuntimeError("disk exploded")):
            with self.assertRaises(RuntimeError):
                self.trader.sell("SPY", 0.04, 510.00)
        self.assertUnchangedEverywhere(before, ledger, csv_rows)
        self.assertIn("SPY", self.trader.positions)

    def test_crash_while_writing_ledger_rolls_back_an_exit(self):
        self.buy_standard()
        before, ledger, csv_rows = self.books(), self.ledger_actions(), self.journal_actions()
        with mock.patch.object(self.trader.store, "add_transaction",
                               side_effect=sqlite3.OperationalError("locked")):
            with self.assertRaises(sqlite3.OperationalError):
                self.trader.check_exits({"SPY": 494.00})
        self.assertUnchangedEverywhere(before, ledger, csv_rows)

    def test_database_refuses_impossible_values(self):
        """CHECK rules are a last line of defense, even for raw SQL."""
        conn = self.trader.store.conn
        for sql in ["UPDATE account SET cash = -1",
                    "UPDATE account SET realized_loss_today = -5",
                    "INSERT INTO positions VALUES ('SPY', 0, 500, 495, 510)",
                    "INSERT INTO positions VALUES ('SPY', 1, -500, 495, 510)",
                    "INSERT INTO account (id, cash, starting_capital, realized_pnl, "
                    "current_day, realized_pnl_today, realized_loss_today, "
                    "created_at, updated_at) VALUES (2, 1, 1, 0, 'x', 0, 0, 'x', 'x')"]:
            with self.subTest(sql=sql):
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(sql)

    def test_nan_can_never_be_saved(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.trader.store.conn.execute("UPDATE account SET cash = ?", (math.nan,))

    def test_journal_write_failure_keeps_the_committed_trade(self):
        with mock.patch.object(journal, "log_decision",
                               side_effect=OSError("disk full")):
            with self.assertWarns(UserWarning):
                result = self.buy_standard()
        self.assertTrue(result.success)
        self.assertIn("SPY", self.saved_books()[1])
        self.assertEqual(self.ledger_actions(), ["BUY"])


# --- Two programs sharing one database --------------------------------------

class SharedDatabaseTests(PersistenceTestCase):
    def test_second_trader_sees_changes_before_acting(self):
        other = self.open()                    # opened BEFORE the buy
        self.buy_standard()
        result = other.buy("QQQ", 0.01, 400.00, 396.00)
        self.assertFalse(result.success)       # sees the open SPY position
        self.assertIn("the limit is 1", result.reason)
        self.assertAlmostEqual(other.cash, 5.00)

    def test_second_trader_cannot_sell_a_position_twice(self):
        other = self.open()
        self.buy_standard()
        self.assertTrue(self.trader.sell("SPY", 0.04, 505.00).success)
        self.assertFalse(other.sell("SPY", 0.04, 505.00).success)
        self.assertAlmostEqual(self.saved_books()[0], 25.20)


# --- Safety -----------------------------------------------------------------

class StorageSafetyTests(PersistenceTestCase):
    def test_no_credential_like_columns(self):
        conn = self.trader.store.conn
        words = ["key", "secret", "password", "token", "credential", "login",
                 "account_number", "broker"]
        for table in ["account", "positions", "transactions"]:
            columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
            for column in columns:
                for word in words:
                    with self.subTest(table=table, column=column):
                        self.assertNotIn(word, column.lower())

    def test_real_data_folder_is_not_used_by_tests(self):
        self.assertTrue(self.trader.store.path.startswith(self.tmp.name))
        self.assertNotEqual(os.path.abspath(self.db_path),
                            os.path.abspath(settings.DATABASE_FILE))


if __name__ == "__main__":
    unittest.main()
