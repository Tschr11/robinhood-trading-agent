"""
storage.py - Saves the paper trading account to a local SQLite file.

SQLite is a small database that ships with Python. It keeps everything in
ONE file on your computer (data/paper_account.db by default). It has no
network ability, and this file holds no credentials - only pretend money.

Tables:
    account       exactly one row: cash, realized P&L, and today's numbers
    positions     one row per symbol we hold
    transactions  a permanent ledger of every buy, sell, exit and deposit

Safety idea - transactions:
    Every change to the account happens inside `with store.transaction():`.
    Either EVERYTHING inside that block is saved, or (if anything goes wrong)
    NOTHING is. The account can never be left half-updated, for example with
    cash taken but no position recorded.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS account (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),  -- only one account
    cash                REAL NOT NULL CHECK (cash >= 0),
    starting_capital    REAL NOT NULL CHECK (starting_capital >= 0),
    realized_pnl        REAL NOT NULL,
    current_day         TEXT NOT NULL,                       -- e.g. 2026-01-05
    realized_pnl_today  REAL NOT NULL,
    realized_loss_today REAL NOT NULL CHECK (realized_loss_today >= 0),
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    symbol            TEXT PRIMARY KEY CHECK (length(symbol) > 0),
    shares            REAL NOT NULL CHECK (shares > 0),
    entry_price       REAL NOT NULL CHECK (entry_price > 0),
    stop_loss_price   REAL NOT NULL CHECK (stop_loss_price > 0),
    take_profit_price REAL NOT NULL CHECK (take_profit_price > 0)
);

CREATE TABLE IF NOT EXISTS transactions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL,
    action       TEXT NOT NULL,
    symbol       TEXT,
    shares       REAL,
    price        REAL,
    realized_pnl REAL,
    cash_after   REAL NOT NULL,
    reason       TEXT,
    order_id     TEXT UNIQUE      -- optional label that blocks duplicates
);
"""


class AccountStore:
    """Reads and writes the paper account in a SQLite file."""

    def __init__(self, path: str):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        # isolation_level=None lets us control transactions ourselves with
        # BEGIN / COMMIT / ROLLBACK. timeout waits if another program is
        # writing at the same moment instead of failing straight away.
        self.path = path
        self.conn = sqlite3.connect(path, isolation_level=None, timeout=5)
        self.conn.row_factory = sqlite3.Row
        self._in_transaction = False
        self.conn.executescript(SCHEMA)
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self.conn.close()

    # -- All-or-nothing changes ---------------------------------------------------

    @contextmanager
    def transaction(self):
        """
        BEGIN IMMEDIATE takes the write lock right away, so no other program
        can change the account between our reading it and saving it.
        """
        if self._in_transaction:
            raise RuntimeError("A transaction is already open.")
        self.conn.execute("BEGIN IMMEDIATE")
        self._in_transaction = True
        try:
            yield
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        finally:
            self._in_transaction = False

    # -- Account row --------------------------------------------------------------

    def load_account(self) -> dict | None:
        """The saved account, or None if this is a brand-new database."""
        row = self.conn.execute("SELECT * FROM account WHERE id = 1").fetchone()
        return dict(row) if row else None

    def create_account(self, cash: float, current_day: str) -> None:
        now = _now()
        self.conn.execute(
            "INSERT INTO account (id, cash, starting_capital, realized_pnl, "
            "current_day, realized_pnl_today, realized_loss_today, "
            "created_at, updated_at) VALUES (1, ?, ?, 0, ?, 0, 0, ?, ?)",
            (cash, cash, current_day, now, now))

    def save_account(self, cash: float, realized_pnl: float, current_day: str,
                     realized_pnl_today: float, realized_loss_today: float) -> None:
        self.conn.execute(
            "UPDATE account SET cash = ?, realized_pnl = ?, current_day = ?, "
            "realized_pnl_today = ?, realized_loss_today = ?, updated_at = ? "
            "WHERE id = 1",
            (cash, realized_pnl, current_day, realized_pnl_today,
             realized_loss_today, _now()))

    # -- Positions ----------------------------------------------------------------

    def load_positions(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM positions ORDER BY symbol")
        return [dict(row) for row in rows]

    def save_positions(self, positions: list[dict]) -> None:
        """Replace the saved positions with exactly this list."""
        self.conn.execute("DELETE FROM positions")
        self.conn.executemany(
            "INSERT INTO positions (symbol, shares, entry_price, "
            "stop_loss_price, take_profit_price) VALUES "
            "(:symbol, :shares, :entry_price, :stop_loss_price, :take_profit_price)",
            positions)

    # -- Ledger -------------------------------------------------------------------

    def add_transaction(self, action: str, symbol, shares, price, realized_pnl,
                        cash_after: float, reason: str, order_id: str | None) -> None:
        self.conn.execute(
            "INSERT INTO transactions (timestamp, action, symbol, shares, price, "
            "realized_pnl, cash_after, reason, order_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(), action, symbol, shares, price, realized_pnl, cash_after,
             reason, order_id))

    def order_exists(self, order_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM transactions WHERE order_id = ?", (order_id,)).fetchone()
        return row is not None

    def transactions(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM transactions ORDER BY id")
        return [dict(row) for row in rows]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
