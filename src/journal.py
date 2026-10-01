"""
journal.py - Writes down everything the agent does and why.

Each decision or simulated transaction becomes one row in a CSV file
(logs/trade_journal.csv by default), which you can open in Excel or
Google Sheets.
"""

import csv
import os
from datetime import datetime

from config import settings

COLUMNS = ["timestamp", "symbol", "signal", "action", "shares", "price",
           "reason", "realized_pnl", "cash_after"]


def log_decision(symbol: str, signal: str, action: str,
                 shares: float, price: float, reason: str,
                 realized_pnl: float | str = "", cash_after: float | str = "",
                 path: str = settings.JOURNAL_FILE) -> None:
    """
    Append one row to the journal, creating the file (and folder) if needed.

    `realized_pnl` and `cash_after` are optional: decisions that did not move
    any money (like HOLD) can leave them blank. `path` lets tests write to a
    temporary file instead of the real journal.
    """
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    is_new_file = not os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(COLUMNS)
        writer.writerow([datetime.now().isoformat(timespec="seconds"),
                         symbol, signal, action, shares, price, reason,
                         realized_pnl, cash_after])


def read_journal(path: str = settings.JOURNAL_FILE) -> list[dict]:
    """Return every journal row as a dictionary (empty list if no file yet)."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
