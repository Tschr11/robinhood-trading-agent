"""
journal.py - Writes down everything the agent does and why.

Each decision becomes one row in a CSV file (logs/trade_journal.csv),
which you can open in Excel or Google Sheets.
"""

import csv
import os
from datetime import datetime

from config import settings

COLUMNS = ["timestamp", "symbol", "signal", "action", "shares", "price", "reason"]


def log_decision(symbol: str, signal: str, action: str,
                 shares: float, price: float, reason: str) -> None:
    """Append one decision to the journal, creating the file if needed."""
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    is_new_file = not os.path.exists(settings.JOURNAL_FILE)

    with open(settings.JOURNAL_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(COLUMNS)
        writer.writerow([datetime.now().isoformat(timespec="seconds"),
                         symbol, signal, action, shares, price, reason])
