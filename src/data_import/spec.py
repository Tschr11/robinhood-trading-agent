"""
spec.py - The import specification: everything the importer must be TOLD
about a raw file, because guessing would risk silently misreading it.

Example (JSON):
{
  "dataset_id": "spy_5m_split_adjusted",
  "symbol": "SPY",
  "source": "Example vendor, manual download of 2025-01-02 to 2025-12-31",
  "interval_minutes": 5,
  "columns": {"timestamp": "Date", "open": "Open", "high": "High",
              "low": "Low", "close": "Close", "volume": "Volume"},
  "timestamp_format": "iso8601",
  "source_timezone": "America/New_York",
  "timestamp_convention": "bar_start",
  "adjustment": "split_adjusted",
  "volume_coverage": "consolidated",
  "empty_bar_policy": "omitted",
  "outside_regular_hours": "exclude",
  "start_date": "2025-01-02",
  "end_date": "2025-12-31"
}

Every field except start_date, end_date and notes is required; there are no
defaults. Unknown fields are refused.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import date

from src.market_data.candles import MarketDataError
from src.market_data.providers import clean_symbol
from src.market_data.quality import EMPTY_BAR_POLICIES
from src.market_data.sessions import SUPPORTED_INTERVALS

TIMESTAMP_FORMATS = ("iso8601", "unix_seconds", "unix_milliseconds", "unix_nanoseconds")
SOURCE_TIMEZONES = ("UTC", "America/New_York", "explicit_offset")
TIMESTAMP_CONVENTIONS = ("bar_start", "bar_end")
ADJUSTMENTS = ("unadjusted", "split_adjusted", "split_and_dividend_adjusted")
VOLUME_COVERAGE = ("consolidated", "single_venue", "partial_venues")
OUTSIDE_REGULAR_HOURS = ("reject", "exclude")
CANONICAL_FIELDS = ("timestamp", "open", "high", "low", "close", "volume")

REQUIRED_KEYS = {"dataset_id", "symbol", "source", "interval_minutes", "columns",
                 "timestamp_format", "source_timezone", "timestamp_convention",
                 "adjustment", "volume_coverage", "empty_bar_policy",
                 "outside_regular_hours"}
OPTIONAL_KEYS = {"start_date", "end_date", "notes"}


class ImportSpecError(MarketDataError):
    """The import specification is incomplete, ambiguous or invalid."""


@dataclass(frozen=True)
class ImportSpec:
    dataset_id: str
    symbol: str
    source: str
    interval_minutes: int
    columns: tuple                 # ((canonical field, header in the raw file), ...)
    timestamp_format: str
    source_timezone: str
    timestamp_convention: str
    adjustment: str
    volume_coverage: str
    empty_bar_policy: str
    outside_regular_hours: str
    start_date: date | None = None
    end_date: date | None = None
    notes: str = ""

    def header_for(self, field: str) -> str:
        return dict(self.columns)[field]

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id, "symbol": self.symbol, "source": self.source,
            "interval_minutes": self.interval_minutes,
            "columns": {field: header for field, header in self.columns},
            "timestamp_format": self.timestamp_format,
            "source_timezone": self.source_timezone,
            "timestamp_convention": self.timestamp_convention,
            "adjustment": self.adjustment, "volume_coverage": self.volume_coverage,
            "empty_bar_policy": self.empty_bar_policy,
            "outside_regular_hours": self.outside_regular_hours,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "notes": self.notes,
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True)
                              .encode("utf-8")).hexdigest()


def spec_from_dict(raw) -> ImportSpec:
    if not isinstance(raw, dict):
        raise ImportSpecError("An import spec must be a JSON object.")
    problems = []
    missing = REQUIRED_KEYS - set(raw)
    unknown = set(raw) - REQUIRED_KEYS - OPTIONAL_KEYS
    if missing:
        problems.append(f"missing required field(s): {', '.join(sorted(missing))} "
                        "(nothing is defaulted or guessed)")
    if unknown:
        problems.append(f"unknown field(s): {', '.join(sorted(unknown))}")
    if problems:
        raise ImportSpecError("Invalid import spec: " + "; ".join(problems) + ".")

    def choice(key, allowed):
        if raw[key] not in allowed:
            problems.append(f"{key} must be one of {allowed} (got {raw[key]!r})")

    dataset_id = raw["dataset_id"]
    if (not isinstance(dataset_id, str) or not 1 <= len(dataset_id) <= 64
            or not all(ch.islower() or ch.isdigit() or ch in "_-" for ch in dataset_id)
            or not dataset_id[0].isalnum()):
        problems.append("dataset_id must be 1-64 lowercase letters, digits, '_' or '-', "
                        f"starting with a letter or digit (got {dataset_id!r})")
    symbol = raw["symbol"]
    try:
        symbol = clean_symbol(symbol)
    except MarketDataError as error:
        problems.append(str(error))
    if not isinstance(raw["source"], str) or not raw["source"].strip():
        problems.append("source must describe where the file came from")
    interval = raw["interval_minutes"]
    if isinstance(interval, bool) or interval not in SUPPORTED_INTERVALS:
        problems.append(f"interval_minutes must be one of {SUPPORTED_INTERVALS} "
                        f"(got {interval!r})")
    choice("timestamp_format", TIMESTAMP_FORMATS)
    choice("source_timezone", SOURCE_TIMEZONES)
    choice("timestamp_convention", TIMESTAMP_CONVENTIONS)
    choice("adjustment", ADJUSTMENTS)
    choice("volume_coverage", VOLUME_COVERAGE)
    choice("empty_bar_policy", EMPTY_BAR_POLICIES)
    choice("outside_regular_hours", OUTSIDE_REGULAR_HOURS)

    # Combinations whose meaning would be ambiguous.
    if (str(raw["timestamp_format"]).startswith("unix_")
            and raw["source_timezone"] != "UTC"):
        problems.append("unix timestamps count seconds since 1970-01-01 UTC, so "
                        "source_timezone must be 'UTC'")
    if raw["source_timezone"] == "explicit_offset" and raw["timestamp_format"] != "iso8601":
        problems.append("source_timezone 'explicit_offset' needs timestamp_format 'iso8601'")

    columns = raw["columns"]
    if (not isinstance(columns, dict) or set(columns) != set(CANONICAL_FIELDS)
            or not all(isinstance(h, str) and h.strip() for h in columns.values())):
        problems.append(f"columns must map exactly {CANONICAL_FIELDS} to header names")
        columns = {}
    elif len(set(columns.values())) != len(columns):
        problems.append("columns must map each field to a different header")

    start = _date(raw.get("start_date"), "start_date", problems)
    end = _date(raw.get("end_date"), "end_date", problems)
    if start and end and start > end:
        problems.append(f"start_date {start} is after end_date {end}")
    notes = raw.get("notes", "")
    if not isinstance(notes, str):
        problems.append("notes must be text")
    if problems:
        raise ImportSpecError("Invalid import spec: " + "; ".join(problems) + ".")

    return ImportSpec(
        dataset_id=dataset_id, symbol=symbol, source=raw["source"].strip(),
        interval_minutes=interval,
        columns=tuple((field, columns[field]) for field in CANONICAL_FIELDS),
        timestamp_format=raw["timestamp_format"], source_timezone=raw["source_timezone"],
        timestamp_convention=raw["timestamp_convention"], adjustment=raw["adjustment"],
        volume_coverage=raw["volume_coverage"], empty_bar_policy=raw["empty_bar_policy"],
        outside_regular_hours=raw["outside_regular_hours"],
        start_date=start, end_date=end, notes=notes)


def load_spec(path: str) -> ImportSpec:
    try:
        with open(path, "rb") as f:
            raw = json.loads(f.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ImportSpecError(f"Could not read import spec {path}: {error}") from None
    return spec_from_dict(raw)


def _date(value, label, problems):
    if value is None:
        return None
    if not isinstance(value, str):
        problems.append(f"{label} must be YYYY-MM-DD")
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        problems.append(f"{label} must be YYYY-MM-DD (got {value!r})")
        return None
