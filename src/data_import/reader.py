"""
reader.py - Turns the bytes of a raw CSV file into regular-session candles in
New York time, exactly as the import spec describes. Strict: any malformed
row rejects the whole file. Nothing is sorted, deduplicated, filled in or
"fixed".

The only rows that may be left out are ones the spec EXPLICITLY asks to
exclude - bars outside regular trading hours on trading days
("outside_regular_hours": "exclude") and bars outside start_date/end_date.
Those are counted and recorded, never dropped silently.
"""

import csv
import io
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from src.data_import.spec import ImportSpec
from src.market_data.candles import Candle, MarketDataError
from src.market_data.sessions import NEW_YORK, TradingCalendar

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
MAX_LISTED_PROBLEMS = 50


class ImportRejected(MarketDataError):
    """The raw file or its data was rejected. Nothing was written."""

    def __init__(self, problems, report=None):
        super().__init__(problems)
        self.report = report


@dataclass
class ParsedRaw:
    candles: list = field(default_factory=list)
    rows: int = 0
    excluded_outside_regular_hours: int = 0
    excluded_outside_date_range: int = 0
    ignored_columns: tuple = ()


def parse_raw(raw_bytes: bytes, spec: ImportSpec, calendar: TradingCalendar) -> ParsedRaw:
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ImportRejected(f"The file is not valid UTF-8 text ({error}).") from None
    if not text.strip():
        raise ImportRejected("The file is empty.")

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except csv.Error as error:
        raise ImportRejected(f"The header row is malformed ({error}).") from None
    header = [h.strip() for h in header]
    duplicates = sorted({h for h in header if header.count(h) > 1})
    if duplicates:
        raise ImportRejected(f"The header repeats column(s): {', '.join(duplicates)}.")
    positions = {}
    for canonical, name in spec.columns:
        if name not in header:
            raise ImportRejected(f"The header has no {name!r} column (mapped to {canonical}). "
                                 f"Header: {header}.")
        positions[canonical] = header.index(name)
    parsed = ParsedRaw(ignored_columns=tuple(h for h in header
                                             if h not in dict(spec.columns).values()))

    problems = []
    try:
        for row in reader:
            line = reader.line_num
            parsed.rows += 1
            if len(row) != len(header):
                problems.append(f"Line {line}: has {len(row)} field(s), the header has "
                                f"{len(header)}.")
                continue
            ts, problem = parse_timestamp(row[positions["timestamp"]].strip(), spec)
            row_problems = [problem] if problem else []
            values = {}
            for name in ("open", "high", "low", "close", "volume"):
                values[name], problem = parse_number(row[positions[name]], name)
                if problem:
                    row_problems.append(problem)
            if row_problems:
                problems += [f"Line {line}: {p}" for p in row_problems]
                continue
            if spec.timestamp_convention == "bar_end":
                ts = ts - timedelta(minutes=spec.interval_minutes)
            local = ts.astimezone(NEW_YORK)
            if not _in_date_range(local, spec):
                parsed.excluded_outside_date_range += 1
                continue
            if (spec.outside_regular_hours == "exclude"
                    and _outside_session_on_trading_day(local, spec, calendar)):
                parsed.excluded_outside_regular_hours += 1
                continue
            parsed.candles.append(Candle(local, **values))
    except csv.Error as error:
        problems.append(f"Line {reader.line_num}: malformed CSV ({error}).")

    if problems:
        shown = problems[:MAX_LISTED_PROBLEMS]
        if len(problems) > len(shown):
            shown.append(f"... and {len(problems) - len(shown)} more problem row(s).")
        raise ImportRejected(shown)
    return parsed


def parse_number(text: str, name: str):
    """Strict decimal number: no blanks, separators, NaN or infinity."""
    text = text.strip()
    if not text:
        return None, f"{name} is blank."
    if any(ch in text for ch in ",_ ") or text.lower().lstrip("+-") in ("nan", "inf",
                                                                        "infinity"):
        return None, f"{name} {text!r} is not a plain decimal number."
    try:
        value = float(text)
    except ValueError:
        return None, f"{name} {text!r} is not a number."
    if not math.isfinite(value):
        return None, f"{name} {text!r} is not a finite number."
    return value, None


def parse_timestamp(text: str, spec: ImportSpec):
    """An aware datetime for `text`, or (None, reason). Never guesses a zone."""
    if not text:
        return None, "timestamp is blank."
    fmt = spec.timestamp_format
    if fmt.startswith("unix_"):
        if not text.isdigit():
            return None, f"timestamp {text!r} is not a whole number of {fmt[5:]}."
        n = int(text)
        if fmt == "unix_seconds":
            return EPOCH + timedelta(seconds=n), None
        if fmt == "unix_milliseconds":
            return EPOCH + timedelta(milliseconds=n), None
        if n % 1000:
            return None, f"timestamp {text!r} has sub-microsecond precision."
        return EPOCH + timedelta(microseconds=n // 1000), None

    try:
        ts = datetime.fromisoformat(text)
    except ValueError:
        return None, f"timestamp {text!r} is not ISO 8601."
    has_offset = ts.utcoffset() is not None
    if spec.source_timezone == "explicit_offset":
        if not has_offset:
            return None, (f"timestamp {text!r} has no UTC offset, but the spec says "
                          "every timestamp carries one.")
        return ts, None
    if has_offset and spec.source_timezone == "UTC" and ts.utcoffset() == timedelta(0):
        return ts, None                 # e.g. "...Z" or "+00:00" in a UTC file
    if has_offset:
        return None, (f"timestamp {text!r} carries a UTC offset, but the spec says "
                      f"timestamps are naive {spec.source_timezone} times; this is "
                      "ambiguous.")
    if spec.source_timezone == "UTC":
        return ts.replace(tzinfo=timezone.utc), None
    return _localize_new_york(ts, text)


def _localize_new_york(naive: datetime, text: str):
    """Attach New York time, refusing times that don't exist or occur twice."""
    first = naive.replace(tzinfo=NEW_YORK, fold=0)
    round_trip = first.astimezone(timezone.utc).astimezone(NEW_YORK).replace(tzinfo=None)
    if round_trip != naive:
        return None, (f"timestamp {text!r} does not exist in New York (clocks skip "
                      "that hour when daylight saving starts).")
    if first.utcoffset() != naive.replace(tzinfo=NEW_YORK, fold=1).utcoffset():
        return None, (f"timestamp {text!r} happens twice in New York (clocks repeat "
                      "that hour when daylight saving ends); it is ambiguous.")
    return first, None


def _in_date_range(local: datetime, spec: ImportSpec) -> bool:
    day = local.date()
    return ((spec.start_date is None or day >= spec.start_date)
            and (spec.end_date is None or day <= spec.end_date))


def _outside_session_on_trading_day(local, spec, calendar) -> bool:
    """True only for bars on a TRADING day that lie outside its regular session."""
    session = calendar.session(local.date())
    if session is None:
        return False                    # closed day: kept, so it is reported and rejected
    end = local + timedelta(minutes=spec.interval_minutes)
    return local < session.open or end > session.close
