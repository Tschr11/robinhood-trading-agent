"""
sessions.py - Exchange time zone, trading calendar and the regular-session grid.

Conventions used for all imported historical data:
    - Exchange: NYSE regular trading hours, 09:30-16:00 America/New_York,
      with official early closes (usually 13:00).
    - Timestamps mark the START of each bar and carry New York's own UTC
      offset for that moment: -05:00 in winter (EST), -04:00 in summer (EDT).
      Example: the 09:30-09:35 bar on 2026-01-05 is 2026-01-05T09:30:00-05:00.
    - A bar must fit completely inside the session: with 5-minute bars the
      first is 09:30 and the last is 15:55 (12:55 on a 13:00 early close).

The calendar (config/market_calendar.json) is the only source of trading
days. It is versioned and covers named years only. A year that is missing,
or not yet marked verified against official NYSE documents, is REFUSED -
it is never assumed to be a normal year.

Nothing here creates, fills in or moves a bar. These functions only answer
"when should bars exist?" and "is this timestamp a valid bar start?".
"""

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from config import settings
from src.market_data.candles import MarketDataError

NEW_YORK = ZoneInfo("America/New_York")
EXCHANGE = "XNYS"
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
# Bar lengths (minutes) that divide every regular and early-close session
# into whole bars starting at 09:30.
SUPPORTED_INTERVALS = (1, 5, 15, 30)

_TOP_KEYS = {"calendar", "description", "version", "timezone", "regular_session",
             "verification_instructions", "years"}
_YEAR_KEYS = {"verified", "verified_by", "verified_on", "evidence", "sources", "holidays",
              "special_closures", "early_closes"}
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


class CalendarError(MarketDataError):
    """The calendar is invalid, or does not (yet) cover a requested year."""


# --- Data holders ------------------------------------------------------------------

@dataclass(frozen=True)
class Session:
    """One trading day's regular session, in New York time."""
    day: date
    open: datetime                 # e.g. 2026-01-05 09:30-05:00
    close: datetime                # 16:00, or 13:00 on an early close
    early_close: bool
    note: str = ""                 # e.g. "Day after Thanksgiving"

    @property
    def minutes(self) -> int:
        return int((self.close - self.open).total_seconds() // 60)


@dataclass(frozen=True)
class YearCalendar:
    year: int
    verified: bool
    verified_by: str | None
    verified_on: str | None
    evidence: tuple[str, ...]      # how the year was checked (required once verified)
    sources: tuple[str, ...]
    holidays: dict                 # date -> name
    special_closures: dict         # date -> name (unscheduled full-day closures)
    early_closes: dict             # date -> (close time, name)


# --- The calendar --------------------------------------------------------------------

class TradingCalendar:
    """
    NYSE regular-hours calendar for the years listed in its data file.

    allow_unverified=False (the default) refuses any year whose entry has not
    been checked against the official documents. Pass True only for tests or
    while you are doing that check yourself.
    """

    def __init__(self, data: dict, *, allow_unverified: bool = False, source: str = ""):
        problems = []
        if not isinstance(data, dict):
            raise CalendarError("Calendar data must be a JSON object.")
        unknown = set(data) - _TOP_KEYS
        missing = {"calendar", "version", "timezone", "regular_session", "years"} - set(data)
        if unknown:
            problems.append(f"unknown key(s): {', '.join(sorted(unknown))}")
        if missing:
            problems.append(f"missing key(s): {', '.join(sorted(missing))}")
        if problems:
            raise CalendarError(f"Calendar {source}: " + "; ".join(problems) + ".")

        if data["calendar"] != EXCHANGE:
            problems.append(f"calendar must be {EXCHANGE!r} (got {data['calendar']!r})")
        if data["timezone"] != "America/New_York":
            problems.append(f"timezone must be 'America/New_York' (got {data['timezone']!r})")
        if not isinstance(data["version"], str) or not data["version"].strip():
            problems.append("version must be non-empty text")
        if data["regular_session"] != {"open": "09:30", "close": "16:00"}:
            problems.append("regular_session must be {'open': '09:30', 'close': '16:00'}")
        years = data["years"]
        if not isinstance(years, dict) or not years:
            problems.append("years must be a non-empty object")
            years = {}

        self.version = data["version"]
        self.source = source
        self.allow_unverified = allow_unverified
        self._years = {}
        for label, entry in years.items():
            year_problems, parsed = _parse_year(label, entry)
            problems += year_problems
            if parsed is not None:
                self._years[parsed.year] = parsed
        if problems:
            raise CalendarError(f"Calendar {source or self.version} is invalid: "
                                + "; ".join(problems) + ".")

    # -- Coverage -------------------------------------------------------------------

    @property
    def covered_years(self) -> tuple[int, ...]:
        return tuple(sorted(self._years))

    @property
    def verified_years(self) -> tuple[int, ...]:
        return tuple(y for y in self.covered_years if self._years[y].verified)

    def year(self, year: int) -> YearCalendar:
        """The calendar for `year`, or CalendarError if it can't be used."""
        if year not in self._years:
            raise CalendarError(
                f"{year} is not covered by market calendar version {self.version} "
                f"(covers {', '.join(map(str, self.covered_years))}). Add it from the "
                "official NYSE holiday and early-close announcements; an uncovered "
                "year is never assumed to be a normal trading year.")
        entry = self._years[year]
        if not entry.verified and not self.allow_unverified:
            raise CalendarError(
                f"{year} in market calendar version {self.version} has not been "
                "verified against the official NYSE documents listed in its "
                "'sources'. Check it, then set \"verified\": true with verified_by, "
                "verified_on and an evidence entry.")
        return entry

    # -- Trading days -------------------------------------------------------------------

    def closure_reason(self, day: date) -> str | None:
        """Why the market is closed on `day`, or None if it is a trading day."""
        entry = self.year(_as_date(day).year)
        day = _as_date(day)
        if day.weekday() >= 5:
            return f"a {_WEEKDAYS[day.weekday()]}"
        if day in entry.holidays:
            return f"an NYSE holiday ({entry.holidays[day]})"
        if day in entry.special_closures:
            return f"an unscheduled NYSE closure ({entry.special_closures[day]})"
        return None

    def session(self, day: date) -> Session | None:
        """The regular session on `day`, or None if the market is closed."""
        day = _as_date(day)
        if self.closure_reason(day) is not None:
            return None
        entry = self.year(day.year)
        open_at = datetime.combine(day, REGULAR_OPEN, tzinfo=NEW_YORK)
        if day in entry.early_closes:
            close_time, note = entry.early_closes[day]
            close_at = datetime.combine(day, close_time, tzinfo=NEW_YORK)
            return Session(day, open_at, close_at, True, note)
        return Session(day, open_at, datetime.combine(day, REGULAR_CLOSE, tzinfo=NEW_YORK),
                       False)

    def sessions_between(self, start: date, end: date) -> list[Session]:
        """Every session from `start` to `end` inclusive (every year must be usable)."""
        start, end = _as_date(start), _as_date(end)
        if start > end:
            raise CalendarError(f"start {start} is after end {end}.")
        for year in range(start.year, end.year + 1):
            self.year(year)                       # refuse uncovered/unverified years first
        sessions, day = [], start
        while day <= end:
            session = self.session(day)
            if session is not None:
                sessions.append(session)
            day += timedelta(days=1)
        return sessions


def load_calendar(path: str = settings.MARKET_CALENDAR_FILE, *,
                  allow_unverified: bool = False) -> TradingCalendar:
    try:
        with open(path) as f:
            data = json.load(f)
    except OSError as error:
        raise CalendarError(f"Could not read market calendar {path}: {error}") from None
    except json.JSONDecodeError as error:
        raise CalendarError(f"Market calendar {path} is not valid JSON: {error}") from None
    return TradingCalendar(data, allow_unverified=allow_unverified, source=path)


# --- Time zone ---------------------------------------------------------------------------

def to_new_york(timestamp: datetime) -> datetime:
    """
    Express an aware timestamp in New York time (same instant, New York's
    offset). Timestamps without a time zone are refused, never guessed.
    """
    if not isinstance(timestamp, datetime) or timestamp.utcoffset() is None:
        raise MarketDataError(f"Timestamp {timestamp!r} has no time zone; it can't be "
                              "converted without guessing.")
    return timestamp.astimezone(NEW_YORK)


def new_york_offset_problem(timestamp: datetime) -> str | None:
    """
    None if `timestamp` carries New York's real UTC offset for that instant;
    otherwise an explanation. 09:30-05:00 in July is refused: in July New
    York is at -04:00, so that would really be 10:30 New York time.
    """
    if not isinstance(timestamp, datetime):
        return f"is not a datetime (got {timestamp!r})"
    offset = timestamp.utcoffset()
    if offset is None:
        return "has no time zone"
    expected = timestamp.astimezone(NEW_YORK).utcoffset()
    if offset != expected:
        return (f"has UTC offset {_format_offset(offset)}, but New York's offset at "
                f"that moment is {_format_offset(expected)}")
    return None


# --- The regular-session bar grid ------------------------------------------------------------

def check_interval(interval_minutes) -> int:
    if (isinstance(interval_minutes, bool) or not isinstance(interval_minutes, int)
            or interval_minutes not in SUPPORTED_INTERVALS):
        raise MarketDataError(
            f"Bar interval must be one of {SUPPORTED_INTERVALS} minutes "
            f"(got {interval_minutes!r}).")
    return interval_minutes


def expected_bar_starts(session: Session, interval_minutes: int) -> list[datetime]:
    """Start time of every bar that fits completely inside `session`."""
    step = timedelta(minutes=check_interval(interval_minutes))
    starts, start = [], session.open
    while start + step <= session.close:
        starts.append(start)
        start += step
    return starts


def bar_timestamp_problems(timestamp: datetime, interval_minutes: int,
                           calendar: TradingCalendar) -> list[str]:
    """
    Every reason `timestamp` is not a valid regular-session bar start.
    Raises CalendarError if its year is not covered or not verified.
    """
    check_interval(interval_minutes)
    offset_problem = new_york_offset_problem(timestamp)
    if offset_problem:
        return [offset_problem]
    local = timestamp.astimezone(NEW_YORK)
    problems = []
    if local.second or local.microsecond:
        problems.append("is not on a whole minute")
    reason = calendar.closure_reason(local.date())
    if reason is not None:
        return problems + [f"falls on {reason}, when the market is closed"]
    session = calendar.session(local.date())
    hours = (f"{session.open:%H:%M}-{session.close:%H:%M}"
             + (f", early close: {session.note}" if session.early_close else ""))
    if local < session.open or local + timedelta(minutes=interval_minutes) > session.close:
        problems.append(f"is outside regular trading hours ({hours})")
    elif ((local - session.open) // timedelta(minutes=1)) % interval_minutes:
        problems.append(f"is not on the {interval_minutes}-minute grid starting at 09:30")
    return problems


def check_bar_timestamps(timestamps, interval_minutes: int,
                         calendar: TradingCalendar) -> list[str]:
    """Problems for a whole list of bar starts, labelled 'Bar N (time): ...'."""
    problems = []
    for number, timestamp in enumerate(timestamps, start=1):
        label = timestamp.isoformat() if isinstance(timestamp, datetime) else repr(timestamp)
        for problem in bar_timestamp_problems(timestamp, interval_minutes, calendar):
            problems.append(f"Bar {number} ({label}) {problem}.")
    return problems


# --- Parsing helpers ------------------------------------------------------------------------

def _parse_year(label, entry):
    problems = []
    try:
        year = int(label)
        if str(year) != label or not 1990 <= year <= 2100:
            raise ValueError
    except (TypeError, ValueError):
        return [f"year key {label!r} must be a four-digit year"], None
    if not isinstance(entry, dict):
        return [f"{year} must be an object"], None
    unknown, missing = set(entry) - _YEAR_KEYS, _YEAR_KEYS - set(entry)
    if unknown or missing:
        return [f"{year}: unknown key(s) {sorted(unknown)}, missing key(s) {sorted(missing)}"], None

    verified = entry["verified"]
    if not isinstance(verified, bool):
        problems.append(f"{year}: verified must be true or false")
    elif verified:
        if not isinstance(entry["verified_by"], str) or not entry["verified_by"].strip():
            problems.append(f"{year}: verified years need verified_by")
        if _parse_iso_date(entry["verified_on"]) is None:
            problems.append(f"{year}: verified years need verified_on as YYYY-MM-DD")
    evidence = entry["evidence"]
    if (not isinstance(evidence, list)
            or not all(isinstance(e, str) and e.strip() for e in evidence)):
        problems.append(f"{year}: evidence must be a list of non-empty text")
        evidence = []
    elif verified is True and not evidence:
        problems.append(f"{year}: verified years need at least one evidence entry")
    sources = entry["sources"]
    if (not isinstance(sources, list) or not sources
            or not all(isinstance(s, str) and s.startswith("https://") for s in sources)):
        problems.append(f"{year}: sources must be a non-empty list of https links")
        sources = []

    seen = {}
    holidays = _parse_days(year, "holidays", entry["holidays"], seen, problems)
    special = _parse_days(year, "special_closures", entry["special_closures"], seen, problems)
    early = {}
    if not isinstance(entry["early_closes"], list):
        problems.append(f"{year}: early_closes must be a list")
    else:
        for item in entry["early_closes"]:
            if not isinstance(item, dict) or set(item) != {"date", "close", "name"}:
                problems.append(f"{year}: each early close needs exactly date, close and name")
                continue
            day = _check_day(year, "early_closes", item["date"], seen, problems)
            close_at = _parse_hhmm(item["close"])
            if close_at is None or not REGULAR_OPEN < close_at < REGULAR_CLOSE:
                problems.append(f"{year}: early close on {item['date']} must be a time "
                                "after 09:30 and before 16:00 (HH:MM)")
            elif day is not None:
                early[day] = (close_at, str(item["name"]))
    if problems:
        return problems, None
    return [], YearCalendar(year, verified, entry["verified_by"], entry["verified_on"],
                            tuple(evidence), tuple(sources), holidays, special, early)


def _parse_days(year, field, items, seen, problems) -> dict:
    days = {}
    if not isinstance(items, list):
        problems.append(f"{year}: {field} must be a list")
        return days
    for item in items:
        if not isinstance(item, dict) or set(item) != {"date", "name"}:
            problems.append(f"{year}: each entry in {field} needs exactly date and name")
            continue
        day = _check_day(year, field, item["date"], seen, problems)
        if day is not None:
            days[day] = str(item["name"])
    return days


def _check_day(year, field, text, seen, problems):
    day = _parse_iso_date(text)
    if day is None:
        problems.append(f"{year}: {field} date {text!r} must be YYYY-MM-DD")
        return None
    if day.year != year:
        problems.append(f"{year}: {field} date {text} is not in {year}")
        return None
    if day.weekday() >= 5:
        problems.append(f"{year}: {field} date {text} is a {_WEEKDAYS[day.weekday()]}")
        return None
    if day in seen:
        problems.append(f"{year}: {text} is listed in both {seen[day]} and {field}")
        return None
    seen[day] = field
    return day


def _parse_iso_date(text):
    if not isinstance(text, str):
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _parse_hhmm(text):
    if not isinstance(text, str) or len(text) != 5 or text[2] != ":":
        return None
    try:
        return time(int(text[:2]), int(text[3:]))
    except ValueError:
        return None


def _as_date(day) -> date:
    if isinstance(day, datetime) or not isinstance(day, date):
        raise CalendarError(f"Expected a date (got {day!r}).")
    return day


def _format_offset(offset: timedelta) -> str:
    minutes = int(offset.total_seconds() // 60)
    sign = "+" if minutes >= 0 else "-"
    return f"{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"
