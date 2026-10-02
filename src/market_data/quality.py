"""
quality.py - Checks a series of regular-session candles against the NYSE
calendar and reports every problem. It never fills, sorts, removes or
repairs anything.

What it checks:
    - invalid candles (missing values, NaN/infinity, impossible OHLC,
      duplicate or out-of-order timestamps) - via validation.find_problems
    - every bar start against the regular-session grid: New York offset,
      grid alignment, inside the session, not on a closed day
    - MISSING bars: every expected bar start in the date range that is not
      in the data, grouped as whole missing sessions, partial sessions
      (late start / early end) and gaps inside a session
    - interval problems: data whose bars are further apart than declared,
      or different sessions that look like different bar lengths
    - explicit zero-volume bars (kept, and listed separately)

An ABSENT bar is never described as "no trades". Some sources leave out
intervals with no trades, but a missing bar can also be lost data, and the
file alone cannot tell the two apart.

Policy: strict. Any finding except zero-volume bars rejects the data.
"""

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

from src.market_data.candles import Candle, MarketDataError
from src.market_data.sessions import (NEW_YORK, TradingCalendar,
                                      bar_timestamp_problems, check_interval,
                                      expected_bar_starts)
from src.market_data.validation import find_problems

EMPTY_BAR_POLICIES = ("omitted", "zero_volume_bars", "unknown")

ABSENT_BAR_NOTE = (
    "An absent bar means the source file has no row for that interval. Its "
    "cause is unknown: some sources leave out intervals with no trades, and "
    "data can also be lost. An absent bar is never treated as 'no trades' and "
    "is never filled in.")

EMPTY_BAR_POLICY_NOTES = {
    "omitted": "The source documents that it leaves out intervals with no "
               "trades, but that cannot be confirmed for any single absent bar.",
    "zero_volume_bars": "The source documents that it writes zero-volume bars "
                        "for intervals with no trades, so an absent bar is more "
                        "likely missing data - but this is not assumed.",
    "unknown": "How the source represents intervals with no trades is not known.",
}


@dataclass(frozen=True)
class SessionGap:
    """A run of consecutive absent bars inside one session."""
    day: str                   # trading day, YYYY-MM-DD
    kind: str                  # missing_session | late_start | early_end | interior
    missing: int               # number of absent bars in this run
    first_missing: str         # first absent bar start (ISO, New York time)
    last_missing: str          # last absent bar start


@dataclass(frozen=True)
class QualityReport:
    interval_minutes: int
    empty_bar_policy: str
    bars: int
    first_session: str | None
    last_session: str | None
    sessions_expected: int
    sessions_with_bars: int
    invalid_candles: tuple[str, ...]
    timestamp_problems: tuple[str, ...]
    missing_bars: int
    gaps: tuple[SessionGap, ...]
    missing_sessions: tuple[str, ...]
    partial_sessions: tuple[str, ...]
    interval_findings: tuple[str, ...]
    zero_volume_bars: tuple[str, ...]

    @property
    def rejection_reasons(self) -> list[str]:
        reasons = []
        if self.bars == 0:
            reasons.append("The data contains no bars.")
        reasons += list(self.invalid_candles)
        reasons += list(self.timestamp_problems)
        if self.missing_bars:
            reasons.append(
                f"{self.missing_bars} expected regular-session bar(s) are absent "
                f"({len(self.missing_sessions)} whole session(s) missing, "
                f"{len(self.partial_sessions)} partial session(s), "
                f"{sum(1 for g in self.gaps if g.kind == 'interior')} gap(s) inside "
                f"sessions). {ABSENT_BAR_NOTE}")
        reasons += list(self.interval_findings)
        return reasons

    @property
    def accepted(self) -> bool:
        return not self.rejection_reasons

    def to_dict(self) -> dict:
        record = asdict(self)
        record["gaps"] = [asdict(g) for g in self.gaps]
        record["accepted"] = self.accepted
        record["absent_bar_note"] = ABSENT_BAR_NOTE
        record["empty_bar_policy_note"] = EMPTY_BAR_POLICY_NOTES[self.empty_bar_policy]
        return record

    def summary(self, limit: int = 20) -> str:
        lines = [f"Quality: {'ACCEPTED' if self.accepted else 'REJECTED'} - {self.bars} bars, "
                 f"{self.sessions_with_bars} of {self.sessions_expected} expected session(s) "
                 f"with data ({self.first_session} to {self.last_session}), "
                 f"{self.interval_minutes}-minute bars"]
        reasons = self.rejection_reasons
        for reason in reasons[:limit]:
            lines.append(f"  - {reason}")
        if len(reasons) > limit:
            lines.append(f"  ... and {len(reasons) - limit} more")
        for gap in self.gaps[:limit]:
            lines.append(f"  absent: {gap.day} {gap.kind}, {gap.missing} bar(s) "
                         f"{gap.first_missing} .. {gap.last_missing}")
        if len(self.gaps) > limit:
            lines.append(f"  ... and {len(self.gaps) - limit} more gap(s)")
        if self.zero_volume_bars:
            lines.append(f"  {len(self.zero_volume_bars)} explicit zero-volume bar(s) kept "
                         "as they are (these are rows in the source, not absent bars).")
        return "\n".join(lines)


def check_quality(candles, interval_minutes: int, calendar: TradingCalendar, *,
                  empty_bar_policy: str, start_date: date | None = None,
                  end_date: date | None = None) -> QualityReport:
    """
    Check `candles` (oldest first) for the regular sessions from `start_date`
    to `end_date`. Without explicit dates, the range runs from the first to
    the last trading day that has bars. Raises CalendarError if any year in
    the range is not covered or not verified.
    """
    check_interval(interval_minutes)
    if empty_bar_policy not in EMPTY_BAR_POLICIES:
        raise MarketDataError(f"empty_bar_policy must be one of {EMPTY_BAR_POLICIES} "
                              f"(got {empty_bar_policy!r}).")
    if not isinstance(candles, (list, tuple)):
        raise MarketDataError("candles must be a list.")

    invalid = tuple(find_problems(list(candles)))

    # Per-bar timestamp checks, and the set of valid bar starts present.
    timestamp_problems, present = [], []
    for number, candle in enumerate(candles, start=1):
        ts = getattr(candle, "timestamp", None)
        if not isinstance(ts, datetime) or ts.utcoffset() is None:
            continue                                   # already in `invalid`
        problems = bar_timestamp_problems(ts, interval_minutes, calendar)
        for problem in problems:
            timestamp_problems.append(f"Bar {number} ({ts.isoformat()}) {problem}.")
        if not problems:
            present.append(ts.astimezone(NEW_YORK))
    present_set = set(present)

    days = sorted({ts.date() for ts in present})
    first = start_date or (days[0] if days else None)
    last = end_date or (days[-1] if days else None)
    sessions = calendar.sessions_between(first, last) if first and last else []
    if first and last:
        for number, ts in enumerate(present, start=1):
            if not first <= ts.date() <= last:
                timestamp_problems.append(
                    f"Bar ({ts.isoformat()}) is outside the declared date range "
                    f"{first} to {last}.")

    gaps, missing_sessions, partial_sessions = [], [], []
    min_steps, interval_findings = {}, []
    missing_total = 0
    for session in sessions:
        expected = expected_bar_starts(session, interval_minutes)
        absent = [i for i, start in enumerate(expected) if start not in present_set]
        missing_total += len(absent)
        day = session.day.isoformat()
        if len(absent) == len(expected):
            missing_sessions.append(day)
            gaps.append(SessionGap(day, "missing_session", len(absent),
                                   expected[0].isoformat(), expected[-1].isoformat()))
            continue
        partial = False
        for run in _runs(absent):
            if run[0] == 0:
                kind, partial = "late_start", True
            elif run[-1] == len(expected) - 1:
                kind, partial = "early_end", True
            else:
                kind = "interior"
            gaps.append(SessionGap(day, kind, len(run), expected[run[0]].isoformat(),
                                   expected[run[-1]].isoformat()))
        if partial:
            partial_sessions.append(day)

        day_bars = [ts for ts in present if ts.date() == session.day]
        steps = [int((b - a) // timedelta(minutes=1)) for a, b in zip(day_bars, day_bars[1:])
                 if b > a]
        if steps:
            min_steps[day] = min(steps)
            if min(steps) != interval_minutes:
                interval_findings.append(
                    f"Session {day}: the closest bars are {min(steps)} minutes apart, but "
                    f"the declared interval is {interval_minutes} minutes.")
    if len(set(min_steps.values())) > 1:
        spacing = ", ".join(f"{m} min" for m in sorted(set(min_steps.values())))
        interval_findings.append(
            f"Mixed intervals: sessions have different closest bar spacings ({spacing}).")

    zero_volume = tuple(c.timestamp.isoformat() for c in candles
                        if isinstance(c, Candle) and isinstance(c.timestamp, datetime)
                        and _is_zero(c.volume))

    return QualityReport(
        interval_minutes=interval_minutes, empty_bar_policy=empty_bar_policy,
        bars=len(candles),
        first_session=first.isoformat() if first else None,
        last_session=last.isoformat() if last else None,
        sessions_expected=len(sessions),
        sessions_with_bars=len(sessions) - len(missing_sessions),
        invalid_candles=invalid, timestamp_problems=tuple(timestamp_problems),
        missing_bars=missing_total, gaps=tuple(gaps),
        missing_sessions=tuple(missing_sessions), partial_sessions=tuple(partial_sessions),
        interval_findings=tuple(interval_findings), zero_volume_bars=zero_volume)


def _runs(indexes: list[int]) -> list[list[int]]:
    """[1,2,3,7,8] -> [[1,2,3],[7,8]]"""
    runs = []
    for i in indexes:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def _is_zero(volume) -> bool:
    return (not isinstance(volume, bool) and isinstance(volume, (int, float))
            and volume == 0)
