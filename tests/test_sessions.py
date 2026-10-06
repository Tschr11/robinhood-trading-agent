"""
Tests for the exchange calendar and regular-session grid (src/market_data/sessions.py).

Run from the project root with:
    python -m unittest discover tests -v

Most logic tests use a copy of config/market_calendar.json with every year
marked verified, because the real file's years stay unverified until a
person checks them against the official NYSE documents.
"""

import copy
import json
import unittest
from datetime import date, datetime, time, timedelta, timezone

from config import settings
from src.market_data.candles import MarketDataError
from src.market_data.sessions import (NEW_YORK, SUPPORTED_INTERVALS, CalendarError,
                                      TradingCalendar, bar_timestamp_problems,
                                      check_bar_timestamps, check_interval,
                                      expected_bar_starts, load_calendar,
                                      new_york_offset_problem, to_new_york)

with open(settings.MARKET_CALENDAR_FILE) as _f:
    REAL_DATA = json.load(_f)


def verified_data(data=None):
    """A copy of the calendar data with every year marked verified."""
    data = copy.deepcopy(data or REAL_DATA)
    for entry in data["years"].values():
        entry.update(verified=True, verified_by="test", verified_on="2026-10-02",
                     evidence=["test fixture - not a real verification"])
    return data


def unverified_data(data=None):
    """A copy of the calendar data with every year marked NOT verified."""
    data = copy.deepcopy(data or REAL_DATA)
    for entry in data["years"].values():
        entry.update(verified=False, verified_by=None, verified_on=None, evidence=[])
    return data


CAL = TradingCalendar(verified_data())


def ny(year, month, day, hour, minute, second=0):
    return datetime(year, month, day, hour, minute, second, tzinfo=NEW_YORK)


# --- The calendar file -------------------------------------------------------------------

class CalendarFileTests(unittest.TestCase):
    def test_file_loads_and_states_its_version_and_coverage(self):
        cal = load_calendar(allow_unverified=True)
        self.assertEqual(cal.covered_years, (2023, 2024, 2025, 2026))
        self.assertTrue(cal.version)
        self.assertEqual(REAL_DATA["timezone"], "America/New_York")
        self.assertEqual(REAL_DATA["regular_session"], {"open": "09:30", "close": "16:00"})

    def test_every_year_names_official_sources_and_its_verification_state(self):
        for year, entry in REAL_DATA["years"].items():
            with self.subTest(year=year):
                self.assertTrue(entry["sources"])
                self.assertTrue(all(s.startswith(("https://ir.theice.com/",
                                                  "https://www.nyse.com/"))
                                    for s in entry["sources"]))
                self.assertIsInstance(entry["evidence"], list)
                if entry["verified"]:
                    self.assertTrue(entry["verified_by"])
                    self.assertTrue(entry["verified_on"])
                    self.assertTrue(entry["evidence"])
                    # A real verification is never recorded under a placeholder name.
                    self.assertNotIn(entry["verified_by"].strip().lower(),
                                     {"test", "claude", "ai", "assistant", "me"})
                else:
                    self.assertIsNone(entry["verified_by"])
                    self.assertIsNone(entry["verified_on"])
                    self.assertEqual(entry["evidence"], [])

    def test_known_closures_and_early_closes(self):
        closed = [date(2023, 1, 2), date(2023, 4, 7), date(2024, 3, 29), date(2024, 6, 19),
                  date(2025, 1, 9), date(2025, 4, 18), date(2026, 4, 3), date(2026, 7, 3)]
        for day in closed:
            with self.subTest(day=day):
                self.assertIsNone(CAL.session(day))
        early = [date(2023, 7, 3), date(2023, 11, 24), date(2024, 7, 3), date(2024, 11, 29),
                 date(2024, 12, 24), date(2025, 7, 3), date(2025, 11, 28),
                 date(2025, 12, 24), date(2026, 11, 27), date(2026, 12, 24)]
        for day in early:
            with self.subTest(day=day):
                session = CAL.session(day)
                self.assertTrue(session.early_close)
                self.assertEqual(session.close.time(), time(13, 0))

    def test_trading_day_counts_per_year(self):
        # Weekdays minus holidays (and the unscheduled 2025-01-09 closure).
        expected = {2023: 250, 2024: 252, 2025: 250, 2026: 251}
        for year, count in expected.items():
            with self.subTest(year=year):
                self.assertEqual(len(CAL.sessions_between(date(year, 1, 1),
                                                          date(year, 12, 31))), count)


# --- Coverage and verification: never assume a normal year ---------------------------------

class CoverageTests(unittest.TestCase):
    def test_unverified_years_are_refused_by_default(self):
        cal = TradingCalendar(unverified_data())
        with self.assertRaises(CalendarError) as caught:
            cal.session(date(2025, 1, 6))
        self.assertIn("has not been verified", str(caught.exception))
        self.assertIn("official NYSE documents", str(caught.exception))

    def test_real_calendar_refuses_its_unverified_years(self):
        cal = load_calendar()
        unverified = [int(y) for y, e in REAL_DATA["years"].items() if not e["verified"]]
        self.assertTrue(unverified)
        for year in unverified:
            with self.subTest(year=year):
                with self.assertRaises(CalendarError):
                    cal.sessions_between(date(year, 1, 1), date(year, 12, 31))

    def test_unverified_years_can_be_used_only_when_explicitly_allowed(self):
        cal = TradingCalendar(unverified_data(), allow_unverified=True)
        self.assertIsNotNone(cal.session(date(2025, 1, 6)))

    def test_a_verified_year_needs_no_override(self):
        data = unverified_data()
        data["years"]["2025"].update(verified=True, verified_by="me", verified_on="2026-10-02",
                                     evidence=["checked against the 2025 PDF"])
        cal = TradingCalendar(data)
        self.assertEqual(cal.verified_years, (2025,))
        self.assertIsNotNone(cal.session(date(2025, 1, 6)))
        with self.assertRaises(CalendarError):
            cal.session(date(2024, 1, 8))               # still unverified

    def test_uncovered_years_are_refused_even_with_the_override(self):
        cal = TradingCalendar(REAL_DATA, allow_unverified=True)
        for day in [date(2022, 12, 30), date(2027, 1, 4), date(2027, 1, 2)]:   # incl. a Saturday
            with self.subTest(day=day):
                with self.assertRaises(CalendarError) as caught:
                    cal.session(day)
                self.assertIn("never assumed to be a normal trading year", str(caught.exception))

    def test_ranges_touching_an_uncovered_year_are_refused_up_front(self):
        with self.assertRaises(CalendarError):
            CAL.sessions_between(date(2026, 12, 1), date(2027, 1, 15))
        with self.assertRaises(CalendarError):
            CAL.sessions_between(date(2026, 2, 1), date(2026, 1, 1))     # start after end

    def test_dates_must_be_dates(self):
        for bad in [datetime(2025, 1, 6, 10, 0), "2025-01-06", None]:
            with self.subTest(day=bad):
                with self.assertRaises(CalendarError):
                    CAL.session(bad)


class CalendarDataValidationTests(unittest.TestCase):
    def broken(self, change):
        data = verified_data()
        change(data)
        with self.assertRaises(CalendarError):
            TradingCalendar(data)

    def test_invalid_calendar_data_is_refused(self):
        y = lambda d: d["years"]["2025"]
        cases = {
            "unknown top key": lambda d: d.update(extra=1),
            "missing years": lambda d: d.pop("years"),
            "wrong exchange": lambda d: d.update(calendar="XNAS"),
            "wrong time zone": lambda d: d.update(timezone="UTC"),
            "wrong hours": lambda d: d.update(regular_session={"open": "09:00", "close": "16:00"}),
            "empty version": lambda d: d.update(version=""),
            "two-digit year": lambda d: d["years"].update({"25": d["years"].pop("2025")}),
            "unknown year key": lambda d: y(d).update(extra=1),
            "missing year key": lambda d: y(d).pop("early_closes"),
            "weekend holiday": lambda d: y(d)["holidays"].append(
                {"date": "2025-01-04", "name": "Saturday"}),
            "holiday in another year": lambda d: y(d)["holidays"].append(
                {"date": "2024-12-31", "name": "Wrong year"}),
            "bad date text": lambda d: y(d)["holidays"].append(
                {"date": "01/02/2025", "name": "Bad"}),
            "date listed twice": lambda d: y(d)["early_closes"].append(
                {"date": "2025-12-25", "close": "13:00", "name": "Also a holiday"}),
            "early close at 16:00": lambda d: y(d)["early_closes"][0].update(close="16:00"),
            "early close before open": lambda d: y(d)["early_closes"][0].update(close="09:00"),
            "early close bad format": lambda d: y(d)["early_closes"][0].update(close="1pm"),
            "verified without who": lambda d: y(d).update(verified_by=None),
            "verified without when": lambda d: y(d).update(verified_on="yesterday"),
            "verified not a bool": lambda d: y(d).update(verified="yes"),
            "verified without evidence": lambda d: y(d).update(evidence=[]),
            "evidence not a list": lambda d: y(d).update(evidence="checked"),
            "blank evidence": lambda d: y(d).update(evidence=["  "]),
            "evidence not text": lambda d: y(d).update(evidence=[1]),
            "missing evidence key": lambda d: y(d).pop("evidence"),
            "no sources": lambda d: y(d).update(sources=[]),
            "insecure source": lambda d: y(d).update(sources=["http://example.com"]),
            "holiday missing name": lambda d: y(d)["holidays"].append({"date": "2025-03-03"}),
        }
        for label, change in cases.items():
            with self.subTest(label):
                self.broken(change)

    def test_unreadable_file_is_refused(self):
        with self.assertRaises(CalendarError):
            load_calendar("does/not/exist.json")
        with self.assertRaises(CalendarError):
            TradingCalendar(["not", "an", "object"])


# --- Sessions -------------------------------------------------------------------------------

class SessionTests(unittest.TestCase):
    def test_normal_day(self):
        session = CAL.session(date(2026, 1, 5))
        self.assertEqual(session.open, ny(2026, 1, 5, 9, 30))
        self.assertEqual(session.close, ny(2026, 1, 5, 16, 0))
        self.assertFalse(session.early_close)
        self.assertEqual(session.minutes, 390)

    def test_bar_counts_on_a_normal_day(self):
        session = CAL.session(date(2026, 1, 5))
        bars = expected_bar_starts(session, 5)
        self.assertEqual(len(bars), 78)
        self.assertEqual(bars[0], ny(2026, 1, 5, 9, 30))
        self.assertEqual(bars[-1], ny(2026, 1, 5, 15, 55))
        self.assertEqual({m: len(expected_bar_starts(session, m)) for m in SUPPORTED_INTERVALS},
                         {1: 390, 5: 78, 15: 26, 30: 13})

    def test_early_close_day(self):
        session = CAL.session(date(2025, 11, 28))
        self.assertTrue(session.early_close)
        self.assertEqual(session.note, "Day after Thanksgiving")
        bars = expected_bar_starts(session, 5)
        self.assertEqual(len(bars), 42)
        self.assertEqual(bars[-1], ny(2025, 11, 28, 12, 55))

    def test_closed_days_and_reasons(self):
        cases = {date(2026, 1, 3): "a Saturday", date(2026, 1, 4): "a Sunday",
                 date(2026, 1, 19): "an NYSE holiday (Martin Luther King, Jr. Day)",
                 date(2025, 1, 9): "an unscheduled NYSE closure"}
        for day, reason in cases.items():
            with self.subTest(day=day):
                self.assertIsNone(CAL.session(day))
                self.assertIn(reason, CAL.closure_reason(day))
        self.assertIsNone(CAL.closure_reason(date(2026, 1, 5)))

    def test_daylight_saving_offsets(self):
        cases = {date(2025, 3, 7): "-05:00", date(2025, 3, 10): "-04:00",     # spring forward
                 date(2025, 10, 31): "-04:00", date(2025, 11, 3): "-05:00"}    # fall back
        for day, offset in cases.items():
            with self.subTest(day=day):
                self.assertTrue(CAL.session(day).open.isoformat().endswith(offset))
        # 09:30 New York is 14:30 UTC in winter and 13:30 UTC in summer.
        self.assertEqual(CAL.session(date(2025, 3, 7)).open.astimezone(timezone.utc).hour, 14)
        self.assertEqual(CAL.session(date(2025, 3, 10)).open.astimezone(timezone.utc).hour, 13)

    def test_sessions_between(self):
        week = CAL.sessions_between(date(2025, 1, 6), date(2025, 1, 12))
        self.assertEqual([s.day.day for s in week], [6, 7, 8, 10])          # 9th closed


# --- Time zone -------------------------------------------------------------------------------

class TimeZoneTests(unittest.TestCase):
    def test_to_new_york_keeps_the_instant(self):
        utc = datetime(2025, 7, 7, 13, 30, tzinfo=timezone.utc)
        local = to_new_york(utc)
        self.assertEqual(local, utc)
        self.assertEqual(local.isoformat(), "2025-07-07T09:30:00-04:00")
        self.assertEqual(to_new_york(datetime(2025, 1, 6, 14, 30, tzinfo=timezone.utc))
                         .isoformat(), "2025-01-06T09:30:00-05:00")

    def test_naive_timestamps_are_never_guessed(self):
        for bad in [datetime(2025, 7, 7, 9, 30), "2025-07-07T09:30", None]:
            with self.subTest(value=bad):
                with self.assertRaises(MarketDataError):
                    to_new_york(bad)

    def test_offset_must_be_new_yorks_real_offset(self):
        self.assertIsNone(new_york_offset_problem(ny(2025, 7, 7, 9, 30)))
        self.assertIsNone(new_york_offset_problem(
            datetime(2025, 1, 6, 9, 30, tzinfo=timezone(timedelta(hours=-5)))))
        wrong_summer = datetime(2025, 7, 7, 9, 30, tzinfo=timezone(timedelta(hours=-5)))
        self.assertIn("New York's offset at that moment is -04:00",
                      new_york_offset_problem(wrong_summer))
        utc_same_instant = datetime(2025, 7, 7, 13, 30, tzinfo=timezone.utc)
        self.assertIn("+00:00", new_york_offset_problem(utc_same_instant))
        self.assertEqual(new_york_offset_problem(datetime(2025, 7, 7, 9, 30)), "has no time zone")


# --- The regular-session grid -----------------------------------------------------------------

class GridTests(unittest.TestCase):
    def problems(self, timestamp, interval=5):
        return bar_timestamp_problems(timestamp, interval, CAL)

    def test_every_expected_bar_in_every_covered_year_is_valid(self):
        sessions = CAL.sessions_between(date(2023, 1, 1), date(2026, 12, 31))
        for session in sessions:
            for start in expected_bar_starts(session, 5):
                problems = self.problems(start)
                if problems:
                    self.fail(f"{start.isoformat()}: {problems}")

    def test_fixed_offset_timestamps_with_the_right_offset_are_valid(self):
        winter = timezone(timedelta(hours=-5))
        self.assertEqual(self.problems(datetime(2026, 1, 5, 9, 30, tzinfo=winter)), [])

    def test_off_grid_bars(self):
        self.assertIn("is not on the 5-minute grid starting at 09:30",
                      self.problems(ny(2026, 1, 5, 9, 32)))
        self.assertIn("is not on the 30-minute grid starting at 09:30",
                      self.problems(ny(2026, 1, 5, 10, 15), 30))
        self.assertIn("is not on a whole minute", self.problems(ny(2026, 1, 5, 9, 35, 30)))
        self.assertEqual(self.problems(ny(2026, 1, 5, 9, 32), 1), [])

    def test_bars_outside_regular_hours(self):
        for start in [ny(2026, 1, 5, 9, 25), ny(2026, 1, 5, 4, 0), ny(2026, 1, 5, 16, 0),
                      ny(2026, 1, 5, 19, 55)]:
            with self.subTest(start=start.time()):
                self.assertTrue(any("outside regular trading hours" in p
                                    for p in self.problems(start)))
        self.assertEqual(self.problems(ny(2026, 1, 5, 15, 55)), [])

    def test_bars_after_an_early_close(self):
        problems = self.problems(ny(2025, 11, 28, 13, 0))
        self.assertTrue(any("09:30-13:00, early close: Day after Thanksgiving" in p
                            for p in problems))
        self.assertEqual(self.problems(ny(2025, 11, 28, 12, 55)), [])
        self.assertTrue(self.problems(ny(2025, 11, 28, 12, 45), 30))   # 12:45-13:15 crosses

    def test_bars_on_closed_days(self):
        for start, reason in [(ny(2026, 1, 3, 10, 0), "a Saturday"),
                              (ny(2026, 1, 19, 10, 0), "Martin Luther King"),
                              (ny(2025, 1, 9, 10, 0), "unscheduled NYSE closure")]:
            with self.subTest(start=start.date()):
                problems = self.problems(start)
                self.assertTrue(any(reason in p and "market is closed" in p for p in problems))

    def test_wrong_offset_or_missing_zone(self):
        wrong = datetime(2025, 7, 7, 9, 30, tzinfo=timezone(timedelta(hours=-5)))
        self.assertEqual(len(self.problems(wrong)), 1)
        self.assertIn("UTC offset -05:00", self.problems(wrong)[0])
        self.assertEqual(self.problems(datetime(2025, 7, 7, 9, 30)), ["has no time zone"])
        self.assertTrue(self.problems("2025-07-07 09:30")[0].startswith("is not a datetime"))

    def test_uncovered_year_is_an_error_not_a_problem_list(self):
        with self.assertRaises(CalendarError):
            self.problems(ny(2027, 1, 4, 9, 30))

    def test_only_supported_intervals(self):
        for bad in [2, 60, 0, True, "5", 5.0]:
            with self.subTest(interval=bad):
                with self.assertRaises(MarketDataError):
                    check_interval(bad)
                with self.assertRaises(MarketDataError):
                    expected_bar_starts(CAL.session(date(2026, 1, 5)), bad)

    def test_list_check_labels_each_bar(self):
        stamps = (ny(2026, 1, 5, 9, 30), ny(2026, 1, 5, 9, 33), ny(2026, 1, 3, 9, 30))
        problems = check_bar_timestamps(stamps, 5, CAL)
        self.assertEqual(len(problems), 2)
        self.assertTrue(problems[0].startswith("Bar 2 (2026-01-05T09:33:00-05:00)"))
        self.assertTrue(problems[1].startswith("Bar 3 (2026-01-03T09:30:00-05:00)"))
        self.assertEqual(len(stamps), 3)                     # input untouched

    def test_expected_grid_is_a_fresh_list(self):
        session = CAL.session(date(2026, 1, 5))
        first = expected_bar_starts(session, 5)
        first.clear()
        self.assertEqual(len(expected_bar_starts(session, 5)), 78)


if __name__ == "__main__":
    unittest.main()
