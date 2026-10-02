"""
Tests for regular-session data-quality checks (src/market_data/quality.py).

All candles are generated locally on the NYSE grid. A verified COPY of the
calendar is used (see tests/test_sessions.py); the real calendar file stays
unverified.
"""

import json
import unittest
from datetime import date, datetime, timedelta, timezone

from src.market_data.candles import Candle, MarketDataError
from src.market_data.quality import ABSENT_BAR_NOTE, check_quality
from src.market_data.sessions import (CalendarError, TradingCalendar,
                                      expected_bar_starts)
from tests.test_sessions import REAL_DATA, verified_data

CAL = TradingCalendar(verified_data())
DAYS = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]


def grid(days=DAYS, interval=5):
    return [s for d in days for s in expected_bar_starts(CAL.session(d), interval)]


def candles_at(starts, zero_volume=()):
    out = []
    for i, ts in enumerate(starts):
        o = 100.0 + i * 0.01
        out.append(Candle(ts, o, o + 0.5, o - 0.5, o + 0.1,
                          0.0 if ts in zero_volume else 1000.0 + i))
    return out


def check(candles, interval=5, **kwargs):
    kwargs.setdefault("empty_bar_policy", "omitted")
    return check_quality(candles, interval, CAL, **kwargs)


class CompleteDataTests(unittest.TestCase):
    def test_complete_sessions_are_accepted(self):
        report = check(candles_at(grid()))
        self.assertTrue(report.accepted, report.rejection_reasons)
        self.assertEqual((report.bars, report.sessions_expected, report.missing_bars),
                         (234, 3, 0))
        self.assertEqual((report.first_session, report.last_session),
                         ("2026-01-05", "2026-01-07"))

    def test_early_close_day_is_complete_with_42_bars(self):
        report = check(candles_at(grid([date(2025, 11, 28)])))
        self.assertTrue(report.accepted)
        self.assertEqual(report.bars, 42)

    def test_all_supported_intervals(self):
        for interval in (1, 5, 15, 30):
            with self.subTest(interval=interval):
                self.assertTrue(check(candles_at(grid(interval=interval)), interval).accepted)


class MissingBarTests(unittest.TestCase):
    def test_one_interior_bar_missing(self):
        starts = grid()
        gone = starts[100]
        report = check(candles_at([s for s in starts if s != gone]))
        self.assertFalse(report.accepted)
        self.assertEqual(report.missing_bars, 1)
        self.assertEqual(len(report.gaps), 1)
        self.assertEqual(report.gaps[0].kind, "interior")
        self.assertEqual(report.gaps[0].first_missing, gone.isoformat())
        self.assertEqual(report.partial_sessions, ())

    def test_absent_bars_are_never_called_no_trades(self):
        starts = grid()
        reasons = " ".join(check(candles_at(starts[:50] + starts[51:])).rejection_reasons)
        self.assertIn(ABSENT_BAR_NOTE, reasons)
        self.assertIn("cause is unknown", reasons)
        self.assertIn("never treated as 'no trades'", reasons)

    def test_partial_sessions(self):
        starts = grid()
        late = check(candles_at(starts[3:]))                    # day 1 starts at 09:45
        self.assertEqual(late.partial_sessions, ("2026-01-05",))
        self.assertEqual(late.gaps[0].kind, "late_start")
        self.assertEqual(late.gaps[0].missing, 3)
        early = check(candles_at(starts[:-2]))                  # day 3 ends at 15:45
        self.assertEqual(early.partial_sessions, ("2026-01-07",))
        self.assertEqual(early.gaps[-1].kind, "early_end")
        self.assertFalse(late.accepted or early.accepted)

    def test_whole_missing_session(self):
        report = check(candles_at(grid([date(2026, 1, 5), date(2026, 1, 7)])))
        self.assertEqual(report.missing_sessions, ("2026-01-06",))
        self.assertEqual(report.missing_bars, 78)
        self.assertEqual(report.gaps[0].kind, "missing_session")

    def test_explicit_date_range_detects_missing_edge_sessions(self):
        report = check(candles_at(grid([date(2026, 1, 6)])),
                       start_date=date(2026, 1, 5), end_date=date(2026, 1, 7))
        self.assertEqual(report.missing_sessions, ("2026-01-05", "2026-01-07"))
        self.assertFalse(report.accepted)

    def test_bars_outside_an_explicit_date_range(self):
        report = check(candles_at(grid()), start_date=date(2026, 1, 5),
                       end_date=date(2026, 1, 6))
        self.assertTrue(any("outside the declared date range" in p
                            for p in report.timestamp_problems))


class ZeroVolumeTests(unittest.TestCase):
    def test_zero_volume_bars_are_kept_and_listed_not_missing(self):
        starts = grid()
        zero = {starts[10], starts[11]}
        report = check(candles_at(starts, zero_volume=zero))
        self.assertTrue(report.accepted)
        self.assertEqual(report.missing_bars, 0)
        self.assertEqual(set(report.zero_volume_bars), {s.isoformat() for s in zero})

    def test_zero_volume_bar_and_absent_bar_are_distinguished(self):
        starts = grid()
        report = check(candles_at(starts[:20] + starts[21:], zero_volume={starts[10]}))
        self.assertEqual(report.zero_volume_bars, (starts[10].isoformat(),))
        self.assertEqual(report.gaps[0].first_missing, starts[20].isoformat())


class InvalidDataTests(unittest.TestCase):
    def test_duplicates_and_out_of_order_are_rejected_not_fixed(self):
        candles = candles_at(grid())
        dup = candles[:5] + [candles[4]] + candles[5:]
        swapped = candles[:5] + [candles[6], candles[5]] + candles[7:]
        before = list(swapped)
        for label, data in [("duplicate", dup), ("order", swapped)]:
            with self.subTest(label):
                report = check(data)
                self.assertFalse(report.accepted)
                self.assertTrue(report.invalid_candles)
        self.assertEqual(swapped, before)                         # nothing reordered

    def test_invalid_ohlc_is_rejected(self):
        candles = candles_at(grid())
        bad = candles[:3] + [Candle(candles[3].timestamp, 100, 99, 101, 100, 5)] + candles[4:]
        self.assertTrue(check(bad).invalid_candles)

    def test_bad_timestamps(self):
        good = candles_at(grid())
        ny_offset = good[0].timestamp.utcoffset()
        cases = {
            "off grid": good[0].timestamp + timedelta(minutes=2),
            "after the close": good[77].timestamp + timedelta(minutes=5),
            "holiday": datetime(2026, 1, 19, 10, 0, tzinfo=timezone(ny_offset)),
            "wrong offset": datetime(2026, 1, 5, 9, 30, tzinfo=timezone(timedelta(hours=-4))),
        }
        for label, ts in cases.items():
            with self.subTest(label):
                extra = Candle(ts, 100, 101, 99, 100, 10)
                report = check(sorted(good + [extra], key=lambda c: c.timestamp))
                self.assertFalse(report.accepted)
                self.assertTrue(report.timestamp_problems)

    def test_no_bars(self):
        report = check([])
        self.assertFalse(report.accepted)
        self.assertIn("The data contains no bars.", report.rejection_reasons)


class IntervalTests(unittest.TestCase):
    def test_declared_interval_smaller_than_data(self):
        report = check(candles_at(grid()), interval=1)          # 5-minute data declared 1
        self.assertFalse(report.accepted)
        self.assertTrue(any("closest bars are 5 minutes apart" in f
                            for f in report.interval_findings))

    def test_mixed_intervals_across_sessions(self):
        day1 = grid([date(2026, 1, 5)])
        day2 = grid([date(2026, 1, 6)])[::2]                     # every other bar
        report = check(candles_at(day1 + day2))
        self.assertTrue(any(f.startswith("Mixed intervals") for f in report.interval_findings))

    def test_finer_data_than_declared_is_off_grid(self):
        report = check(candles_at(grid(interval=1)), interval=5)
        self.assertTrue(any("not on the 5-minute grid" in p for p in report.timestamp_problems))


class CalendarRefusalTests(unittest.TestCase):
    def test_unverified_calendar_years_are_refused(self):
        with self.assertRaises(CalendarError):
            check_quality(candles_at(grid()), 5, TradingCalendar(REAL_DATA),
                          empty_bar_policy="omitted")

    def test_uncovered_years_are_refused(self):
        late = datetime(2027, 1, 4, 9, 30, tzinfo=timezone(timedelta(hours=-5)))
        with self.assertRaises(CalendarError):
            check([Candle(late, 100, 101, 99, 100, 10)])

    def test_bad_arguments(self):
        with self.assertRaises(MarketDataError):
            check(candles_at(grid()), empty_bar_policy="guess")
        with self.assertRaises(MarketDataError):
            check(candles_at(grid()), interval=2)


class ReportTests(unittest.TestCase):
    def test_report_is_json_ready_and_summarised(self):
        starts = grid()
        report = check(candles_at(starts[:30] + starts[31:]))
        record = report.to_dict()
        json.dumps(record)
        self.assertFalse(record["accepted"])
        self.assertEqual(record["gaps"][0]["kind"], "interior")
        self.assertIn("leaves out intervals with no trades", record["empty_bar_policy_note"])
        self.assertIn("REJECTED", report.summary())


if __name__ == "__main__":
    unittest.main()
