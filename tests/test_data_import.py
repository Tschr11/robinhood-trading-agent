"""
Tests for the offline historical-data importer (src/data_import/) and the
verified dataset reader (src/market_data/manifest.py).

Every raw file is generated locally in a temporary folder. A verified COPY
of the calendar is passed in; the real calendar file stays unverified, and
one test proves real imports are refused because of that.
"""

import json
import os
import stat
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from unittest import mock

from config import settings
from src.data_import import importer as importer_module
from src.data_import.importer import import_csv, main
from src.data_import.reader import ImportRejected
from src.data_import.spec import ImportSpecError, load_spec, spec_from_dict
from src.market_data.candles import Candle, DataKind
from src.market_data.manifest import (DatasetIntegrityError, ManifestCSVProvider,
                                      REQUIRED_MANIFEST_KEYS, canonical_csv_bytes,
                                      load_verified_dataset, sha256_hex)
from src.market_data.sessions import (NEW_YORK, CalendarError, TradingCalendar,
                                      bar_timestamp_problems, expected_bar_starts)
from tests.test_sessions import REAL_DATA, verified_data

CAL = TradingCalendar(verified_data())
DAYS = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
FIXED_NOW = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)

BASE_SPEC = {
    "dataset_id": "spy_5m_test", "symbol": "SPY", "source": "locally generated test file",
    "interval_minutes": 5,
    "columns": {"timestamp": "Date", "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume"},
    "timestamp_format": "iso8601", "source_timezone": "America/New_York",
    "timestamp_convention": "bar_start", "adjustment": "split_adjusted",
    "volume_coverage": "consolidated", "empty_bar_policy": "omitted",
    "outside_regular_hours": "reject",
}


def spec(**changes):
    raw = json.loads(json.dumps(BASE_SPEC))
    raw.update(changes)
    return spec_from_dict(raw)


def bars(days=DAYS, interval=5, drop=(), zero_volume=()):
    """(start, open, high, low, close, volume) rows on the regular grid."""
    rows = []
    for i, start in enumerate(s for d in days
                              for s in expected_bar_starts(CAL.session(d), interval)):
        if start in drop:
            continue
        o = round(100 + i * 0.01, 2)
        rows.append((start, o, round(o + 0.5, 2), round(o - 0.5, 2), round(o + 0.1, 2),
                     0 if start in zero_volume else 1000 + i))
    return rows


def render(rows, style="ny_naive", interval=5, header="Date,Open,High,Low,Close,Volume",
           extra_column=False):
    lines = [header + (",Note" if extra_column else "")]
    for start, o, h, l, c, v in rows:
        if style == "ny_naive":
            ts = start.astimezone(NEW_YORK).strftime("%Y-%m-%d %H:%M:%S")
        elif style == "ny_naive_bar_end":
            ts = (start + timedelta(minutes=interval)).astimezone(NEW_YORK) \
                .strftime("%Y-%m-%d %H:%M:%S")
        elif style == "utc_ms":
            ts = str((start - EPOCH) // timedelta(milliseconds=1))
        elif style == "utc_z":
            ts = start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        elif style == "offset":
            ts = start.astimezone(NEW_YORK).isoformat()
        lines.append(f"{ts},{o:.2f},{h:.2f},{l:.2f},{c:.2f},{v}" + (",x" if extra_column else ""))
    return "\n".join(lines) + "\n"


def expected_canonical(rows):
    return canonical_csv_bytes([Candle(s.astimezone(NEW_YORK), float(f"{o:.2f}"),
                                       float(f"{h:.2f}"), float(f"{l:.2f}"),
                                       float(f"{c:.2f}"), float(v))
                                for s, o, h, l, c, v in rows])


class ImportTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_dir = os.path.join(self.tmp.name, "historical_data")

    def raw(self, text, name="vendor_SPY.csv"):
        path = os.path.join(self.tmp.name, name)
        mode = "wb" if isinstance(text, bytes) else "w"
        with open(path, mode) as f:
            f.write(text)
        return path

    def do_import(self, path, the_spec=None, **kwargs):
        kwargs.setdefault("calendar", CAL)
        kwargs.setdefault("data_dir", self.data_dir)
        kwargs.setdefault("now", FIXED_NOW)
        return import_csv(path, the_spec or spec(), **kwargs)

    def assertNothingWritten(self):
        datasets = os.path.join(self.data_dir, "datasets")
        committed = []
        if os.path.isdir(datasets):
            for name in os.listdir(datasets):
                committed += os.listdir(os.path.join(datasets, name))
        self.assertEqual(committed, [], "a rejected import left files behind")
        raw_dir = os.path.join(self.data_dir, "raw")
        self.assertFalse(os.path.isdir(raw_dir) and os.listdir(raw_dir))

    def assertRejected(self, path, the_spec=None, words="", **kwargs):
        with self.assertRaises(ImportRejected) as caught:
            self.do_import(path, the_spec, **kwargs)
        text = " ".join(caught.exception.problems)
        self.assertIn(words, text)
        self.assertNothingWritten()
        return caught.exception


# --- Import spec ---------------------------------------------------------------------------

class SpecTests(unittest.TestCase):
    def test_valid_spec(self):
        s = spec(start_date="2026-01-05", end_date="2026-01-07", symbol=" spy ")
        self.assertEqual(s.symbol, "SPY")
        self.assertEqual(s.start_date, date(2026, 1, 5))
        self.assertEqual(s.header_for("timestamp"), "Date")
        self.assertEqual(spec().fingerprint(), spec().fingerprint())
        self.assertNotEqual(spec().fingerprint(), spec(adjustment="unadjusted").fingerprint())

    def test_every_required_field_must_be_given(self):
        for key in sorted(BASE_SPEC):
            with self.subTest(missing=key):
                raw = dict(BASE_SPEC)
                raw.pop(key)
                with self.assertRaises(ImportSpecError) as caught:
                    spec_from_dict(raw)
                self.assertIn("nothing is defaulted or guessed", str(caught.exception))

    def test_ambiguous_or_invalid_specs_are_refused(self):
        cases = [dict(extra=1), dict(interval_minutes=60), dict(interval_minutes=True),
                 dict(timestamp_format="auto"), dict(source_timezone="EST"),
                 dict(timestamp_convention="bar_middle"), dict(adjustment="adjusted"),
                 dict(adjustment=None), dict(volume_coverage="all"),
                 dict(empty_bar_policy="no_trades"), dict(outside_regular_hours="keep"),
                 dict(timestamp_format="unix_seconds"),              # with New York zone
                 dict(source_timezone="explicit_offset", timestamp_format="unix_seconds"),
                 dict(columns={"timestamp": "Date"}),
                 dict(columns=dict(BASE_SPEC["columns"], close="Open")),
                 dict(dataset_id="../x"), dict(dataset_id="SPY"), dict(dataset_id=""),
                 dict(symbol="../etc"), dict(source=" "),
                 dict(start_date="2026-01-07", end_date="2026-01-05"),
                 dict(start_date="01/05/2026"), dict(notes=5)]
        for change in cases:
            with self.subTest(change=change):
                with self.assertRaises(ImportSpecError):
                    spec(**change)

    def test_load_spec_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "spec.json")
            with open(path, "w") as f:
                json.dump(BASE_SPEC, f)
            self.assertEqual(load_spec(path).dataset_id, "spy_5m_test")
            with open(path, "w") as f:
                f.write("{broken")
            with self.assertRaises(ImportSpecError):
                load_spec(path)


# --- Successful imports ---------------------------------------------------------------------

class NormalImportTests(ImportTestCase):
    def test_import_creates_a_verified_canonical_dataset(self):
        rows = bars()
        result = self.do_import(self.raw(render(rows)))
        self.assertEqual((result.status, result.version), ("imported", 1))
        with open(os.path.join(result.folder, "SPY.csv"), "rb") as f:
            self.assertEqual(f.read(), expected_canonical(rows))
        m = result.manifest
        self.assertEqual(REQUIRED_MANIFEST_KEYS, set(m))
        self.assertEqual((m["symbol"], m["source"], m["interval_minutes"], m["timezone"],
                          m["timestamp_convention"], m["adjustment"], m["volume_coverage"],
                          m["empty_bar_policy"], m["session"]),
                         ("SPY", "locally generated test file", 5, "America/New_York",
                          "bar_start", "split_adjusted", "consolidated", "omitted",
                          "regular_trading_hours"))
        self.assertEqual(m["date_range"]["first_bar"], "2026-01-05T09:30:00-05:00")
        self.assertEqual(m["date_range"]["last_bar"], "2026-01-07T15:55:00-05:00")
        self.assertEqual(m["date_range"]["sessions"], 3)
        self.assertEqual(m["canonical_file"]["rows"], 234)
        self.assertTrue(m["quality"]["accepted"])
        self.assertEqual(m["calendar"]["years_used"], [2026])
        self.assertEqual(m["imported_at"], FIXED_NOW.isoformat())

    def test_every_timestamp_format_gives_the_same_canonical_bytes(self):
        rows = bars()
        expected = expected_canonical(rows)
        variants = {
            "ny_naive": spec(),
            "utc_ms": spec(timestamp_format="unix_milliseconds", source_timezone="UTC"),
            "utc_z": spec(source_timezone="UTC"),
            "offset": spec(source_timezone="explicit_offset"),
            "ny_naive_bar_end": spec(timestamp_convention="bar_end"),
        }
        for style, the_spec in variants.items():
            with self.subTest(style=style):
                result = self.do_import(self.raw(render(rows, style), f"{style}.csv"),
                                        the_spec, data_dir=os.path.join(self.tmp.name, style))
                with open(os.path.join(result.folder, "SPY.csv"), "rb") as f:
                    self.assertEqual(f.read(), expected)

    def test_canonical_timestamps_are_valid_new_york_bar_starts(self):
        self.do_import(self.raw(render(bars(), "utc_ms")),
                       spec(timestamp_format="unix_milliseconds", source_timezone="UTC"))
        data = ManifestCSVProvider("spy_5m_test", self.data_dir).get_candles("SPY")
        for c in data.candles:
            self.assertEqual(bar_timestamp_problems(c.timestamp, 5, CAL), [])
            self.assertEqual(c.timestamp.utcoffset(), timedelta(hours=-5))

    def test_explicit_extended_hours_exclusion_is_counted(self):
        rows = bars()
        pre = [(rows[0][0] - timedelta(minutes=5 * k),) + rows[0][1:] for k in (2, 1)]
        post = [(rows[-1][0] + timedelta(minutes=5 * k),) + rows[-1][1:] for k in (1, 2)]
        text = render(pre + rows + post)
        # Without the explicit setting, the same file is rejected ...
        self.assertRejected(self.raw(text, "reject.csv"), words="outside regular trading hours")
        # ... with it, the extended-hours rows are left out and counted.
        result = self.do_import(self.raw(text), spec(outside_regular_hours="exclude"))
        self.assertEqual(result.manifest["filtering"]["excluded_outside_regular_hours"], 4)
        with open(os.path.join(result.folder, "SPY.csv"), "rb") as f:
            self.assertEqual(f.read(), expected_canonical(rows))

    def test_date_range_filter_is_counted(self):
        result = self.do_import(self.raw(render(bars())),
                                spec(start_date="2026-01-06", end_date="2026-01-07"))
        self.assertEqual(result.manifest["filtering"]["excluded_outside_date_range"], 78)
        self.assertEqual(result.manifest["canonical_file"]["rows"], 156)

    def test_zero_volume_bars_are_kept_and_recorded(self):
        rows = bars()
        zero = {rows[5][0]}
        result = self.do_import(self.raw(render(bars(zero_volume=zero))))
        self.assertEqual(result.manifest["quality"]["zero_volume_bars"],
                         [rows[5][0].isoformat()])
        self.assertEqual(result.manifest["quality"]["missing_bars"], 0)

    def test_extra_columns_are_ignored_and_recorded(self):
        result = self.do_import(self.raw(render(bars(), extra_column=True)))
        self.assertEqual(result.manifest["filtering"]["ignored_columns"], ["Note"])

    def test_utf8_byte_order_mark_is_accepted(self):
        result = self.do_import(self.raw(b"\xef\xbb\xbf" + render(bars()).encode()))
        self.assertEqual(result.status, "imported")


class RawFileTests(ImportTestCase):
    def test_raw_input_is_never_modified(self):
        path = self.raw(render(bars()))
        with open(path, "rb") as f:
            before = f.read()
        before_stat = os.stat(path)
        result = self.do_import(path)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), before)
        self.assertEqual(os.stat(path).st_mtime_ns, before_stat.st_mtime_ns)
        self.assertEqual(result.manifest["source_file"]["sha256"], sha256_hex(before))

    def test_exact_read_only_copy_is_stored_by_fingerprint(self):
        path = self.raw(render(bars()))
        result = self.do_import(path)
        stored = os.path.join(self.data_dir, result.manifest["source_file"]["stored_as"])
        with open(stored, "rb") as f, open(path, "rb") as g:
            self.assertEqual(f.read(), g.read())
        self.assertFalse(os.stat(stored).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    def test_damaged_stored_raw_copy_is_detected(self):
        path = self.raw(render(bars()))
        result = self.do_import(path)
        stored = os.path.join(self.data_dir, result.manifest["source_file"]["stored_as"])
        os.chmod(stored, 0o644)
        with open(stored, "ab") as f:
            f.write(b"tampered")
        with self.assertRaises(ImportRejected):
            self.do_import(path, spec(notes="re-import"), new_version=True)


class DeterminismTests(ImportTestCase):
    def test_identical_inputs_give_identical_output_and_metadata(self):
        text = render(bars())
        a = self.do_import(self.raw(text), data_dir=os.path.join(self.tmp.name, "a"))
        b = self.do_import(self.raw(text), data_dir=os.path.join(self.tmp.name, "b"),
                           now=FIXED_NOW + timedelta(days=5))
        for name in ("SPY.csv",):
            with open(os.path.join(a.folder, name), "rb") as f, \
                 open(os.path.join(b.folder, name), "rb") as g:
                self.assertEqual(f.read(), g.read())
        diff = {k for k in a.manifest if a.manifest[k] != b.manifest[k]}
        self.assertEqual(diff, {"imported_at"})
        self.assertEqual(a.manifest["content_fingerprint"], b.manifest["content_fingerprint"])

    def test_returned_manifest_equals_the_stored_manifest(self):
        result = self.do_import(self.raw(render(bars())))
        with open(os.path.join(result.folder, "manifest.json")) as f:
            self.assertEqual(json.load(f), result.manifest)

    def test_reimporting_identical_data_changes_nothing(self):
        path = self.raw(render(bars()))
        first = self.do_import(path)
        before = sorted(os.walk(self.data_dir))
        again = self.do_import(path, now=FIXED_NOW + timedelta(hours=1))
        self.assertEqual((again.status, again.version), ("unchanged", 1))
        self.assertEqual(sorted(os.walk(self.data_dir)), before)
        self.assertEqual(again.manifest["imported_at"], first.manifest["imported_at"])


# --- Rejected imports ------------------------------------------------------------------------

class RejectionTests(ImportTestCase):
    def test_missing_bar(self):
        rows = bars()
        error = self.assertRejected(self.raw(render(bars(drop={rows[100][0]}))),
                                    words="absent")
        self.assertEqual(error.report.missing_bars, 1)

    def test_partial_and_missing_sessions(self):
        rows = bars()
        self.assertRejected(self.raw(render(rows[3:])), words="partial session")
        self.assertRejected(self.raw(render(rows), "b.csv"),
                            spec(start_date="2026-01-02", end_date="2026-01-07"),
                            words="whole session(s) missing")

    def test_duplicates_and_out_of_order_rows(self):
        rows = bars()
        self.assertRejected(self.raw(render(rows[:5] + [rows[4]] + rows[5:])),
                            words="duplicate timestamp")
        self.assertRejected(self.raw(render(rows[:5] + [rows[6], rows[5]] + rows[7:]), "b.csv"),
                            words="out of order")

    def test_invalid_ohlc(self):
        rows = bars()
        bad = list(rows)
        bad[3] = (rows[3][0], 100.0, 99.0, 101.0, 100.0, 5)
        self.assertRejected(self.raw(render(bad)), words="is below")

    def test_off_grid_out_of_session_and_closed_day_bars(self):
        rows = bars()
        off_grid = (rows[1][0] + timedelta(minutes=2),) + rows[1][1:]
        holiday = (datetime(2026, 1, 19, 10, 0, tzinfo=NEW_YORK),) + rows[0][1:]
        self.assertRejected(self.raw(render(rows[:2] + [off_grid] + rows[2:])),
                            words="not on the 5-minute grid")
        for policy in ("reject", "exclude"):
            with self.subTest(policy=policy):
                self.assertRejected(self.raw(render(rows + [holiday]), f"{policy}.csv"),
                                    spec(outside_regular_hours=policy),
                                    words="market is closed")

    def test_mislabelled_conventions_and_zones_are_caught(self):
        rows = bars()
        text = render(rows)
        self.assertRejected(self.raw(text), spec(timestamp_convention="bar_end"))
        self.assertRejected(self.raw(text, "b.csv"),
                            spec(timestamp_convention="bar_end", outside_regular_hours="exclude"),
                            words="absent")
        self.assertRejected(self.raw(text, "c.csv"), spec(source_timezone="UTC"))
        self.assertRejected(self.raw(text, "d.csv"), spec(interval_minutes=1),
                            words="closest bars are 5 minutes apart")

    def test_ambiguous_timestamps(self):
        rows = bars()
        offsets = render(rows, "offset")
        self.assertRejected(self.raw(offsets), words="carries a UTC offset")       # spec: naive NY
        mixed = render(rows[:1], "offset") + "\n".join(render(rows[1:]).splitlines()[1:]) + "\n"
        self.assertRejected(self.raw(mixed, "mixed.csv"), spec(source_timezone="explicit_offset"),
                            words="has no UTC offset")
        header = "Date,Open,High,Low,Close,Volume\n"
        for label, ts, words in [("spring", "2025-03-09 02:30:00", "does not exist"),
                                 ("fall", "2025-11-02 01:30:00", "happens twice")]:
            with self.subTest(label):
                self.assertRejected(self.raw(header + f"{ts},1,2,0.5,1,10\n", f"{label}.csv"),
                                    spec(outside_regular_hours="exclude"), words=words)

    def test_malformed_files(self):
        good = render(bars())
        lines = good.splitlines()
        cases = {
            "empty": ("", "The file is empty"),
            "not utf-8": (b"\xff\xfe\x00garbage", "not valid UTF-8"),
            "header only": (lines[0] + "\n", "contains no bars"),
            "missing column": ("Date,Open,High,Low,Close\n1,2,3,4,5\n", "no 'Volume' column"),
            "duplicate header": ("Date,Open,High,Low,Close,Volume,Open\n", "repeats column"),
            "short row": (lines[0] + "\n" + lines[1].rsplit(",", 1)[0] + "\n", "has 5 field(s)"),
            "blank value": (lines[0] + "\n" + lines[1].replace(",100.00,", ",,", 1) + "\n",
                            "is blank"),
            "text value": (lines[0] + "\n" + lines[1].replace("100.00", "abc", 1) + "\n",
                           "is not a number"),
            "nan": (lines[0] + "\n" + lines[1].replace("100.00", "nan", 1) + "\n",
                    "not a plain decimal"),
            "thousands separator": (lines[0] + '\n' + lines[1].replace(",1000", ',"1,000"') + "\n",
                                    "not a plain decimal"),
            "bad timestamp": (lines[0] + "\nyesterday,1,2,0.5,1,10\n", "is not ISO 8601"),
        }
        for label, (text, words) in cases.items():
            with self.subTest(label):
                self.assertRejected(self.raw(text, f"{label}.csv"), words=words)

    def test_unix_timestamps_must_be_whole_numbers(self):
        utc_spec = spec(timestamp_format="unix_nanoseconds", source_timezone="UTC")
        header = "Date,Open,High,Low,Close,Volume\n"
        self.assertRejected(self.raw(header + "1767623400000000001,1,2,0.5,1,10\n"), utc_spec,
                            words="sub-microsecond")
        self.assertRejected(self.raw(header + "1767623400.5,1,2,0.5,1,10\n", "b.csv"),
                            spec(timestamp_format="unix_seconds", source_timezone="UTC"),
                            words="not a whole number")

    def test_unverified_calendar_years_are_refused(self):
        path = self.raw(render(bars()))
        with self.assertRaises(CalendarError):
            self.do_import(path, calendar=TradingCalendar(REAL_DATA))
        with self.assertRaises(CalendarError):
            self.do_import(path, calendar=None)            # the project's real calendar
        self.assertNothingWritten()

    def test_command_line_refuses_unverified_years(self):
        spec_path = os.path.join(self.tmp.name, "spec.json")
        with open(spec_path, "w") as f:
            json.dump(BASE_SPEC, f)
        with mock.patch("builtins.print") as printed:
            code = main([spec_path, self.raw(render(bars())), "--data-dir", self.data_dir])
        self.assertEqual(code, 2)
        self.assertIn("has not been verified", str(printed.call_args_list))
        self.assertNothingWritten()

    def test_protected_folders_are_refused(self):
        data = os.path.join(self.tmp.name, "data")
        logs = os.path.join(self.tmp.name, "logs")
        reports = os.path.join(self.tmp.name, "reports")
        with mock.patch.multiple(settings, DATA_DIR=data, LOG_DIR=logs, REPORTS_DIR=reports,
                                 DATABASE_FILE=os.path.join(data, "paper_account.db"),
                                 JOURNAL_FILE=os.path.join(logs, "trade_journal.csv")):
            for folder in (data, logs, reports, os.path.join(data, "historical")):
                with self.subTest(folder=folder):
                    with self.assertRaises(ImportRejected):
                        self.do_import(self.raw(render(bars())), data_dir=folder)
                    self.assertFalse(os.path.exists(folder))


# --- Versions, replacement and failures --------------------------------------------------------

class VersioningTests(ImportTestCase):
    def snapshot(self, folder):
        out = {}
        for root, _, files in os.walk(folder):
            for name in files:
                with open(os.path.join(root, name), "rb") as f:
                    out[os.path.relpath(os.path.join(root, name), folder)] = f.read()
        return out

    def test_different_data_never_overwrites_without_explicit_new_version(self):
        first = self.do_import(self.raw(render(bars())))
        v1 = self.snapshot(first.folder)
        changed = self.raw(render(bars(zero_volume={bars()[7][0]})), "changed.csv")
        with self.assertRaises(ImportRejected) as caught:
            self.do_import(changed)
        self.assertIn("--new-version", " ".join(caught.exception.problems))
        self.assertEqual(self.snapshot(first.folder), v1)
        second = self.do_import(changed, new_version=True)
        self.assertEqual((second.status, second.version), ("imported", 2))
        self.assertEqual(self.snapshot(first.folder), v1)          # v1 untouched
        self.assertEqual(ManifestCSVProvider("spy_5m_test", self.data_dir)
                         .get_candles("SPY").source.split()[0], "dataset:spy_5m_test/v2")
        self.assertEqual(len(ManifestCSVProvider("spy_5m_test", self.data_dir, version=1)
                             .get_candles("SPY")), 234)

    def test_metadata_change_alone_also_needs_a_new_version(self):
        path = self.raw(render(bars()))
        self.do_import(path)
        with self.assertRaises(ImportRejected):
            self.do_import(path, spec(adjustment="unadjusted"))

    def test_failure_before_commit_leaves_nothing(self):
        path = self.raw(render(bars()))
        for target in ("_store_raw", "_write_new"):
            with self.subTest(failure_in=target):
                with mock.patch.object(importer_module, target, side_effect=OSError("disk full")):
                    with self.assertRaises(OSError):
                        self.do_import(path)
                self.assertNothingWritten()
        with mock.patch.object(importer_module.os, "rename", side_effect=OSError("crash")):
            with self.assertRaises(OSError):
                self.do_import(path)
        dataset_dir = os.path.join(self.data_dir, "datasets", "spy_5m_test")
        self.assertEqual(os.listdir(dataset_dir), [])               # no staging left behind

    def test_failed_new_version_leaves_the_existing_dataset_intact(self):
        first = self.do_import(self.raw(render(bars())))
        v1 = self.snapshot(first.folder)
        changed = self.raw(render(bars(zero_volume={bars()[9][0]})), "changed.csv")
        with mock.patch.object(importer_module.os, "rename", side_effect=OSError("crash")):
            with self.assertRaises(OSError):
                self.do_import(changed, new_version=True)
        dataset_dir = os.path.dirname(first.folder)
        self.assertEqual(sorted(os.listdir(dataset_dir)), ["v1"])
        self.assertEqual(self.snapshot(first.folder), v1)
        self.assertEqual(len(ManifestCSVProvider("spy_5m_test", self.data_dir)
                             .get_candles("SPY")), 234)

    def test_rejected_data_never_touches_an_existing_dataset(self):
        first = self.do_import(self.raw(render(bars())))
        v1 = self.snapshot(first.folder)
        with self.assertRaises(ImportRejected):
            self.do_import(self.raw(render(bars()[1:]), "partial.csv"), new_version=True)
        self.assertEqual(sorted(os.listdir(os.path.dirname(first.folder))), ["v1"])
        self.assertEqual(self.snapshot(first.folder), v1)

    def test_import_does_not_touch_the_paper_account_or_journal(self):
        boom = AssertionError("must not be touched")
        with mock.patch("src.storage.AccountStore.__init__", side_effect=boom), \
             mock.patch("src.paper_trader.PaperTrader.__init__", side_effect=boom), \
             mock.patch("src.journal.log_decision", side_effect=boom):
            self.do_import(self.raw(render(bars())))


# --- Reading datasets back (read-only, verified) -----------------------------------------------

class ManifestProviderTests(ImportTestCase):
    def setUp(self):
        super().setUp()
        self.rows = bars()
        self.result = self.do_import(self.raw(render(self.rows)))
        self.csv_path = os.path.join(self.result.folder, "SPY.csv")
        self.manifest_path = os.path.join(self.result.folder, "manifest.json")

    def test_provider_returns_the_imported_candles_with_a_full_label(self):
        data = ManifestCSVProvider("spy_5m_test", self.data_dir).get_candles("spy")
        self.assertEqual(data.kind, DataKind.HISTORICAL)
        self.assertEqual(len(data), 234)
        self.assertEqual(canonical_csv_bytes(data.candles), expected_canonical(self.rows))
        for words in ("dataset:spy_5m_test/v1", "5-minute bars", "split_adjusted",
                      "consolidated volume", "sha256"):
            self.assertIn(words, data.source)

    def test_edited_csv_is_refused(self):
        with open(self.csv_path, "ab") as f:
            f.write(b"2026-01-07T16:00:00-05:00,1.0,1.0,1.0,1.0,1.0\n")
        with self.assertRaises(DatasetIntegrityError) as caught:
            ManifestCSVProvider("spy_5m_test", self.data_dir).get_candles("SPY")
        self.assertIn("changed or damaged after import", str(caught.exception))

    def test_edited_manifest_is_refused(self):
        with open(self.manifest_path) as f:
            manifest = json.load(f)
        manifest["adjustment"] = "unadjusted"
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f)
        with self.assertRaises(DatasetIntegrityError) as caught:
            load_verified_dataset("spy_5m_test", self.data_dir)
        self.assertIn("content fingerprint does not match", str(caught.exception))

    def test_broken_or_missing_datasets_are_refused(self):
        with self.assertRaises(DatasetIntegrityError):
            load_verified_dataset("nope", self.data_dir)
        with self.assertRaises(DatasetIntegrityError):
            load_verified_dataset("spy_5m_test", self.data_dir, version=7)
        with open(self.manifest_path, "w") as f:
            f.write("{not json")
        with self.assertRaises(DatasetIntegrityError):
            load_verified_dataset("spy_5m_test", self.data_dir)

    def test_wrong_symbol_is_refused(self):
        with self.assertRaises(Exception) as caught:
            ManifestCSVProvider("spy_5m_test", self.data_dir).get_candles("QQQ")
        self.assertIn("holds SPY, not QQQ", str(caught.exception))

    def test_import_refuses_to_build_on_a_damaged_dataset(self):
        with open(self.csv_path, "ab") as f:
            f.write(b"x")
        with self.assertRaises(DatasetIntegrityError):
            self.do_import(self.raw(render(self.rows), "again.csv"), new_version=True)

    def test_leftover_staging_folders_are_ignored(self):
        os.makedirs(os.path.join(os.path.dirname(self.result.folder), ".staging-old"))
        self.assertEqual(load_verified_dataset("spy_5m_test", self.data_dir).version, 1)


if __name__ == "__main__":
    unittest.main()
