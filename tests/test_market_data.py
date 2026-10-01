"""
Tests for the Market Data Engine (src/market_data/).

Run from the project root with:
    python -m unittest discover tests -v

All data comes from tests/market_fixtures.py (built in code) or from CSV
files written to temporary folders. Nothing is downloaded.
"""

import ast
import dataclasses
import math
import os
import pathlib
import tempfile
import unittest
from datetime import date, timedelta, timezone
from unittest import mock

from src import journal
from src.market_data import (MIN_CANDLES_FOR_INDICATORS, CSVHistoricalProvider,
                             DataKind, InMemoryProvider, InsufficientDataError,
                             MarketDataError, MarketDataProvider, MarketDataSet,
                             average_volume, compute_indicators, find_problems,
                             latest_prices, latest_session, rsi, session_vwap,
                             sma, validate_candles, vwap)
from src.paper_trader import PaperTrader
from tests.market_fixtures import (FIVE_MINUTES, MARKET_OPEN, candle,
                                   candles_from_closes, rising_closes,
                                   write_csv, write_raw_csv, zigzag_closes)

NAN, INF = math.nan, math.inf
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def dataset(candles=None, kind=DataKind.HISTORICAL, symbol="SPY", source="test"):
    candles = candles if candles is not None else candles_from_closes(rising_closes(60))
    return MarketDataSet(symbol, kind, source, tuple(candles))


# --- Datasets and labels ------------------------------------------------------

class DatasetTests(unittest.TestCase):
    def test_valid_dataset_is_created_and_labelled(self):
        data = dataset(source="csv:SPY.csv")
        self.assertEqual(len(data), 60)
        self.assertEqual(data.kind, DataKind.HISTORICAL)
        self.assertTrue(data.is_historical)
        self.assertFalse(data.is_live)
        self.assertEqual(data.source, "csv:SPY.csv")
        self.assertEqual(data.latest_close, 159.0)

    def test_symbol_is_tidied_and_list_becomes_tuple(self):
        data = MarketDataSet(" spy ", DataKind.LIVE, "feed", [candle()])
        self.assertEqual(data.symbol, "SPY")
        self.assertIsInstance(data.candles, tuple)
        self.assertTrue(data.is_live)

    def test_dataset_cannot_be_changed_after_creation(self):
        data = dataset()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            data.candles = ()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            data.latest.close = 1.0

    def test_bad_labels_are_rejected(self):
        for kwargs in [dict(kind="historical"), dict(kind=None), dict(source=""),
                       dict(source=None), dict(symbol=""), dict(symbol=None)]:
            with self.subTest(**{k: repr(v) for k, v in kwargs.items()}):
                with self.assertRaises(MarketDataError):
                    dataset(**kwargs)

    def test_empty_dataset_is_rejected(self):
        with self.assertRaises(MarketDataError):
            dataset(candles=[])

    def test_invalid_candles_cannot_make_a_dataset(self):
        with self.assertRaises(MarketDataError):
            dataset(candles=[candle(close=NAN)])

    def test_require_kind(self):
        historical = dataset()
        self.assertIs(historical.require_kind(DataKind.HISTORICAL), historical)
        with self.assertRaises(MarketDataError) as caught:
            historical.require_kind(DataKind.LIVE)
        self.assertIn("is historical, but live data is required", str(caught.exception))

    def test_last_keeps_most_recent_candles_and_labels(self):
        recent = dataset(kind=DataKind.LIVE).last(5)
        self.assertEqual(len(recent), 5)
        self.assertEqual(recent.latest_close, 159.0)
        self.assertEqual(recent.kind, DataKind.LIVE)
        for bad in [0, -1, 2.5, True]:
            with self.subTest(count=bad):
                with self.assertRaises(MarketDataError):
                    dataset().last(bad)


# --- Validation -----------------------------------------------------------------

class ValidationTests(unittest.TestCase):
    def assertProblem(self, candles, words):
        problems = find_problems(candles)
        self.assertTrue(any(words in p for p in problems),
                        f"Expected a problem containing {words!r}, got {problems}")
        with self.assertRaises(MarketDataError):
            validate_candles(candles)

    def test_fixture_candles_are_valid(self):
        self.assertEqual(find_problems(candles_from_closes(zigzag_closes(100))), [])

    def test_missing_values(self):
        for field in ["timestamp", "open", "high", "low", "close", "volume"]:
            with self.subTest(field=field):
                self.assertProblem([candle(**{field: None})], f"{field} is missing")

    def test_nan_and_infinity(self):
        for field in ["open", "high", "low", "close", "volume"]:
            for bad in [NAN, INF, -INF]:
                with self.subTest(field=field, value=bad):
                    self.assertProblem([candle(**{field: bad})],
                                       f"{field} must be a real, finite number")

    def test_negative_and_zero_prices(self):
        for field in ["open", "high", "low", "close"]:
            for bad in [-100.0, 0]:
                with self.subTest(field=field, value=bad):
                    self.assertProblem([candle(**{field: bad})],
                                       f"{field} price must be greater than zero")

    def test_negative_volume_rejected_but_zero_volume_allowed(self):
        self.assertProblem([candle(volume=-1)], "volume cannot be negative")
        self.assertEqual(find_problems([candle(volume=0)]), [])

    def test_wrong_types(self):
        for field, bad in [("close", "100.5"), ("open", True), ("volume", [1]),
                           ("timestamp", "2026-01-05 09:30")]:
            with self.subTest(field=field, value=bad):
                self.assertProblem([candle(**{field: bad})], "must be a")

    def test_invalid_ohlc_relationships(self):
        cases = [
            (dict(high=99.5, low=99.0, open=100.0, close=99.2), "high (99.5) is below open"),
            (dict(high=100.2, open=100.0, close=100.5), "high (100.2) is below close"),
            (dict(low=100.2, open=100.0, close=100.5), "low (100.2) is above open"),
            (dict(low=100.2, open=100.5, close=100.0), "low (100.2) is above close"),
            (dict(high=98.0, low=99.0), "high (98.0) is below low"),
        ]
        for fields, words in cases:
            with self.subTest(**fields):
                self.assertProblem([candle(**fields)], words)

    def test_duplicate_timestamps(self):
        self.assertProblem([candle(), candle()], "duplicate timestamp")

    def test_same_moment_in_another_time_zone_is_a_duplicate(self):
        utc = MARKET_OPEN.astimezone(timezone.utc)          # 14:30 UTC
        self.assertProblem([candle(), candle(timestamp=utc)], "duplicate timestamp")

    def test_out_of_order_timestamps(self):
        later = candle(timestamp=MARKET_OPEN + FIVE_MINUTES)
        self.assertProblem([later, candle()], "out of order")

    def test_timestamp_without_time_zone(self):
        self.assertProblem([candle(timestamp=MARKET_OPEN.replace(tzinfo=None))],
                           "has no time zone")

    def test_not_candles(self):
        self.assertProblem([{"close": 100}], "is not a Candle")
        self.assertEqual(len(find_problems("SPY")), 1)

    def test_every_problem_is_reported_together(self):
        bad = [candle(open=NAN, volume=-5),
               candle(timestamp=MARKET_OPEN + FIVE_MINUTES, high=98.0, low=99.0)]
        with self.assertRaises(MarketDataError) as caught:
            validate_candles(bad)
        problems = caught.exception.problems
        self.assertEqual(len(problems), 3)
        self.assertTrue(problems[0].startswith("Candle 1"))
        self.assertTrue(problems[2].startswith("Candle 2"))

    def test_long_error_messages_are_shortened(self):
        bad = [candle(timestamp=MARKET_OPEN + i * FIVE_MINUTES, close=NAN)
               for i in range(15)]
        with self.assertRaises(MarketDataError) as caught:
            validate_candles(bad)
        self.assertEqual(len(caught.exception.problems), 15)
        self.assertIn("and 5 more problem(s)", str(caught.exception))

    def test_insufficient_candles(self):
        with self.assertRaises(InsufficientDataError) as caught:
            validate_candles(candles_from_closes(rising_closes(10)), min_candles=20)
        self.assertIn("Need at least 20 candle(s) but got 10", str(caught.exception))

    def test_bad_data_is_reported_even_when_also_too_short(self):
        with self.assertRaises(MarketDataError) as caught:
            validate_candles([candle(close=NAN)], min_candles=5)
        self.assertEqual(len(caught.exception.problems), 2)


# --- Indicators -------------------------------------------------------------------

class IndicatorTests(unittest.TestCase):
    def test_sma_known_values(self):
        self.assertEqual(sma(rising_closes(20, start=1.0), 20), 10.5)      # 1..20
        self.assertEqual(sma(rising_closes(50, start=1.0), 50), 25.5)      # 1..50
        self.assertEqual(sma(rising_closes(60, start=1.0), 20), 50.5)      # 41..60

    def test_sma_needs_enough_data(self):
        with self.assertRaises(InsufficientDataError) as caught:
            sma(rising_closes(19), 20)
        self.assertIn("SMA 20 needs at least 20", str(caught.exception))

    def test_bad_indicator_inputs(self):
        for closes in [[100.0] * 19 + [NAN], [100.0] * 19 + [-1.0], "100", None]:
            with self.subTest(closes=repr(closes)[:30]):
                with self.assertRaises(MarketDataError):
                    sma(closes, 20)
        for period in [0, -5, 2.5, True, None]:
            with self.subTest(period=period):
                with self.assertRaises(MarketDataError):
                    sma(rising_closes(30), period)

    def test_rsi_extremes(self):
        self.assertEqual(rsi(rising_closes(15)), 100.0)                    # only gains
        self.assertEqual(rsi(rising_closes(15, start=200, step=-1)), 0.0)  # only losses
        self.assertEqual(rsi([100.0] * 15), 50.0)                          # no movement

    def test_rsi_hand_calculated(self):
        # 14 changes: ten +1 and four -1.
        # average gain = 10/14, average loss = 4/14, RS = 2.5
        # RSI = 100 - 100 / 3.5 = 71.428571...
        changes = [1] * 10 + [-1] * 4
        closes = [100.0]
        for change in changes:
            closes.append(closes[-1] + change)
        self.assertAlmostEqual(rsi(closes), 100 - 100 / 3.5)

        # One more change of +2, using Wilder's smoothing:
        # gain = (10/14 x 13 + 2) / 14,  loss = (4/14 x 13 + 0) / 14
        closes.append(closes[-1] + 2)
        gain = (10 / 14 * 13 + 2) / 14
        loss = (4 / 14 * 13) / 14
        self.assertAlmostEqual(rsi(closes), 100 - 100 / (1 + gain / loss))

    def test_rsi_needs_15_closes(self):
        with self.assertRaises(InsufficientDataError) as caught:
            rsi(rising_closes(14))
        self.assertIn("RSI 14 needs at least 15", str(caught.exception))

    def test_rsi_stays_between_0_and_100(self):
        value = rsi(zigzag_closes(100))
        self.assertGreater(value, 0)
        self.assertLess(value, 100)

    def test_vwap_hand_calculated(self):
        # typical prices 10 (volume 100) and 20 (volume 300):
        # (10 x 100 + 20 x 300) / 400 = 17.5
        bars = [candle(open=10, high=11, low=9, close=10, volume=100),
                candle(timestamp=MARKET_OPEN + FIVE_MINUTES,
                       open=20, high=21, low=19, close=20, volume=300)]
        self.assertAlmostEqual(vwap(bars), 17.5)

    def test_vwap_with_zero_volume_is_refused(self):
        with self.assertRaises(MarketDataError) as caught:
            vwap(candles_from_closes(rising_closes(5), volume=0.0))
        self.assertIn("total volume is zero", str(caught.exception))

    def test_session_vwap_uses_only_the_latest_day(self):
        day1 = candles_from_closes([50.0, 50.0], wick=0)
        day2 = candles_from_closes([10.0, 20.0], start=MARKET_OPEN + timedelta(days=1),
                                   volume=[100.0, 300.0], wick=0)
        # day 2 typical prices: (10,10,10)->10 and (20,20,10)->16.67 (low is 10)
        both = day1 + day2
        self.assertEqual(len(latest_session(both)), 2)
        self.assertEqual(latest_session(both)[0].timestamp.date(), date(2026, 1, 6))
        self.assertAlmostEqual(session_vwap(both), vwap(day2))
        self.assertNotAlmostEqual(session_vwap(both), vwap(both))

    def test_average_volume(self):
        volumes = [float(v) for v in range(1, 31)]                     # 1..30
        bars = candles_from_closes(rising_closes(30), volume=volumes)
        self.assertEqual(average_volume(bars, 20), 20.5)                # 11..30
        with self.assertRaises(InsufficientDataError):
            average_volume(bars[:19], 20)


class SnapshotTests(unittest.TestCase):
    def test_snapshot_matches_individual_indicators(self):
        closes = zigzag_closes(60)
        volumes = [1000.0 + 10 * i for i in range(60)]
        data = dataset(candles_from_closes(closes, volume=volumes), source="fixture")
        snap = compute_indicators(data)
        self.assertEqual(snap.symbol, "SPY")
        self.assertEqual(snap.close, closes[-1])
        self.assertEqual(snap.sma_20, sma(closes, 20))
        self.assertEqual(snap.sma_50, sma(closes, 50))
        self.assertEqual(snap.rsi_14, rsi(closes, 14))
        self.assertEqual(snap.vwap, session_vwap(list(data.candles)))
        self.assertEqual(snap.average_volume_20, sum(volumes[-20:]) / 20)

    def test_snapshot_carries_labels(self):
        snap = compute_indicators(dataset(kind=DataKind.LIVE, source="feed"))
        self.assertEqual(snap.kind, DataKind.LIVE)
        self.assertEqual(snap.source, "feed")
        self.assertEqual(snap.as_of, MARKET_OPEN + 59 * FIVE_MINUTES)
        self.assertIn("live data from feed", snap.describe())

    def test_snapshot_needs_50_candles(self):
        self.assertEqual(MIN_CANDLES_FOR_INDICATORS, 50)
        compute_indicators(dataset(candles_from_closes(rising_closes(50))))
        with self.assertRaises(InsufficientDataError) as caught:
            compute_indicators(dataset(candles_from_closes(rising_closes(49))))
        self.assertIn("need at least 50 candles for SMA 50 but got 49",
                      str(caught.exception))

    def test_snapshot_needs_a_dataset(self):
        with self.assertRaises(MarketDataError):
            compute_indicators(candles_from_closes(rising_closes(60)))


# --- Providers --------------------------------------------------------------------

class InMemoryProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = InMemoryProvider({"SPY": candles_from_closes(rising_closes(60))})

    def test_returns_validated_labelled_data(self):
        data = self.provider.get_candles("spy")
        self.assertIsInstance(data, MarketDataSet)
        self.assertEqual(data.symbol, "SPY")
        self.assertEqual(data.kind, DataKind.HISTORICAL)
        self.assertEqual(data.source, "in-memory")

    def test_limit_and_min_candles(self):
        self.assertEqual(len(self.provider.get_candles("SPY", limit=10)), 10)
        self.assertEqual(len(self.provider.get_candles("SPY", min_candles=60)), 60)
        with self.assertRaises(InsufficientDataError):
            self.provider.get_candles("SPY", min_candles=61)
        with self.assertRaises(InsufficientDataError):
            self.provider.get_candles("SPY", limit=10, min_candles=20)
        for bad in [0, -1, 1.5, True, "10"]:
            with self.subTest(limit=bad):
                with self.assertRaises(MarketDataError):
                    self.provider.get_candles("SPY", limit=bad)

    def test_unknown_symbol_is_an_error_not_empty_data(self):
        with self.assertRaises(MarketDataError) as caught:
            self.provider.get_candles("QQQ")
        self.assertIn("No in-memory data for QQQ", str(caught.exception))

    def test_one_bad_candle_rejects_everything(self):
        good = candles_from_closes(rising_closes(60))
        bad = good[:30] + [dataclasses.replace(good[30], close=NAN)] + good[31:]
        provider = InMemoryProvider({"SPY": bad})
        with self.assertRaises(MarketDataError) as caught:
            provider.get_candles("SPY", limit=10)     # even if not in the last 10
        self.assertIn("Candle 31", str(caught.exception))

    def test_live_kind_is_carried_through(self):
        provider = InMemoryProvider({"SPY": [candle()]}, kind=DataKind.LIVE, name="feed")
        data = provider.get_candles("SPY")
        self.assertTrue(data.is_live)
        self.assertEqual(data.source, "feed")

    def test_bad_symbols_are_refused(self):
        for bad in ["", "   ", None, 5, "../secret", "SP Y", "A" * 11]:
            with self.subTest(symbol=bad):
                with self.assertRaises(MarketDataError):
                    self.provider.get_candles(bad)


class ProviderInterfaceTests(unittest.TestCase):
    def test_interface_cannot_be_used_directly(self):
        with self.assertRaises(TypeError):
            MarketDataProvider()

    def test_new_provider_cannot_skip_validation(self):
        class SloppyFeed(MarketDataProvider):
            name, kind = "sloppy", DataKind.LIVE
            def _load_candles(self, symbol):
                return [candle(high=1.0)]                 # impossible bar
        with self.assertRaises(MarketDataError):
            SloppyFeed().get_candles("SPY")

    def test_provider_must_declare_its_kind(self):
        class Unlabelled(MarketDataProvider):
            name = "mystery"
            def _load_candles(self, symbol):
                return [candle()]
        with self.assertRaises(MarketDataError) as caught:
            Unlabelled().get_candles("SPY")
        self.assertIn("must declare kind", str(caught.exception))

    def test_provider_must_return_a_list(self):
        class Broken(MarketDataProvider):
            name, kind = "broken", DataKind.HISTORICAL
            def _load_candles(self, symbol):
                return None
        with self.assertRaises(MarketDataError):
            Broken().get_candles("SPY")


class CSVProviderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = self.tmp.name
        self.provider = CSVHistoricalProvider(self.folder)
        self.header = "timestamp,open,high,low,close,volume\n"

    def test_reads_candles_exactly(self):
        original = candles_from_closes(zigzag_closes(60))
        path = write_csv(self.folder, "SPY", original)
        data = self.provider.get_candles("spy")
        self.assertEqual(data.candles, tuple(original))
        self.assertEqual(data.kind, DataKind.HISTORICAL)
        self.assertEqual(data.source, f"csv:{path}")
        compute_indicators(data)                          # usable end to end

    def test_missing_file(self):
        with self.assertRaises(MarketDataError) as caught:
            self.provider.get_candles("SPY")
        self.assertIn("No historical data file for SPY", str(caught.exception))

    def test_missing_column(self):
        write_raw_csv(self.folder, "SPY", "timestamp,open,high,low,close\n")
        with self.assertRaises(MarketDataError) as caught:
            self.provider.get_candles("SPY")
        self.assertIn("missing column(s): volume", str(caught.exception))

    def test_blank_cell_is_reported_missing_never_filled_in(self):
        write_raw_csv(self.folder, "SPY", self.header +
                      "2026-01-05T09:30:00-05:00,100,101,99,100.5,1000\n"
                      "2026-01-05T09:35:00-05:00,100.5,,99,100,1000\n")
        with self.assertRaises(MarketDataError) as caught:
            self.provider.get_candles("SPY")
        self.assertIn("Candle 2", str(caught.exception))
        self.assertIn("high is missing", str(caught.exception))

    def test_unreadable_values_name_the_line(self):
        write_raw_csv(self.folder, "SPY", self.header +
                      "2026-01-05T09:30:00-05:00,100,101,99,abc,1000\n"
                      "yesterday,100,101,99,100.5,1000\n")
        with self.assertRaises(MarketDataError) as caught:
            self.provider.get_candles("SPY")
        message = str(caught.exception)
        self.assertIn("line 2: close 'abc' is not a number", message)
        self.assertIn("line 3: timestamp 'yesterday' is not a valid date", message)

    def test_nan_text_and_naive_timestamps_are_rejected(self):
        write_raw_csv(self.folder, "SPY", self.header +
                      "2026-01-05T09:30:00-05:00,100,101,99,nan,1000\n"
                      "2026-01-05T09:35:00,100,101,99,100.5,1000\n")
        with self.assertRaises(MarketDataError) as caught:
            self.provider.get_candles("SPY")
        message = str(caught.exception)
        self.assertIn("close must be a real, finite number", message)
        self.assertIn("has no time zone", message)

    def test_duplicate_rows_are_rejected(self):
        row = "2026-01-05T09:30:00-05:00,100,101,99,100.5,1000\n"
        write_raw_csv(self.folder, "SPY", self.header + row + row)
        with self.assertRaises(MarketDataError) as caught:
            self.provider.get_candles("SPY")
        self.assertIn("duplicate timestamp", str(caught.exception))

    def test_utc_z_suffix_is_accepted(self):
        write_raw_csv(self.folder, "SPY", self.header +
                      "2026-01-05T14:30:00Z,100,101,99,100.5,1000\n")
        self.assertEqual(len(self.provider.get_candles("SPY")), 1)


# --- No fabricated data -----------------------------------------------------------

def module_imports(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


class NoFabricationTests(unittest.TestCase):
    def test_market_data_package_never_uses_random(self):
        for path in (PROJECT_ROOT / "src" / "market_data").glob("*.py"):
            with self.subTest(file=path.name):
                self.assertNotIn("random", {m.split(".")[0] for m in module_imports(path)})

    def test_old_random_price_module_is_gone(self):
        self.assertFalse((PROJECT_ROOT / "src" / "market_data.py").exists())

    def test_main_skips_symbols_without_data_instead_of_inventing_prices(self):
        from src import main
        with tempfile.TemporaryDirectory() as tmp:
            trader = PaperTrader(db_path=os.path.join(tmp, "a.db"),
                                 journal_file=os.path.join(tmp, "j.csv"))
            self.addCleanup(trader.close)
            empty_provider = CSVHistoricalProvider(os.path.join(tmp, "market"))
            with mock.patch.object(journal, "log_decision") as log, \
                 mock.patch("builtins.print"):
                main.run_once(trader, provider=empty_provider)
            actions = [call.args[2] for call in log.call_args_list]
            self.assertEqual(actions, ["SKIPPED", "SKIPPED"])     # SPY and QQQ
            self.assertEqual(trader.cash, 25.00)
            self.assertEqual(trader.positions, {})


# --- Keeping trading independent from market data ---------------------------------

class IndependenceTests(unittest.TestCase):
    def test_trading_modules_do_not_import_market_data(self):
        for name in ["paper_trader.py", "risk_manager.py", "storage.py"]:
            imports = set(module_imports(PROJECT_ROOT / "src" / name))
            with self.subTest(file=name):
                self.assertFalse(any(m.startswith("src.market_data") for m in imports))
                self.assertFalse(any("market_data" in m for m in imports))

    def test_latest_prices_feeds_the_paper_trader(self):
        with tempfile.TemporaryDirectory() as tmp:
            trader = PaperTrader(db_path=os.path.join(tmp, "a.db"),
                                 journal_file=os.path.join(tmp, "j.csv"))
            self.addCleanup(trader.close)
            self.assertTrue(trader.buy("SPY", 0.04, 500.00, 495.00).success)
            falling = InMemoryProvider(
                {"SPY": candles_from_closes([500.0, 497.0, 494.0])}).get_candles("SPY")
            prices = latest_prices([falling])
            self.assertEqual(prices, {"SPY": 494.0})
            results = trader.check_exits(prices)
            self.assertEqual(results[0].action, "STOP-LOSS SELL")

    def test_latest_prices_rejects_bad_input(self):
        with self.assertRaises(MarketDataError):
            latest_prices([dataset(), dataset()])           # SPY twice
        with self.assertRaises(MarketDataError):
            latest_prices([{"SPY": 500.0}])


if __name__ == "__main__":
    unittest.main()
