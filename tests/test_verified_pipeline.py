"""
End-to-end tests for Stage 3: raw file -> importer -> verified canonical
dataset -> evaluation -> backtest -> report.

Every raw file is generated locally in a temporary folder, and a verified
COPY of the calendar is passed to the importer (the real calendar stays
unverified). These tests check plumbing and provenance only; they make no
claim about how well the strategy performs.
"""

import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import date, datetime, timezone
from unittest import mock

from config import settings
from src import evaluation
from src.backtest import BacktestError, run_backtest
from src.data_import.importer import import_csv
from src.data_import.spec import spec_from_dict
from src.evaluation import (UNVERIFIED_CSV_STAMP, DatasetSpec, EvaluationError,
                            EvaluationPlan, evaluate, main, plan_from_dict, save_report)
from src.market_data import DataKind, MarketDataSet
from src.market_data.manifest import (DatasetIntegrityError,
                                      content_fingerprint, load_verified_market_data)
from src.market_data.sessions import TradingCalendar
from tests.market_fixtures import write_csv
from tests.test_evaluation import four_days
from tests.test_sessions import verified_data

CAL = TradingCalendar(verified_data())
FIXED_NOW = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
IMPORTED_AT = datetime(2026, 1, 31, 12, 0, tzinfo=timezone.utc)
SPLIT = "2026-01-07"
DATASET_ID = "spy_5m_test"

SPEC = {
    "dataset_id": DATASET_ID, "symbol": "SPY", "source": "locally generated test file",
    "interval_minutes": 5,
    "columns": {"timestamp": "timestamp", "open": "open", "high": "high", "low": "low",
                "close": "close", "volume": "volume"},
    "timestamp_format": "iso8601", "source_timezone": "explicit_offset",
    "timestamp_convention": "bar_start", "adjustment": "split_adjusted",
    "volume_coverage": "consolidated", "empty_bar_policy": "omitted",
    "outside_regular_hours": "reject",
}


def _snapshot(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


class PipelineTestCase(unittest.TestCase):
    """Imports four days of locally generated SPY bars as dataset v1."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.join(self.tmp.name, "historical_data")
        self.reports = os.path.join(self.tmp.name, "reports")
        self.log = os.path.join(self.tmp.name, "exposure", "oos_exposure_log.jsonl")
        self.candles = four_days()
        self.raw = write_csv(os.path.join(self.tmp.name, "raw"), "SPY", self.candles)
        self.imported = self.do_import(self.raw)
        self.loaded = load_verified_market_data(DATASET_ID, 1, self.root)
        self.identity = self.loaded.identity
        # Safety net: no test may create or change the REAL exposure log.
        real_log = _snapshot(settings.OOS_EXPOSURE_LOG)
        self.addCleanup(lambda: self.assertEqual(
            _snapshot(settings.OOS_EXPOSURE_LOG), real_log,
            "a test touched the real exposure log"))

    def do_import(self, path, **kwargs):
        return import_csv(path, spec_from_dict(dict(SPEC)), calendar=CAL,
                          data_dir=self.root, now=IMPORTED_AT, **kwargs)

    @property
    def version_folder(self):
        return os.path.join(self.root, "datasets", DATASET_ID, "v1")

    def plan(self, allow_unverified_csv=False, **changes):
        dataset = {"dataset_id": DATASET_ID, "version": 1, "split_date": SPLIT}
        dataset.update(changes)
        return plan_from_dict({"name": "verified_plan", "datasets": [dataset],
                               "allow_unverified_csv": allow_unverified_csv})

    def run_eval(self, plan=None, **kwargs):
        kwargs.setdefault("exposure_log", self.log)
        kwargs.setdefault("now", FIXED_NOW)
        kwargs.setdefault("historical_data_dir", self.root)
        return evaluate(plan or self.plan(), **kwargs)

    def forge_manifest(self, change):
        """Edit the manifest and recompute its fingerprint, as a careful forger would."""
        path = os.path.join(self.version_folder, "manifest.json")
        with open(path) as f:
            manifest = json.load(f)
        change(manifest)
        manifest["content_fingerprint"] = content_fingerprint(manifest)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)


# --- The verified route end to end ----------------------------------------------------------

class VerifiedRouteTests(PipelineTestCase):
    def test_imported_candles_are_exactly_the_raw_candles(self):
        self.assertEqual(self.imported.status, "imported")
        self.assertEqual(list(self.loaded.dataset.candles), self.candles)
        self.assertEqual(self.loaded.dataset.symbol, "SPY")
        self.assertIn(f"dataset {DATASET_ID} v1", self.loaded.dataset.source)

    def test_identity_matches_the_manifest(self):
        m = self.imported.manifest
        self.assertEqual(self.identity.dataset_id, DATASET_ID)
        self.assertEqual(self.identity.version, 1)
        self.assertEqual(self.identity.symbol, "SPY")
        self.assertEqual(self.identity.canonical_sha256, m["canonical_file"]["sha256"])
        self.assertEqual(self.identity.raw_sha256, m["source_file"]["sha256"])
        self.assertEqual(self.identity.content_fingerprint, m["content_fingerprint"])
        self.assertEqual(self.identity.calendar_version, CAL.version)
        self.assertEqual(self.identity.imported_at, IMPORTED_AT.isoformat())

    def test_report_records_the_dataset_identity(self):
        report = self.run_eval()
        result = report.datasets[0]
        self.assertEqual(result.status, "ok", result.error)
        self.assertEqual(result.identity, self.identity)
        self.assertEqual(result.sha256, self.identity.canonical_sha256)
        self.assertEqual(result.symbol, "SPY")
        self.assertTrue(all(p.status == "ok" for p in result.periods))

        record = report.to_dict()
        self.assertEqual(record["data_verification"], "verified dataset")
        self.assertIsNone(record["stamp"])
        entry = record["datasets"][0]
        self.assertEqual(entry["verification"], "verified dataset")
        self.assertEqual(entry["dataset_identity"], self.identity.to_dict())
        self.assertEqual((entry["dataset_id"], entry["version"]), (DATASET_ID, 1))
        self.assertEqual(entry["symbol"], "SPY")

        text = report.to_text()
        self.assertIn(f"verified dataset {DATASET_ID} v1", text)
        self.assertIn(self.identity.raw_sha256, text)
        self.assertIn(self.identity.canonical_sha256, text)
        self.assertNotIn(UNVERIFIED_CSV_STAMP, text)
        self.assertTrue(text.startswith("EVALUATION - historical backtests"))

    def test_every_backtest_result_retains_the_identity(self):
        results, real = [], run_backtest

        def recording(dataset, *args, **kwargs):
            result = real(dataset, *args, **kwargs)
            results.append((dataset, result))
            return result

        with mock.patch.object(evaluation, "run_backtest", side_effect=recording):
            report = self.run_eval()
        self.assertEqual(report.datasets[0].status, "ok")
        self.assertEqual(len(results), 2)                 # in-sample and out-of-sample
        for dataset, result in results:
            self.assertEqual(result.dataset_identity, self.identity)
            self.assertIn(f"verified dataset {DATASET_ID} v1", result.report())
            # The backtest only ever saw candles from the verified dataset.
            self.assertTrue(set(dataset.candles) <= set(self.loaded.dataset.candles))

    def test_symbol_in_plan_is_optional_but_must_match(self):
        self.assertEqual(self.run_eval(self.plan(symbol="spy")).datasets[0].status, "ok")
        with mock.patch.object(evaluation, "run_backtest") as backtest:
            report = self.run_eval(self.plan(symbol="QQQ"))
        backtest.assert_not_called()
        self.assertEqual(report.datasets[0].status, "error")
        self.assertIn("holds SPY", report.datasets[0].error)

    def test_same_metrics_as_the_csv_route_on_the_same_candles(self):
        # The canonical CSV is also a valid plain CSV; evaluate it both ways.
        market = os.path.join(self.tmp.name, "market")
        os.makedirs(market)
        with open(os.path.join(self.version_folder, "SPY.csv"), "rb") as src, \
                open(os.path.join(market, "SPY.csv"), "wb") as dst:
            dst.write(src.read())
        verified = self.run_eval()
        unverified = self.run_eval(plan_from_dict({
            "name": "csv_plan", "allow_unverified_csv": True,
            "datasets": [{"symbol": "SPY", "folder": market, "split_date": SPLIT}]}))
        a, b = verified.datasets[0], unverified.datasets[0]
        self.assertEqual(a.sha256, b.sha256)
        self.assertEqual(a.oos_key, b.oos_key)
        for pa, pb in zip(a.periods, b.periods):
            self.assertEqual((pa.status, pa.candles, pa.history_candles),
                             (pb.status, pb.candles, pb.history_candles))
            self.assertEqual(pa.strategy, pb.strategy)
            self.assertEqual(pa.benchmark, pb.benchmark)
        self.assertIsNone(b.identity)


# --- Provenance can never be dropped -----------------------------------------------------

class ProvenanceTests(PipelineTestCase):
    def assertProvenanceRefused(self, wrapper):
        with mock.patch.object(evaluation, "run_backtest", side_effect=wrapper):
            with self.assertRaises(EvaluationError) as caught:
                self.run_eval()
        self.assertIn("provenance", str(caught.exception))

    def test_identity_dropped_from_the_result_is_refused(self):
        def dropping(*args, **kwargs):
            return replace(run_backtest(*args, **kwargs), dataset_identity=None)
        self.assertProvenanceRefused(dropping)

    def test_identity_not_passed_to_the_backtester_is_refused(self):
        def forgetting(*args, **kwargs):
            kwargs.pop("dataset_identity", None)
            return run_backtest(*args, **kwargs)
        self.assertProvenanceRefused(forgetting)

    def test_identity_swapped_for_another_is_refused(self):
        def swapping(*args, **kwargs):
            result = run_backtest(*args, **kwargs)
            return replace(result, dataset_identity=replace(result.dataset_identity,
                                                            version=2))
        self.assertProvenanceRefused(swapping)

    def test_evaluation_passes_the_identity_to_run_backtest(self):
        seen = []

        def recording(*args, **kwargs):
            seen.append(kwargs.get("dataset_identity"))
            return run_backtest(*args, **kwargs)

        with mock.patch.object(evaluation, "run_backtest", side_effect=recording):
            self.run_eval()
        self.assertEqual(seen, [self.identity, self.identity])

    def test_csv_route_results_carry_no_identity(self):
        market = os.path.join(self.tmp.name, "market")
        write_csv(market, "SPY", self.candles)
        seen = []

        def recording(*args, **kwargs):
            result = run_backtest(*args, **kwargs)
            seen.append(result.dataset_identity)
            return result

        plan = plan_from_dict({"name": "csv_plan", "allow_unverified_csv": True,
                               "datasets": [{"symbol": "SPY", "folder": market,
                                             "split_date": SPLIT}]})
        with mock.patch.object(evaluation, "run_backtest", side_effect=recording):
            report = self.run_eval(plan)
        self.assertEqual(seen, [None, None])
        self.assertIsNone(report.datasets[0].identity)

    def test_run_backtest_keeps_and_checks_the_identity(self):
        dataset = self.loaded.dataset
        result = run_backtest(dataset, dataset_identity=self.identity)
        self.assertEqual(result.dataset_identity, self.identity)
        self.assertIn(f"Data: verified dataset {DATASET_ID} v1", result.report())
        self.assertIn(self.identity.raw_sha256, result.report())

        plain = run_backtest(dataset)
        self.assertIsNone(plain.dataset_identity)
        self.assertIn("no dataset provenance recorded", plain.report())

        with self.assertRaises(BacktestError):
            run_backtest(dataset, dataset_identity=self.identity.to_dict())
        with self.assertRaises(BacktestError):
            run_backtest(dataset, dataset_identity=replace(self.identity, symbol="QQQ"))
        with self.assertRaises(BacktestError):
            run_backtest(MarketDataSet("QQQ", DataKind.HISTORICAL, "fixture", tuple(self.candles)),
                         dataset_identity=self.identity)


# --- Only data that still verifies reaches the backtester -----------------------------------

class IntegrityTests(PipelineTestCase):
    def assertRefused(self, *needles, plan=None):
        with mock.patch.object(evaluation, "run_backtest") as backtest:
            report = self.run_eval(plan)
        backtest.assert_not_called()
        result = report.datasets[0]
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.identity)
        self.assertIsNone(result.oos_key)
        self.assertEqual(result.periods, ())
        for needle in needles:
            self.assertIn(needle, result.error)
        return result

    def test_edited_canonical_csv_is_refused(self):
        path = os.path.join(self.version_folder, "SPY.csv")
        with open(path, "rb") as f:
            content = f.read()
        lines = content.split(b"\n")
        fields = lines[5].split(b",")
        fields[4] = repr(float(fields[4]) + 0.01).encode()       # one close changed
        lines[5] = b",".join(fields)
        with open(path, "wb") as f:
            f.write(b"\n".join(lines))
        self.assertRefused("SHA-256")

    def test_edited_manifest_is_refused(self):
        path = os.path.join(self.version_folder, "manifest.json")
        with open(path) as f:
            manifest = json.load(f)
        manifest["adjustment"] = "unadjusted"
        with open(path, "w") as f:
            json.dump(manifest, f)
        self.assertRefused("fingerprint")

    def test_missing_version_is_refused_not_replaced_by_another(self):
        self.assertRefused("no version 2", plan=self.plan(version=2))

    def test_missing_dataset_is_refused(self):
        self.assertRefused("No dataset", plan=self.plan(dataset_id="qqq_5m_test"))

    def test_manifest_without_an_accepted_quality_check_is_refused(self):
        self.forge_manifest(lambda m: m["quality"].update(accepted=False))
        self.assertRefused("accepted quality check")

    def test_manifest_with_unverified_calendar_years_is_refused(self):
        self.forge_manifest(lambda m: m["calendar"].update(verified_years=[]))
        self.assertRefused("calendar year")

    def test_deleted_canonical_csv_is_refused(self):
        os.remove(os.path.join(self.version_folder, "SPY.csv"))
        self.assertRefused("unreadable")

    def test_loader_refuses_versions_that_are_not_explicit(self):
        for version in [None, 0, -1, True, 1.0, "1", "latest"]:
            with self.subTest(version=version):
                with self.assertRaises(DatasetIntegrityError):
                    load_verified_market_data(DATASET_ID, version, self.root)


# --- Plans: the two routes, and Option A for plain CSV --------------------------------------

class PlanTests(PipelineTestCase):
    def test_verified_plan_needs_no_flag(self):
        plan = self.plan()
        spec = plan.datasets[0]
        self.assertTrue(spec.verified)
        self.assertEqual((spec.dataset_id, spec.version, spec.folder, spec.symbol),
                         (DATASET_ID, 1, None, None))
        self.assertFalse(plan.uses_unverified_csv)
        self.assertEqual(plan.to_dict()["datasets"][0]["dataset_id"], DATASET_ID)
        self.assertEqual(plan.to_dict()["datasets"][0]["version"], 1)
        self.assertFalse(plan.to_dict()["allow_unverified_csv"])

    def test_csv_dataset_without_the_flag_is_rejected(self):
        raw = {"name": "p", "datasets": [{"symbol": "SPY", "folder": "x",
                                          "split_date": SPLIT}]}
        for flag in [None, False]:
            with self.subTest(flag=flag):
                if flag is not None:
                    raw["allow_unverified_csv"] = flag
                with self.assertRaises(EvaluationError) as caught:
                    plan_from_dict(raw)
                self.assertIn("allow_unverified_csv", str(caught.exception))
        with self.assertRaises(EvaluationError):
            EvaluationPlan("p", (DatasetSpec("SPY", "x", date(2026, 1, 7)),))
        with self.assertRaises(EvaluationError):            # one CSV dataset is enough
            plan_from_dict({"name": "p", "datasets": [
                {"dataset_id": DATASET_ID, "version": 1, "split_date": SPLIT},
                {"symbol": "SPY", "split_date": SPLIT}]})

    def test_flag_must_be_a_real_boolean(self):
        for flag in ["true", 1, None]:
            with self.subTest(flag=flag):
                with self.assertRaises(EvaluationError):
                    plan_from_dict({"name": "p", "allow_unverified_csv": flag,
                                    "datasets": [{"symbol": "SPY", "split_date": SPLIT}]})
        with self.assertRaises(EvaluationError):
            EvaluationPlan("p", (DatasetSpec("SPY", "x", date(2026, 1, 7)),),
                           allow_unverified_csv=1)

    def test_invalid_dataset_routes_are_rejected(self):
        bad = [
            {"dataset_id": DATASET_ID, "split_date": SPLIT},                  # no version
            {"dataset_id": DATASET_ID, "version": "latest", "split_date": SPLIT},
            {"dataset_id": DATASET_ID, "version": 0, "split_date": SPLIT},
            {"dataset_id": DATASET_ID, "version": True, "split_date": SPLIT},
            {"dataset_id": DATASET_ID, "version": 1.0, "split_date": SPLIT},
            {"dataset_id": DATASET_ID, "version": 1, "folder": "x", "split_date": SPLIT},
            {"dataset_id": "../escape", "version": 1, "split_date": SPLIT},
            {"dataset_id": "SPY_5m", "version": 1, "split_date": SPLIT},
            {"dataset_id": "", "version": 1, "split_date": SPLIT},
            {"symbol": "SPY", "version": 1, "split_date": SPLIT},             # no dataset_id
            {"dataset_id": DATASET_ID, "version": 1},                         # no split
            {"split_date": SPLIT},                                            # no source
        ]
        for dataset in bad:
            with self.subTest(dataset=dataset):
                with self.assertRaises(EvaluationError):
                    plan_from_dict({"name": "p", "allow_unverified_csv": True,
                                    "datasets": [dataset]})

    def test_fingerprint_changes_with_dataset_version_and_flag(self):
        base = self.plan().fingerprint()
        self.assertNotEqual(base, self.plan(version=2).fingerprint())
        self.assertNotEqual(base, self.plan(allow_unverified_csv=True).fingerprint())

    def test_csv_route_reports_are_stamped(self):
        market = os.path.join(self.tmp.name, "market")
        write_csv(market, "SPY", self.candles)
        plan = plan_from_dict({"name": "mixed", "allow_unverified_csv": True, "datasets": [
            {"dataset_id": DATASET_ID, "version": 1, "split_date": SPLIT},
            {"symbol": "SPY", "folder": market, "split_date": SPLIT}]})
        report = self.run_eval(plan)
        self.assertEqual([d.status for d in report.datasets], ["ok", "ok"])
        json_path, text_path = save_report(report, self.reports, self.log)
        with open(json_path) as f:
            record = json.load(f)
        with open(text_path) as f:
            text = f.read()
        self.assertEqual(record["stamp"], UNVERIFIED_CSV_STAMP)
        self.assertEqual(record["data_verification"], "unverified CSV")
        self.assertEqual([d["verification"] for d in record["datasets"]],
                         ["verified dataset", "unverified CSV"])
        self.assertEqual(record["datasets"][0]["dataset_identity"], self.identity.to_dict())
        self.assertIsNone(record["datasets"][1]["dataset_identity"])
        lines = text.rstrip("\n").split("\n")
        self.assertEqual(lines[0], UNVERIFIED_CSV_STAMP)
        self.assertEqual(lines[-1], UNVERIFIED_CSV_STAMP)
        csv_header = [line for line in lines if line.startswith("== SPY  (") and
                      market in line]
        self.assertEqual(len(csv_header), 1)
        self.assertTrue(csv_header[0].endswith(UNVERIFIED_CSV_STAMP))

    def test_flag_alone_does_not_stamp_a_fully_verified_report(self):
        report = self.run_eval(self.plan(allow_unverified_csv=True))
        self.assertIsNone(report.to_dict()["stamp"])
        self.assertNotIn(UNVERIFIED_CSV_STAMP, report.to_text())


# --- Out-of-sample exposure and the command line ----------------------------------------

class ExposureAndCommandLineTests(PipelineTestCase):
    def test_oos_key_uses_the_canonical_sha256(self):
        result = self.run_eval().datasets[0]
        self.assertEqual(result.oos_key, {"symbol": "SPY",
                                          "sha256": self.identity.canonical_sha256,
                                          "split_date": SPLIT, "end_date": "2026-01-08"})

    def test_same_canonical_data_in_a_new_version_shares_the_exposure_count(self):
        save_report(self.run_eval(), self.reports, self.log)
        # Re-import the same candles from a differently named raw file as v2.
        other = write_csv(os.path.join(self.tmp.name, "raw2"), "SPY_copy", self.candles)
        second = self.do_import(other, new_version=True)
        self.assertEqual(second.version, 2)
        self.assertEqual(second.manifest["canonical_file"]["sha256"],
                         self.identity.canonical_sha256)
        report = self.run_eval(self.plan(version=2))
        self.assertEqual(report.datasets[0].identity.version, 2)
        self.assertEqual(report.datasets[0].prior_oos_evaluations, 1)

    def test_command_line_runs_a_verified_plan(self):
        plan_path = os.path.join(self.tmp.name, "plan.json")
        with open(plan_path, "w") as f:
            json.dump({"name": "cli", "datasets": [
                {"dataset_id": DATASET_ID, "version": 1, "split_date": SPLIT}]}, f)
        with mock.patch("builtins.print"):
            json_path, text_path = main([plan_path, "--reports-dir", self.reports,
                                         "--exposure-log", self.log,
                                         "--historical-data-dir", self.root])
        with open(json_path) as f:
            record = json.load(f)
        self.assertEqual(record["datasets"][0]["dataset_identity"], self.identity.to_dict())
        self.assertIsNone(record["stamp"])

    def test_evaluation_never_writes_into_the_dataset(self):
        def listing():
            found = {}
            for folder, _, names in os.walk(self.root):
                for name in names:
                    path = os.path.join(folder, name)
                    found[path] = _snapshot(path)
            return found

        before = listing()
        save_report(self.run_eval(), self.reports, self.log)
        self.assertEqual(listing(), before)


if __name__ == "__main__":
    unittest.main()
