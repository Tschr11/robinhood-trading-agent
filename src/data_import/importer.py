"""
importer.py - Offline import of one raw CSV file into a versioned, verified,
canonical historical dataset. No network access of any kind.

    python -m src.data_import <spec.json> <raw.csv> [--new-version]

Steps (any failure stops the import and writes nothing that is accepted):
    1. read the raw file ONCE as bytes and fingerprint it (SHA-256); the
       input file itself is only ever opened for reading
    2. parse it strictly according to the spec (reader.py)
    3. check quality strictly against the NYSE calendar (quality.py);
       unverified or uncovered calendar years are refused
    4. build the canonical CSV and manifest deterministically
    5. compare with the latest existing version:
         same content  -> "unchanged", nothing written
         different     -> refused, unless new_version=True (creates vN+1)
       Existing versions are never overwritten or edited.
    6. write into a hidden staging folder, re-verify the bytes, store an
       exact read-only copy of the raw file, then commit with a single
       atomic rename. On any error the staging folder is removed.

Everything in the manifest is deterministic except `imported_at` (and the
version number, which depends on what already exists). The manifest's
`content_fingerprint` covers everything else, so two imports of the same
file with the same spec and calendar have the same fingerprint.
"""

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone

from config import settings
from src.data_import.reader import ImportRejected, parse_raw
from src.data_import.spec import ImportSpec, load_spec
from src.market_data.candles import MarketDataError
from src.market_data.manifest import (CANONICAL_COLUMNS, MANIFEST_FORMAT, MANIFEST_NAME,
                                      canonical_csv_bytes, content_fingerprint,
                                      dataset_versions, load_verified_dataset, sha256_hex)
from src.market_data.quality import ABSENT_BAR_NOTE, check_quality
from src.market_data.sessions import TradingCalendar, load_calendar

IMPORTER_VERSION = "1"


@dataclass(frozen=True)
class ImportResult:
    status: str               # "imported" or "unchanged"
    dataset_id: str
    version: int
    folder: str
    manifest: dict


def import_csv(raw_path: str, spec: ImportSpec, *, calendar: TradingCalendar | None = None,
               data_dir: str = settings.HISTORICAL_DATA_DIR, new_version: bool = False,
               now: datetime | None = None) -> ImportResult:
    """
    Import `raw_path` as described by `spec`. `calendar` defaults to the
    project calendar, which refuses unverified years.
    """
    if not isinstance(spec, ImportSpec):
        raise ImportRejected("spec must be an ImportSpec (see load_spec).")
    _refuse_protected(data_dir)
    calendar = calendar or load_calendar()
    now = now or datetime.now(timezone.utc)

    # 1. Read once; everything below uses these exact bytes.
    name = os.path.basename(raw_path)
    try:
        with open(raw_path, "rb") as f:
            raw_bytes = f.read()
    except OSError as error:
        raise ImportRejected(f"Could not read raw file {raw_path}: {error}") from None
    raw_sha = sha256_hex(raw_bytes)

    # 2-3. Parse and check (CalendarError propagates: unverified years refused).
    parsed = parse_raw(raw_bytes, spec, calendar)
    report = check_quality(parsed.candles, spec.interval_minutes, calendar,
                           empty_bar_policy=spec.empty_bar_policy,
                           start_date=spec.start_date, end_date=spec.end_date)
    if not report.accepted:
        raise ImportRejected(report.rejection_reasons, report)

    # 4. Canonical bytes and manifest.
    canonical = canonical_csv_bytes(parsed.candles)
    years = sorted({c.timestamp.year for c in parsed.candles})
    manifest = {
        "format": MANIFEST_FORMAT,
        "importer_version": IMPORTER_VERSION,
        "dataset_id": spec.dataset_id,
        "symbol": spec.symbol,
        "source": spec.source,
        "interval_minutes": spec.interval_minutes,
        "timezone": "America/New_York",
        "timestamp_convention": "bar_start",
        "session": "regular_trading_hours",
        "source_timestamp": {"format": spec.timestamp_format,
                             "timezone": spec.source_timezone,
                             "convention": spec.timestamp_convention},
        "adjustment": spec.adjustment,
        "volume_coverage": spec.volume_coverage,
        "empty_bar_policy": spec.empty_bar_policy,
        "date_range": {"first_bar": parsed.candles[0].timestamp.isoformat(),
                       "last_bar": parsed.candles[-1].timestamp.isoformat(),
                       "first_session": report.first_session,
                       "last_session": report.last_session,
                       "sessions": report.sessions_expected},
        "source_file": {"name": name, "sha256": raw_sha, "size_bytes": len(raw_bytes),
                        "stored_as": f"raw/{raw_sha}/{name}"},
        "canonical_file": {"name": f"{spec.symbol}.csv", "sha256": sha256_hex(canonical),
                           "rows": len(parsed.candles), "columns": list(CANONICAL_COLUMNS)},
        "calendar": {"version": calendar.version, "years_used": years,
                     "verified_years": [y for y in years if y in calendar.verified_years]},
        "import_spec": spec.to_dict(),
        "import_spec_sha256": spec.fingerprint(),
        "filtering": {"rows_in_source": parsed.rows,
                      "excluded_outside_regular_hours": parsed.excluded_outside_regular_hours,
                      "excluded_outside_date_range": parsed.excluded_outside_date_range,
                      "ignored_columns": list(parsed.ignored_columns)},
        "quality": report.to_dict(),
        "absent_bar_note": ABSENT_BAR_NOTE,
    }
    # Normalise through JSON so the manifest in memory is exactly what is stored
    # (e.g. tuples become lists) and fingerprints are computed on that form.
    manifest = json.loads(json.dumps(manifest, sort_keys=True))
    fingerprint = content_fingerprint(manifest)

    # 5. Compare with what already exists.
    dataset_dir = os.path.join(data_dir, "datasets", spec.dataset_id)
    versions = dataset_versions(dataset_dir)
    if versions:
        latest = load_verified_dataset(spec.dataset_id, data_dir)   # integrity-checked
        if latest.manifest["content_fingerprint"] == fingerprint:
            return ImportResult("unchanged", spec.dataset_id, latest.version,
                                latest.folder, latest.manifest)
        if not new_version:
            raise ImportRejected(
                f"Dataset {spec.dataset_id!r} already exists (latest v{versions[-1]}) with "
                "different content or metadata. Nothing was changed. To keep both, "
                "import again with --new-version; existing versions are never overwritten.")
    version = (versions[-1] + 1) if versions else 1
    manifest = dict(manifest, version=version, imported_at=now.isoformat(),
                    content_fingerprint=fingerprint)

    # 6. Stage, verify, store raw copy, commit atomically.
    os.makedirs(dataset_dir, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".staging-", dir=dataset_dir)
    try:
        _write_new(os.path.join(staging, f"{spec.symbol}.csv"), canonical)
        _write_new(os.path.join(staging, MANIFEST_NAME),
                   (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        with open(os.path.join(staging, f"{spec.symbol}.csv"), "rb") as f:
            if sha256_hex(f.read()) != manifest["canonical_file"]["sha256"]:
                raise ImportRejected("The staged canonical file did not verify.")
        _store_raw(data_dir, raw_bytes, raw_sha, name)
        target = os.path.join(dataset_dir, f"v{version}")
        if os.path.exists(target):
            raise ImportRejected(f"{target} appeared during the import; nothing was changed.")
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ImportResult("imported", spec.dataset_id, version, target, manifest)


def _write_new(path: str, data: bytes) -> None:
    with open(path, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _store_raw(data_dir: str, raw_bytes: bytes, raw_sha: str, name: str) -> None:
    """Keep an exact, read-only copy of the raw file, addressed by its SHA-256."""
    folder = os.path.join(data_dir, "raw", raw_sha)
    path = os.path.join(folder, name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            if sha256_hex(f.read()) != raw_sha:
                raise ImportRejected(f"The stored raw copy {path} no longer matches its "
                                     "fingerprint; it was changed or damaged.")
        return
    os.makedirs(folder, exist_ok=True)
    temp = tempfile.mkstemp(prefix=".partial-", dir=folder)
    os.close(temp[0])
    try:
        with open(temp[1], "wb") as f:
            f.write(raw_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(temp[1], 0o444)
        os.replace(temp[1], path)
    except BaseException:
        if os.path.exists(temp[1]):
            os.chmod(temp[1], 0o644)
            os.remove(temp[1])
        raise


def _refuse_protected(data_dir: str) -> None:
    """Historical data must stay apart from the paper account, journal and reports."""
    folder = os.path.abspath(data_dir)
    for place in {settings.DATA_DIR, settings.LOG_DIR, settings.REPORTS_DIR,
                  os.path.dirname(settings.DATABASE_FILE) or ".",
                  os.path.dirname(settings.JOURNAL_FILE) or "."}:
        place = os.path.abspath(place)
        if folder == place or folder.startswith(place + os.sep):
            raise ImportRejected(
                f"Historical data can't be stored in {data_dir}: that folder holds the "
                "paper account, the trading journal or evaluation reports.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Import a raw historical CSV file (offline) into a verified dataset.")
    parser.add_argument("spec", help="path to the JSON import spec")
    parser.add_argument("raw", help="path to the raw CSV file (never modified)")
    parser.add_argument("--new-version", action="store_true",
                        help="create a new version if the dataset already exists")
    parser.add_argument("--data-dir", default=settings.HISTORICAL_DATA_DIR)
    args = parser.parse_args(argv)
    try:
        result = import_csv(args.raw, load_spec(args.spec), data_dir=args.data_dir,
                            new_version=args.new_version)
    except ImportRejected as error:
        print("IMPORT REJECTED - nothing was written.")
        if error.report is not None:
            print(error.report.summary())
        else:
            for problem in error.problems:
                print(f"  - {problem}")
        return 1
    except MarketDataError as error:
        print(f"IMPORT REFUSED - nothing was written.\n  - {error}")
        return 2
    m = result.manifest
    print(f"{result.status.upper()}: {result.dataset_id} v{result.version} at {result.folder}")
    print(f"  {m['canonical_file']['rows']} bars, {m['date_range']['first_session']} to "
          f"{m['date_range']['last_session']}, {m['interval_minutes']}-minute, "
          f"{m['adjustment']}, {m['volume_coverage']} volume")
    print(f"  raw sha256 {m['source_file']['sha256']}")
    print(f"  canonical sha256 {m['canonical_file']['sha256']}")
    return 0
