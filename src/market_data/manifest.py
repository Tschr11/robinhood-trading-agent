"""
manifest.py - The canonical historical dataset format, and reading it back
with integrity checks.

A canonical dataset version is a folder:

    historical_data/datasets/<dataset_id>/v<N>/<SYMBOL>.csv
    historical_data/datasets/<dataset_id>/v<N>/manifest.json

<SYMBOL>.csv has exactly these columns, oldest bar first:
    timestamp,open,high,low,close,volume
where timestamp is the bar START in New York time with New York's UTC offset
(e.g. 2026-01-05T09:30:00-05:00) and numbers are written in Python's exact
shortest form, so the same candles always produce the same bytes.

manifest.json records where the data came from (source, raw file SHA-256),
how it was interpreted (interval, timestamp convention, time zone,
adjustment basis, volume coverage, how the source treats intervals with no
trades), the date range, the calendar version, the quality findings, and the
SHA-256 of the canonical CSV.

ManifestCSVProvider reads a version READ-ONLY. It checks the manifest and the
CSV's SHA-256 against the exact bytes it then parses, so a dataset that was
edited or damaged after import is refused instead of used.
"""

import csv
import hashlib
import io
import json
import os

from config import settings
from src.market_data.candles import Candle, DataKind, MarketDataError
from src.market_data.providers import MarketDataProvider, _parse_cell, clean_symbol

MANIFEST_FORMAT = "historical-dataset/1"
MANIFEST_NAME = "manifest.json"
CANONICAL_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
REQUIRED_MANIFEST_KEYS = {
    "format", "importer_version", "dataset_id", "version", "symbol", "source",
    "interval_minutes", "timezone", "timestamp_convention", "source_timestamp",
    "adjustment", "volume_coverage", "empty_bar_policy", "session", "date_range",
    "source_file", "canonical_file", "calendar", "import_spec", "import_spec_sha256",
    "filtering", "quality", "absent_bar_note", "content_fingerprint", "imported_at"}
# Fields that are NOT part of the content fingerprint (documented as varying).
VOLATILE_MANIFEST_KEYS = ("version", "imported_at", "content_fingerprint")


class DatasetIntegrityError(MarketDataError):
    """A stored dataset is missing, malformed, or does not match its manifest."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_csv_bytes(candles) -> bytes:
    """The canonical CSV for `candles`, byte-for-byte deterministic."""
    lines = [",".join(CANONICAL_COLUMNS)]
    for c in candles:
        lines.append(",".join([c.timestamp.isoformat(), repr(float(c.open)),
                               repr(float(c.high)), repr(float(c.low)),
                               repr(float(c.close)), repr(float(c.volume))]))
    return ("\n".join(lines) + "\n").encode("utf-8")


def content_fingerprint(manifest: dict) -> str:
    """SHA-256 of the manifest without its volatile fields (sorted JSON)."""
    stable = {k: v for k, v in manifest.items() if k not in VOLATILE_MANIFEST_KEYS}
    return sha256_hex(json.dumps(stable, sort_keys=True).encode("utf-8"))


def dataset_versions(dataset_dir: str) -> list[int]:
    """Committed versions (folders named v1, v2, ...), ignoring staging folders."""
    if not os.path.isdir(dataset_dir):
        return []
    versions = []
    for name in os.listdir(dataset_dir):
        if name.startswith("v") and name[1:].isdigit() and str(int(name[1:])) == name[1:]:
            versions.append(int(name[1:]))
    return sorted(versions)


class VerifiedDataset:
    """A dataset version whose manifest and CSV bytes have been checked."""

    def __init__(self, manifest: dict, csv_bytes: bytes, folder: str):
        self.manifest = manifest
        self.csv_bytes = csv_bytes
        self.folder = folder

    @property
    def version(self) -> int:
        return self.manifest["version"]

    def candles(self) -> list[Candle]:
        """Parse the verified bytes (never re-read from disk)."""
        reader = csv.DictReader(io.StringIO(self.csv_bytes.decode("utf-8"), newline=""))
        if tuple(reader.fieldnames or ()) != CANONICAL_COLUMNS:
            raise DatasetIntegrityError(f"{self.folder}: canonical CSV has the wrong columns.")
        candles, problems = [], []
        for row in reader:
            values = {}
            for column in CANONICAL_COLUMNS:
                values[column], problem = _parse_cell(column, row.get(column))
                if problem:
                    problems.append(f"line {reader.line_num}: {problem}")
            candles.append(Candle(**values))
        if problems:
            raise DatasetIntegrityError(problems)
        return candles


def load_verified_dataset(dataset_id: str, root: str = settings.HISTORICAL_DATA_DIR,
                          version: int | None = None) -> VerifiedDataset:
    """
    Load one version (default: the latest) and verify it. Raises
    DatasetIntegrityError if anything does not match.
    """
    dataset_dir = os.path.join(root, "datasets", dataset_id)
    versions = dataset_versions(dataset_dir)
    if not versions:
        raise DatasetIntegrityError(f"No dataset {dataset_id!r} in {root}.")
    if version is None:
        version = versions[-1]
    if version not in versions:
        raise DatasetIntegrityError(f"Dataset {dataset_id!r} has no version {version} "
                                    f"(has {versions}).")
    folder = os.path.join(dataset_dir, f"v{version}")
    try:
        with open(os.path.join(folder, MANIFEST_NAME), "rb") as f:
            manifest = json.loads(f.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetIntegrityError(f"{folder}: unreadable manifest ({error}).") from None

    problems = []
    if not isinstance(manifest, dict):
        raise DatasetIntegrityError(f"{folder}: manifest must be a JSON object.")
    missing = REQUIRED_MANIFEST_KEYS - set(manifest)
    if missing:
        raise DatasetIntegrityError(f"{folder}: manifest is missing {sorted(missing)}.")
    if manifest["format"] != MANIFEST_FORMAT:
        problems.append(f"unsupported manifest format {manifest['format']!r}")
    if manifest["dataset_id"] != dataset_id:
        problems.append(f"manifest dataset_id {manifest['dataset_id']!r} does not match")
    if manifest["version"] != version:
        problems.append(f"manifest version {manifest['version']!r} does not match folder")
    if manifest["content_fingerprint"] != content_fingerprint(manifest):
        problems.append("manifest content fingerprint does not match its contents")
    file_info = manifest["canonical_file"]
    expected_name = f"{manifest['symbol']}.csv"
    if not isinstance(file_info, dict) or file_info.get("name") != expected_name:
        problems.append("canonical file entry is malformed")
    if problems:
        raise DatasetIntegrityError(f"{folder}: " + "; ".join(problems) + ".")

    try:
        with open(os.path.join(folder, expected_name), "rb") as f:
            csv_bytes = f.read()
    except OSError as error:
        raise DatasetIntegrityError(f"{folder}: canonical CSV unreadable ({error}).") from None
    if sha256_hex(csv_bytes) != file_info.get("sha256"):
        raise DatasetIntegrityError(
            f"{folder}: {expected_name} does not match the SHA-256 recorded in its "
            "manifest; it was changed or damaged after import.")
    rows = csv_bytes.count(b"\n") - 1
    if rows != file_info.get("rows"):
        raise DatasetIntegrityError(f"{folder}: row count does not match the manifest.")
    return VerifiedDataset(manifest, csv_bytes, folder)


class ManifestCSVProvider(MarketDataProvider):
    """
    Historical data from one imported dataset version, verified on every read.
    The symbol you ask for must be the dataset's symbol.
    """

    kind = DataKind.HISTORICAL
    name = "dataset"

    def __init__(self, dataset_id: str, root: str = settings.HISTORICAL_DATA_DIR,
                 version: int | None = None):
        self.dataset_id = dataset_id
        self.root = root
        self.version = version
        self._label = None

    def source_for(self, symbol: str) -> str:
        return self._label or f"dataset:{self.dataset_id}"

    def _load_candles(self, symbol: str) -> list[Candle]:
        dataset = load_verified_dataset(self.dataset_id, self.root, self.version)
        m = dataset.manifest
        if clean_symbol(m["symbol"]) != symbol:
            raise MarketDataError(f"Dataset {self.dataset_id!r} holds {m['symbol']}, "
                                  f"not {symbol}.")
        self._label = (f"dataset:{self.dataset_id}/v{dataset.version} ({m['source']}; "
                       f"{m['interval_minutes']}-minute bars; {m['adjustment']}; "
                       f"{m['volume_coverage']} volume; sha256 "
                       f"{m['canonical_file']['sha256'][:12]})")
        return dataset.candles()
