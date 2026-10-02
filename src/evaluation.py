"""
evaluation.py - A repeatable historical evaluation of trend_vwap_v1.

The goal is to MEASURE the current strategy honestly, not to make it look
good. It never changes the strategy's thresholds and contains no code that
searches for "better" parameters.

How it works:
    1. A small JSON "plan" lists the CSV datasets, an optional date range for
       each, and a split date.
    2. Each dataset is cut into two periods by calendar date:
           in-sample      = dates BEFORE split_date
           out-of-sample  = dates ON or AFTER split_date
       The periods never overlap, and each is backtested separately using
       only its own candles.
    3. Each period is compared with simply buying and holding over the same
       candles, with the same starting capital and costs.
    4. A report (JSON + text) is saved in reports/, separate from the paper
       account (data/) and the trading journal (logs/).

Why the split matters: if you look at results and then tweak rules, you
start fitting the rules to that data. The out-of-sample period is meant to
be looked at rarely, as a check. Every report counts how many earlier
out-of-sample runs exist for the same plan and data, so repeated peeking is
visible. Use --in-sample-only while exploring.

Reproducible: the report records the plan, every setting, and a SHA-256
fingerprint of each CSV file. The same plan on the same files always gives
the same results.

Run:
    python -m src.evaluation plans/example_plan.json
    python -m src.evaluation plans/example_plan.json --in-sample-only
"""

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime, timezone

from config import settings
from src import strategy
from src.backtest import (DISCLAIMER, BacktestConfig, BacktestError,
                          BacktestMetrics, compute_metrics, run_backtest)
from src.market_data import (Candle, CSVHistoricalProvider, DataKind,
                             InsufficientDataError, MarketDataError,
                             MarketDataSet, clean_symbol)

REPORT_KIND = ("EVALUATION - historical backtests of the current strategy "
               "(not live trading, not the paper account)")
IN_SAMPLE = "in-sample"
OUT_OF_SAMPLE = "out-of-sample"

PLAN_KEYS = {"name", "datasets", "backtest"}
DATASET_KEYS = {"symbol", "folder", "start", "end", "split_date"}

ASSUMPTIONS = (
    "Data is historical OHLCV from your own CSV files; nothing is live.",
    "Strategy thresholds are the current values in config/settings.py; they "
    "are not tuned by this evaluation.",
    "Each period is backtested separately with only its own candles. The first "
    "warm-up candles of each period are used for indicators only (no trades).",
    "The split is by calendar date: in-sample is before split_date, "
    "out-of-sample is on or after it.",
    "Buy-and-hold buys at the open of the first candle the strategy could "
    "trade (right after warm-up) and sells at the period's last close.",
    "Buy-and-hold invests all available cash, with the same starting capital, "
    "slippage, commission and fractional-share rule as the strategy.",
    "The strategy exits every day (day trading) while buy-and-hold stays "
    "invested overnight.",
)

LIMITATIONS = (
    "One historical sample says little about the future; results can be "
    "luck, and a small out-of-sample period is especially noisy.",
    "Looking at out-of-sample results and then changing the strategy turns "
    "that period into in-sample data. The prior-run count makes this visible.",
    "Buy-and-hold and the strategy take different risks: the strategy is out "
    "of the market most of the time, buy-and-hold never is. Returns are not "
    "risk-adjusted.",
    "No statistical significance test is done; a difference between the two "
    "may be random.",
    "Costs (slippage, commission) are assumptions, and fills are idealized "
    "(see the backtesting limitations in README.md).",
    "Results depend on the quality of your data (missing candles, splits, "
    "dividends and survivorship bias are not detected or adjusted).",
)


class EvaluationError(ValueError):
    """Raised when a plan or report location is invalid."""


# --- The plan ------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    """One CSV dataset: which symbol, where, which dates, and where to split."""
    symbol: str
    folder: str
    split_date: date
    start: date | None = None
    end: date | None = None

    def __post_init__(self):
        try:
            object.__setattr__(self, "symbol", clean_symbol(self.symbol))
        except MarketDataError as error:
            raise EvaluationError(str(error)) from None
        if not isinstance(self.folder, str) or not self.folder.strip():
            raise EvaluationError(f"{self.symbol}: folder must be a path (got {self.folder!r}).")
        for label in ["split_date", "start", "end"]:
            value = getattr(self, label)
            if value is None and label != "split_date":
                continue
            if not isinstance(value, date) or isinstance(value, datetime):
                raise EvaluationError(f"{self.symbol}: {label} must be a date (got {value!r}).")
        if self.start and self.end and self.start > self.end:
            raise EvaluationError(f"{self.symbol}: start {self.start} is after end {self.end}.")
        if (self.start and self.split_date <= self.start) or \
                (self.end and self.split_date > self.end):
            raise EvaluationError(
                f"{self.symbol}: split_date {self.split_date} must fall inside the "
                "date range, so that both periods can contain data.")


@dataclass(frozen=True)
class EvaluationPlan:
    name: str
    datasets: tuple[DatasetSpec, ...]
    backtest: BacktestConfig = BacktestConfig()

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name
                or not all(ch.isalnum() or ch in "-_" for ch in self.name)):
            raise EvaluationError("Plan name must use only letters, digits, '-' and '_' "
                                  f"(got {self.name!r}).")
        if not isinstance(self.datasets, tuple) or not self.datasets:
            raise EvaluationError("A plan needs at least one dataset.")
        if not all(isinstance(d, DatasetSpec) for d in self.datasets):
            raise EvaluationError("Every dataset must be a DatasetSpec.")
        if not isinstance(self.backtest, BacktestConfig):
            raise EvaluationError("backtest must be a BacktestConfig.")

    def to_dict(self) -> dict:
        return {"name": self.name,
                "datasets": [_spec_dict(d) for d in self.datasets],
                "backtest": asdict(self.backtest)}

    def fingerprint(self) -> str:
        """A short code that changes if anything in the plan changes."""
        return _sha256(json.dumps(self.to_dict(), sort_keys=True).encode())


def plan_from_dict(raw) -> EvaluationPlan:
    """Build a plan from parsed JSON, rejecting anything unexpected."""
    if not isinstance(raw, dict):
        raise EvaluationError("A plan must be a JSON object.")
    if "strategy" in raw:
        raise EvaluationError(
            "Plans cannot set strategy thresholds. The evaluation always measures "
            "the current strategy settings in config/settings.py.")
    unknown = set(raw) - PLAN_KEYS
    if unknown:
        raise EvaluationError(f"Unknown plan key(s): {', '.join(sorted(unknown))}.")
    datasets = raw.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise EvaluationError("A plan needs a non-empty 'datasets' list.")

    specs = []
    for number, item in enumerate(datasets, start=1):
        if not isinstance(item, dict):
            raise EvaluationError(f"Dataset {number} must be a JSON object.")
        unknown = set(item) - DATASET_KEYS
        if unknown:
            raise EvaluationError(f"Dataset {number}: unknown key(s) {', '.join(sorted(unknown))}.")
        if "symbol" not in item or "split_date" not in item:
            raise EvaluationError(f"Dataset {number} needs 'symbol' and 'split_date'.")
        specs.append(DatasetSpec(
            symbol=item["symbol"],
            folder=item.get("folder", settings.MARKET_DATA_DIR),
            split_date=_parse_date(item["split_date"], f"dataset {number} split_date"),
            start=_parse_date(item.get("start"), f"dataset {number} start"),
            end=_parse_date(item.get("end"), f"dataset {number} end")))

    overrides = raw.get("backtest", {})
    if not isinstance(overrides, dict):
        raise EvaluationError("'backtest' must be a JSON object of backtest settings.")
    allowed = {f.name for f in fields(BacktestConfig)}
    unknown = set(overrides) - allowed
    if unknown:
        raise EvaluationError(f"Unknown backtest setting(s): {', '.join(sorted(unknown))}.")
    try:
        backtest = BacktestConfig(**overrides)
    except BacktestError as error:
        raise EvaluationError(str(error)) from None

    return EvaluationPlan(raw.get("name", ""), tuple(specs), backtest)


def load_plan(path: str) -> EvaluationPlan:
    try:
        with open(path) as f:
            raw = json.load(f)
    except OSError as error:
        raise EvaluationError(f"Could not read plan {path}: {error}") from None
    except json.JSONDecodeError as error:
        raise EvaluationError(f"Plan {path} is not valid JSON: {error}") from None
    return plan_from_dict(raw)


# --- Splitting by date --------------------------------------------------------------

def split_candles(candles, spec: DatasetSpec) -> tuple[list[Candle], list[Candle]]:
    """
    Keep candles inside [start, end] (inclusive calendar dates), then split:
        in-sample      date <  split_date
        out-of-sample  date >= split_date
    Dates are read in each candle's own time zone, like everywhere else.
    """
    kept = [c for c in candles
            if (spec.start is None or c.timestamp.date() >= spec.start)
            and (spec.end is None or c.timestamp.date() <= spec.end)]
    in_sample = [c for c in kept if c.timestamp.date() < spec.split_date]
    out_of_sample = [c for c in kept if c.timestamp.date() >= spec.split_date]
    return in_sample, out_of_sample


# --- Buy-and-hold benchmark -----------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkResult:
    """Buying at the first tradable open and selling at the last close."""
    entry_time: datetime
    entry_price: float       # fill, including slippage
    exit_time: datetime
    exit_price: float        # fill, including slippage
    shares: float
    final_equity: float
    total_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    total_commission: float
    total_slippage: float


def buy_and_hold(candles, config: BacktestConfig) -> BenchmarkResult:
    """
    Buy-and-hold over the same candles the strategy could trade: buy at the
    open of candle number `warmup_candles` (the strategy's first possible
    fill), sell at the last close. Same capital, slippage and commission.
    """
    start = config.warmup_candles
    if len(candles) <= start:
        raise InsufficientDataError(
            f"Buy-and-hold needs more than {start} candles (got {len(candles)}).")
    first, last = candles[start], candles[-1]
    slip, fee = config.slippage_pct, config.commission_per_trade

    entry_fill = first.open * (1 + slip)
    spendable = config.starting_capital - fee
    shares = spendable / entry_fill if spendable > 0 else 0.0
    shares = (float(int(shares)) if not config.allow_fractional_shares
              else int(shares * 10_000 + 1e-9) / 10_000)
    traded = shares > 0
    commission = 2 * fee if traded else 0.0
    cash = config.starting_capital - (shares * entry_fill + fee if traded else 0.0)

    exit_fill = last.close * (1 - slip)
    # Equity at each close while holding; the last point is after selling.
    values = [cash + shares * c.close for c in candles[start:-1]]
    final = cash + shares * exit_fill - (fee if traded else 0.0)
    values.append(final)
    curve = compute_metrics([], values, config.starting_capital)

    return BenchmarkResult(
        entry_time=first.timestamp, entry_price=entry_fill,
        exit_time=last.timestamp, exit_price=exit_fill, shares=shares,
        final_equity=final,
        total_return_pct=(final - config.starting_capital) / config.starting_capital,
        max_drawdown=curve.max_drawdown, max_drawdown_pct=curve.max_drawdown_pct,
        total_commission=commission,
        total_slippage=shares * (entry_fill - first.open) + shares * (last.close - exit_fill))


# --- Results -------------------------------------------------------------------------------

@dataclass(frozen=True)
class PeriodResult:
    period: str                       # "in-sample" or "out-of-sample"
    status: str                       # "ok", "skipped" or "not evaluated"
    reason: str = ""
    candles: int = 0
    first_candle: datetime | None = None
    last_candle: datetime | None = None
    strategy: BacktestMetrics | None = None
    benchmark: BenchmarkResult | None = None

    @property
    def strategy_beat_benchmark(self) -> bool | None:
        if self.status != "ok":
            return None
        return self.strategy.total_return_pct > self.benchmark.total_return_pct


@dataclass(frozen=True)
class DatasetResult:
    spec: DatasetSpec
    path: str
    sha256: str | None                # None if the file couldn't be read
    status: str                       # "ok" or "error"
    error: str
    periods: tuple[PeriodResult, ...]


@dataclass(frozen=True)
class EvaluationReport:
    kind: str
    plan: EvaluationPlan
    plan_fingerprint: str
    data_fingerprint: str
    strategy_name: str
    strategy_config: dict
    generated_at: datetime
    in_sample_only: bool
    prior_out_of_sample_runs: int
    datasets: tuple[DatasetResult, ...]

    def summary(self) -> dict:
        out = {}
        for period in [IN_SAMPLE, OUT_OF_SAMPLE]:
            done = [p for d in self.datasets for p in d.periods
                    if p.period == period and p.status == "ok"]
            out[period] = {
                "periods_evaluated": len(done),
                "strategy_beat_buy_and_hold": sum(1 for p in done if p.strategy_beat_benchmark),
                "total_trades": sum(p.strategy.number_of_trades for p in done),
            }
        return out

    def to_dict(self) -> dict:
        return _jsonable({
            "kind": self.kind,
            "generated_at": self.generated_at,
            "plan": self.plan.to_dict(),
            "plan_fingerprint": self.plan_fingerprint,
            "data_fingerprint": self.data_fingerprint,
            "strategy_name": self.strategy_name,
            "strategy_config": self.strategy_config,
            "in_sample_only": self.in_sample_only,
            "prior_out_of_sample_runs": self.prior_out_of_sample_runs,
            "datasets": [{"symbol": d.spec.symbol, "path": d.path, "sha256": d.sha256,
                          "status": d.status, "error": d.error,
                          "periods": [asdict(p) for p in d.periods]}
                         for d in self.datasets],
            "summary": self.summary(),
            "assumptions": list(ASSUMPTIONS),
            "limitations": list(LIMITATIONS),
            "disclaimer": DISCLAIMER,
        })

    def to_text(self) -> str:
        lines = [self.kind,
                 f"Plan '{self.plan.name}' (fingerprint {self.plan_fingerprint[:12]}), "
                 f"data fingerprint {self.data_fingerprint[:12]}",
                 f"Strategy {self.strategy_name}: "
                 + ", ".join(f"{k}={v}" for k, v in self.strategy_config.items()
                             if k != "name"),
                 f"Generated {self.generated_at.isoformat()}"]
        c = self.plan.backtest
        lines.append(f"Costs: commission ${c.commission_per_trade:.2f}/order, slippage "
                     f"{c.slippage_pct * 100:.3f}%; start ${c.starting_capital:,.2f}")
        if self.in_sample_only:
            lines.append("Out-of-sample results were NOT evaluated (in-sample-only run).")
        lines.append(f"Earlier out-of-sample runs of this plan and data: "
                     f"{self.prior_out_of_sample_runs}"
                     + ("  <- each extra look weakens the out-of-sample test"
                        if self.prior_out_of_sample_runs else ""))

        for d in self.datasets:
            lines += ["", f"== {d.spec.symbol}  ({d.path}, sha256 {str(d.sha256)[:12]})"]
            if d.status != "ok":
                lines.append(f"   ERROR: {d.error}")
                continue
            for p in d.periods:
                lines.append(f"-- {p.period}: " + (
                    f"{p.first_candle.isoformat()} to {p.last_candle.isoformat()}, "
                    f"{p.candles} candles" if p.status == "ok"
                    else f"{p.status.upper()} - {p.reason}"))
                if p.status == "ok":
                    lines += _comparison_table(p)

        lines += ["", "Summary (count of datasets, not a combined return):"]
        for period, s in self.summary().items():
            lines.append(f"  {period}: strategy beat buy-and-hold in "
                         f"{s['strategy_beat_buy_and_hold']} of {s['periods_evaluated']} "
                         f"evaluated period(s); {s['total_trades']} trades in total")
        lines += ["", "Assumptions:"] + [f"  - {a}" for a in ASSUMPTIONS]
        lines += ["", "Limitations:"] + [f"  - {item}" for item in LIMITATIONS]
        lines += ["", f"NOTE: {DISCLAIMER}"]
        return "\n".join(lines)


def _comparison_table(p: PeriodResult) -> list[str]:
    s, b = p.strategy, p.benchmark

    def pct(v):
        return "n/a" if v is None else f"{v * 100:+.2f}%"

    def money(v):
        return "n/a" if v is None else f"${v:,.2f}"

    def number(v):
        return "n/a" if v is None else f"{v:.2f}"

    rows = [("Total return", pct(s.total_return_pct), pct(b.total_return_pct)),
            ("Max drawdown", f"{money(s.max_drawdown)} ({s.max_drawdown_pct * 100:.2f}%)",
             f"{money(b.max_drawdown)} ({b.max_drawdown_pct * 100:.2f}%)"),
            ("Win rate", "n/a" if s.win_rate is None else f"{s.win_rate * 100:.1f}%", "n/a"),
            ("Profit factor", number(s.profit_factor), "n/a"),
            ("Number of trades", str(s.number_of_trades), "1" if b.shares > 0 else "0"),
            ("Average win", money(s.average_win), "n/a"),
            ("Average loss", money(s.average_loss), "n/a"),
            ("Commissions", money(s.total_commission), money(b.total_commission)),
            ("Slippage", money(s.total_slippage), money(b.total_slippage))]
    out = [f"   {'':<18}{'Strategy':>22}{'Buy-and-hold':>22}"]
    out += [f"   {name:<18}{left:>22}{right:>22}" for name, left, right in rows]
    return out


# --- Running an evaluation -----------------------------------------------------------------

def evaluate(plan: EvaluationPlan, *, in_sample_only: bool = False,
             reports_dir: str = settings.REPORTS_DIR,
             now: datetime | None = None) -> EvaluationReport:
    """Run every dataset and period in the plan. Changes nothing anywhere."""
    if not isinstance(plan, EvaluationPlan):
        raise EvaluationError("evaluate() needs an EvaluationPlan.")
    now = now or datetime.now(timezone.utc)
    config = strategy.DEFAULT_CONFIG                  # the CURRENT thresholds, untouched

    results = [_evaluate_dataset(spec, plan.backtest, config, in_sample_only)
               for spec in plan.datasets]
    data_fingerprint = _sha256(json.dumps(
        [[r.spec.symbol, r.sha256] for r in results]).encode())
    return EvaluationReport(
        kind=REPORT_KIND, plan=plan, plan_fingerprint=plan.fingerprint(),
        data_fingerprint=data_fingerprint, strategy_name=config.name,
        strategy_config=asdict(config), generated_at=now,
        in_sample_only=in_sample_only,
        prior_out_of_sample_runs=_count_prior_oos_runs(
            reports_dir, plan.fingerprint(), data_fingerprint),
        datasets=tuple(results))


def _evaluate_dataset(spec, backtest_config, strategy_config, in_sample_only):
    provider = CSVHistoricalProvider(spec.folder)
    path = provider.path_for(spec.symbol)
    sha = _file_sha256(path)
    try:
        data = provider.get_candles(spec.symbol)
    except MarketDataError as error:
        return DatasetResult(spec, path, sha, "error", str(error), ())

    in_sample, out_of_sample = split_candles(data.candles, spec)
    periods = [_evaluate_period(IN_SAMPLE, in_sample, data, backtest_config, strategy_config)]
    if in_sample_only:
        periods.append(PeriodResult(OUT_OF_SAMPLE, "not evaluated",
                                    "in-sample-only run; out-of-sample kept unseen"))
    else:
        periods.append(_evaluate_period(OUT_OF_SAMPLE, out_of_sample, data,
                                        backtest_config, strategy_config))
    return DatasetResult(spec, path, sha, "ok", "", tuple(periods))


def _evaluate_period(label, candles, data, backtest_config, strategy_config):
    needed = backtest_config.warmup_candles + 1
    if len(candles) < needed:
        return PeriodResult(label, "skipped",
                            f"needs at least {needed} candles, has {len(candles)}",
                            candles=len(candles))
    period_data = MarketDataSet(data.symbol, DataKind.HISTORICAL,
                                f"{data.source} [{label}]", tuple(candles))
    result = run_backtest(period_data, backtest_config, strategy_config=strategy_config)
    return PeriodResult(label, "ok", candles=len(candles),
                        first_candle=candles[0].timestamp, last_candle=candles[-1].timestamp,
                        strategy=result.metrics,
                        benchmark=buy_and_hold(candles, backtest_config))


# --- Saving reports ----------------------------------------------------------------------

def save_report(report: EvaluationReport,
                reports_dir: str = settings.REPORTS_DIR) -> tuple[str, str]:
    """
    Write <name>_<time>.json and .txt into `reports_dir`. Existing reports are
    never overwritten, and the paper account / journal folders are refused.
    """
    folder = os.path.abspath(reports_dir)
    protected = {os.path.abspath(p) for p in [
        settings.DATA_DIR, settings.LOG_DIR,
        os.path.dirname(settings.DATABASE_FILE) or ".",
        os.path.dirname(settings.JOURNAL_FILE) or "."]}
    for place in protected:
        if folder == place or folder.startswith(place + os.sep):
            raise EvaluationError(
                f"Reports can't be saved in {reports_dir}: that folder holds the "
                "paper account or the trading journal.")
    os.makedirs(folder, exist_ok=True)

    stem = f"{report.plan.name}_{report.generated_at.strftime('%Y%m%dT%H%M%S')}"
    if report.in_sample_only:
        stem += "_in-sample-only"
    base, number = os.path.join(folder, stem), 1
    while os.path.exists(base + ".json") or os.path.exists(base + ".txt"):
        number += 1
        base = os.path.join(folder, f"{stem}_{number}")

    with open(base + ".json", "x") as f:
        json.dump(report.to_dict(), f, indent=2, sort_keys=True)
    with open(base + ".txt", "x") as f:
        f.write(report.to_text() + "\n")
    return base + ".json", base + ".txt"


def _count_prior_oos_runs(reports_dir, plan_fingerprint, data_fingerprint) -> int:
    if not os.path.isdir(reports_dir):
        return 0
    count = 0
    for name in sorted(os.listdir(reports_dir)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(reports_dir, name)) as f:
                old = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if (isinstance(old, dict) and old.get("plan_fingerprint") == plan_fingerprint
                and old.get("data_fingerprint") == data_fingerprint
                and old.get("in_sample_only") is False):
            count += 1
    return count


# --- Small helpers ----------------------------------------------------------------------------

def _parse_date(value, label):
    if value is None:
        return None
    if not isinstance(value, str):
        raise EvaluationError(f"{label} must be a date like 2025-10-01 (got {value!r}).")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise EvaluationError(f"{label} must be a date like 2025-10-01 (got {value!r}).") from None


def _spec_dict(spec: DatasetSpec) -> dict:
    return {"symbol": spec.symbol, "folder": spec.folder,
            "split_date": spec.split_date.isoformat(),
            "start": spec.start.isoformat() if spec.start else None,
            "end": spec.end.isoformat() if spec.end else None}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return _sha256(f.read())
    except OSError:
        return None


def _jsonable(value):
    """Convert dates and nested objects into plain JSON values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


# --- Command line ------------------------------------------------------------------------------

def main(argv=None) -> tuple[str, str]:
    parser = argparse.ArgumentParser(description="Evaluate trend_vwap_v1 on historical data.")
    parser.add_argument("plan", help="path to a JSON evaluation plan")
    parser.add_argument("--in-sample-only", action="store_true",
                        help="evaluate only the in-sample period (keep out-of-sample unseen)")
    parser.add_argument("--reports-dir", default=settings.REPORTS_DIR)
    args = parser.parse_args(argv)

    report = evaluate(load_plan(args.plan), in_sample_only=args.in_sample_only,
                      reports_dir=args.reports_dir)
    paths = save_report(report, args.reports_dir)
    print(report.to_text())
    print(f"\nSaved: {paths[0]}\n       {paths[1]}")
    return paths


if __name__ == "__main__":
    main()
