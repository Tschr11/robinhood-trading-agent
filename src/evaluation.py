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
       The periods never overlap and each is backtested separately. Candles
       from BEFORE a period may warm up its indicators, but no decision or
       trade happens before the period's first candle.
    3. Each period is compared with simply buying and holding over the same
       candles, with the same starting capital and costs.
    4. A report (JSON + text) is saved in reports/, separate from the paper
       account (data/) and the trading journal (logs/).

Why the split matters: if you look at results and then tweak rules, you
start fitting the rules to that data. The out-of-sample period is meant to
be looked at rarely, as a check. Each SAVED out-of-sample evaluation is
written to an append-only exposure log, and every report shows, per dataset,
how many earlier saved out-of-sample evaluations of that exact dataset period
exist (see OOS_COUNT_NOTES for what this can and cannot detect). Use
--in-sample-only while exploring.

Reproducible: the report records the plan, every setting, a SHA-256
fingerprint of each CSV file, and a fingerprint of the evaluation source
code. The same plan on the same files and code gives the same results; only
the generation time and the out-of-sample counts (which grow as reports are
saved) differ between runs.

Run:
    python -m src.evaluation plans/example_plan.json
    python -m src.evaluation plans/example_plan.json --in-sample-only
"""

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields, replace
from datetime import date, datetime, timezone

from config import settings
from src import strategy
from src.backtest import (DISCLAIMER, BacktestConfig, BacktestError,
                          BacktestMetrics, compute_metrics,
                          first_tradable_index, run_backtest)
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
    "Each period is backtested separately. Up to lookback_candles candles from "
    "immediately BEFORE the period (same file) warm up the indicators only: no "
    "decision, trade or equity value happens before the period's first candle. "
    "For the out-of-sample period these come from the earlier in-sample period "
    "(no look-ahead; nothing is fitted to them). With no earlier candles, the "
    "period's own first warm-up candles are used for indicators only.",
    "The split is by calendar date: in-sample is before split_date, "
    "out-of-sample is on or after it.",
    "Buy-and-hold buys at the open of the first candle on which the strategy "
    "is permitted to trade (the same rule the backtester uses) and sells at "
    "the period's last close.",
    "Buy-and-hold invests all available cash, with the same starting capital, "
    "slippage, commission and fractional-share rule as the strategy.",
    "The strategy exits every day (day trading) while buy-and-hold stays "
    "invested overnight.",
    "Commission and slippage are already included in every return, P&L and "
    "drawdown figure. The cost rows show how much they took; do not subtract "
    "them again.",
)

LIMITATIONS = (
    "One historical sample says little about the future; results can be "
    "luck, and a small out-of-sample period is especially noisy.",
    "Looking at out-of-sample results and then changing the strategy turns "
    "that period into in-sample data. The out-of-sample count helps make this "
    "visible but only counts saved evaluations (see the notes on the count).",
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


OOS_COUNT_NOTES = (
    "Counts only out-of-sample evaluations whose report was SAVED (the command "
    "line always saves) to this exposure log. Evaluations run from Python "
    "without save_report(), and reports saved with a different exposure log, "
    "are not counted.",
    "A dataset period is matched exactly by symbol, CSV SHA-256 fingerprint, "
    "split date and end date. The plan name, costs, other datasets in the plan "
    "and the report folder do not affect the count.",
    "It cannot detect looks at overlapping but different periods (another "
    "split or end date), the same prices in an edited or re-saved CSV (new "
    "fingerprint), or viewing the data in other ways such as charts.",
    "Deleting or editing the exposure log changes the count.",
)

# Source files whose code determines the evaluation's results.
CODE_FILES = ("src/strategy.py", "src/backtest.py", "src/evaluation.py",
              "src/risk_manager.py")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    max_drawdown: float              # largest $ fall ...
    max_drawdown_pct: float          # ... and that same fall in %
    total_commission: float
    total_slippage: float
    largest_pct_drawdown: float = 0.0          # largest % fall ...
    largest_pct_drawdown_dollars: float = 0.0  # ... and that same fall in $


def buy_and_hold(candles, config: BacktestConfig,
                 trade_start: datetime | None = None) -> BenchmarkResult:
    """
    Buy-and-hold over the same candles the strategy may trade: buy at the open
    of the first candle on which the strategy is permitted to trade (decided
    by backtest.first_tradable_index, exactly as the backtester does), sell at
    the last close. Same capital, slippage and commission. Candles before
    `trade_start` are history only.
    """
    start = first_tradable_index(candles, config.warmup_candles, trade_start)
    if start >= len(candles):
        raise InsufficientDataError(
            f"Buy-and-hold has no tradable candle: the first permitted entry is "
            f"candle {start + 1}, but there are only {len(candles)}.")
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
        largest_pct_drawdown=curve.largest_pct_drawdown,
        largest_pct_drawdown_dollars=curve.largest_pct_drawdown_dollars,
        total_commission=commission,
        total_slippage=shares * (entry_fill - first.open) + shares * (last.close - exit_fill))


# --- Results -------------------------------------------------------------------------------

@dataclass(frozen=True)
class PeriodResult:
    period: str                       # "in-sample" or "out-of-sample"
    status: str                       # "ok", "skipped" or "not evaluated"
    reason: str = ""
    candles: int = 0                  # candles inside the period
    history_candles: int = 0          # earlier candles used for warm-up only
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
    oos_key: dict | None = None       # identifies this dataset's out-of-sample period
    prior_oos_evaluations: int | None = None   # earlier SAVED looks at it

    @property
    def oos_evaluated(self) -> bool:
        return any(p.period == OUT_OF_SAMPLE and p.status == "ok" for p in self.periods)


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
    code_fingerprint: dict
    exposure_log: str
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
            "code_fingerprint": self.code_fingerprint,
            "exposure_log": self.exposure_log,
            "oos_count_notes": list(OOS_COUNT_NOTES),
            "datasets": [{"symbol": d.spec.symbol, "path": d.path, "sha256": d.sha256,
                          "status": d.status, "error": d.error,
                          "oos_key": d.oos_key,
                          "oos_evaluated": d.oos_evaluated,
                          "prior_oos_evaluations": d.prior_oos_evaluations,
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
                 f"Generated {self.generated_at.isoformat()}",
                 f"Code fingerprint {self.code_fingerprint['combined'][:12]} "
                 "(source files only - not the Python version, installed "
                 "packages or config/settings.py)"]
        c = self.plan.backtest
        lines.append(f"Costs: commission ${c.commission_per_trade:.2f}/order, slippage "
                     f"{c.slippage_pct * 100:.3f}%; start ${c.starting_capital:,.2f}")
        if self.in_sample_only:
            lines.append("Out-of-sample results were NOT evaluated (in-sample-only run).")

        for d in self.datasets:
            lines += ["", f"== {d.spec.symbol}  ({d.path}, sha256 {str(d.sha256)[:12]})"]
            if d.status != "ok":
                lines.append(f"   ERROR: {d.error}")
                continue
            n = d.prior_oos_evaluations
            lines.append(f"   Earlier SAVED out-of-sample evaluations of this exact dataset "
                         f"period ({d.oos_key['split_date']} to {d.oos_key['end_date']}): {n}"
                         + ("  <- each extra look weakens the out-of-sample test" if n else ""))
            for p in d.periods:
                lines.append(f"-- {p.period}: " + (
                    f"{p.first_candle.isoformat()} to {p.last_candle.isoformat()}, "
                    f"{p.candles} candles (+{p.history_candles} earlier candles for "
                    "indicator warm-up only)" if p.status == "ok"
                    else f"{p.status.upper()} - {p.reason}"))
                if p.status == "ok":
                    lines += _comparison_table(p)

        lines += ["", "Summary (count of datasets, not a combined return):"]
        for period, s in self.summary().items():
            lines.append(f"  {period}: strategy beat buy-and-hold in "
                         f"{s['strategy_beat_buy_and_hold']} of {s['periods_evaluated']} "
                         f"evaluated period(s); {s['total_trades']} trades in total")
        lines += ["", f"About the out-of-sample count (log: {self.exposure_log}):"]
        lines += [f"  - {note}" for note in OOS_COUNT_NOTES]
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
            ("Max $ drawdown", f"{money(s.max_drawdown)} ({s.max_drawdown_pct * 100:.2f}%)",
             f"{money(b.max_drawdown)} ({b.max_drawdown_pct * 100:.2f}%)"),
            ("Max % drawdown",
             f"{s.largest_pct_drawdown * 100:.2f}% ({money(s.largest_pct_drawdown_dollars)})",
             f"{b.largest_pct_drawdown * 100:.2f}% ({money(b.largest_pct_drawdown_dollars)})"),
            ("Win rate", "n/a" if s.win_rate is None else f"{s.win_rate * 100:.1f}%", "n/a"),
            ("Profit factor", number(s.profit_factor), "n/a"),
            ("Number of trades", str(s.number_of_trades), "1" if b.shares > 0 else "0"),
            ("Average win", money(s.average_win), "n/a"),
            ("Average loss", money(s.average_loss), "n/a"),
            ("Commissions *", money(s.total_commission), money(b.total_commission)),
            ("Slippage *", money(s.total_slippage), money(b.total_slippage))]
    out = [f"   {'':<18}{'Strategy':>22}{'Buy-and-hold':>22}"]
    out += [f"   {name:<18}{left:>22}{right:>22}" for name, left, right in rows]
    out += ["   Max $ drawdown: largest fall in dollars, with that same fall in %.",
            "   Max % drawdown: largest fall in %, with that same fall in $ "
            "(may be a different fall).",
            "   * Already included in Total return and all P&L figures - "
            "do not subtract again."]
    return out


# --- Running an evaluation -----------------------------------------------------------------

def evaluate(plan: EvaluationPlan, *, in_sample_only: bool = False,
             exposure_log: str = settings.OOS_EXPOSURE_LOG,
             now: datetime | None = None) -> EvaluationReport:
    """
    Run every dataset and period in the plan. Changes nothing anywhere: it
    only READS the exposure log to count earlier saved out-of-sample looks.
    Saving the report (save_report) is what records a new look.
    """
    if not isinstance(plan, EvaluationPlan):
        raise EvaluationError("evaluate() needs an EvaluationPlan.")
    now = now or datetime.now(timezone.utc)
    config = strategy.DEFAULT_CONFIG                  # the CURRENT thresholds, untouched

    log = _read_exposure_log(exposure_log)
    results = []
    for spec in plan.datasets:
        result = _evaluate_dataset(spec, plan.backtest, config, in_sample_only)
        if result.oos_key is not None:
            result = replace(result, prior_oos_evaluations=sum(
                1 for record in log if _same_oos_key(record, result.oos_key)))
        results.append(result)
    data_fingerprint = _sha256(json.dumps(
        [[r.spec.symbol, r.sha256] for r in results]).encode())
    return EvaluationReport(
        kind=REPORT_KIND, plan=plan, plan_fingerprint=plan.fingerprint(),
        data_fingerprint=data_fingerprint, strategy_name=config.name,
        strategy_config=asdict(config), generated_at=now,
        in_sample_only=in_sample_only, code_fingerprint=code_fingerprint(),
        exposure_log=exposure_log, datasets=tuple(results))


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
    return DatasetResult(spec, path, sha, "ok", "", tuple(periods),
                         oos_key=oos_key(spec, sha, data.candles))


def _evaluate_period(label, candles, data, backtest_config, strategy_config):
    if not candles:
        return PeriodResult(label, "skipped", "no candles in this period")
    # Up to lookback_candles EARLIER candles from the same file: indicator
    # warm-up only. They come strictly before the period's first candle.
    start = candles[0].timestamp
    history = [c for c in data.candles if c.timestamp < start]
    history = history[-backtest_config.lookback_candles:]
    combined = tuple(history + list(candles))
    if first_tradable_index(combined, backtest_config.warmup_candles, start) >= len(combined):
        return PeriodResult(
            label, "skipped",
            f"no tradable candle: needs {backtest_config.warmup_candles} candles of "
            "history before the first decision plus one more to trade on; has "
            f"{len(history)} earlier candle(s) and {len(candles)} in the period",
            candles=len(candles), history_candles=len(history))
    period_data = MarketDataSet(data.symbol, DataKind.HISTORICAL,
                                f"{data.source} [{label}]", combined)
    result = run_backtest(period_data, backtest_config, strategy_config=strategy_config,
                          trade_start=start)
    return PeriodResult(label, "ok", candles=len(candles), history_candles=len(history),
                        first_candle=candles[0].timestamp, last_candle=candles[-1].timestamp,
                        strategy=result.metrics,
                        benchmark=buy_and_hold(combined, backtest_config, trade_start=start))


# --- Out-of-sample exposure tracking -----------------------------------------------------

def oos_key(spec: DatasetSpec, sha256: str | None, candles) -> dict | None:
    """
    Identifies one dataset's out-of-sample period: symbol, CSV fingerprint,
    split date, and end date (the plan's end, or else the file's last candle
    date, which is where the out-of-sample period then ends).
    """
    if sha256 is None or not candles:
        return None
    end = spec.end or candles[-1].timestamp.date()
    return {"symbol": spec.symbol, "sha256": sha256,
            "split_date": spec.split_date.isoformat(), "end_date": end.isoformat()}


OOS_KEY_FIELDS = ("symbol", "sha256", "split_date", "end_date")


def _same_oos_key(record: dict, key: dict) -> bool:
    return all(record.get(field) == key[field] for field in OOS_KEY_FIELDS)


def _read_exposure_log(path: str) -> list[dict]:
    """Every record in the exposure log. A damaged line is an error, never skipped."""
    if not os.path.exists(path):
        return []
    if not os.path.isfile(path):
        raise EvaluationError(f"Exposure log {path} is not a file.")
    records = []
    with open(path) as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = None
            if not isinstance(record, dict) or not all(
                    isinstance(record.get(field), str) for field in OOS_KEY_FIELDS):
                raise EvaluationError(
                    f"Exposure log {path} line {number} is unreadable. Fix or restore "
                    "it; it is never ignored, because that would hide earlier looks.")
            records.append(record)
    return records


# --- Code fingerprint ------------------------------------------------------------------------

def code_fingerprint() -> dict:
    """
    SHA-256 of each source file that determines evaluation results, plus a
    combined hash. This identifies SOURCE FILES only - not the Python version,
    the operating system, installed packages, or config/settings.py (whose
    relevant values are recorded separately in the report).
    """
    market_dir = os.path.join(PROJECT_ROOT, "src", "market_data")
    paths = list(CODE_FILES) + sorted(
        f"src/market_data/{name}" for name in os.listdir(market_dir) if name.endswith(".py"))
    files = {rel: _file_sha256(os.path.join(PROJECT_ROOT, rel)) for rel in sorted(paths)}
    return {"files": files,
            "combined": _sha256(json.dumps(files, sort_keys=True).encode()),
            "scope": "source files only; not the Python version, installed "
                     "packages, operating system or config/settings.py"}


# --- Saving reports ----------------------------------------------------------------------

def _refuse_protected(folder_path: str, what: str) -> str:
    """Refuse folders that hold the paper account or the trading journal."""
    folder = os.path.abspath(folder_path)
    protected = {os.path.abspath(p) for p in [
        settings.DATA_DIR, settings.LOG_DIR,
        os.path.dirname(settings.DATABASE_FILE) or ".",
        os.path.dirname(settings.JOURNAL_FILE) or "."]}
    for place in protected:
        if folder == place or folder.startswith(place + os.sep):
            raise EvaluationError(
                f"{what} can't be saved in {folder_path}: that folder holds the "
                "paper account or the trading journal.")
    return folder


def save_report(report: EvaluationReport, reports_dir: str = settings.REPORTS_DIR,
                exposure_log: str = settings.OOS_EXPOSURE_LOG) -> tuple[str, str]:
    """
    Write <name>_<time>.json and .txt into `reports_dir` (never overwriting),
    THEN append one line per dataset whose out-of-sample period was evaluated
    to `exposure_log`. The paper account / journal folders are refused.
    """
    folder = _refuse_protected(reports_dir, "Reports")
    log_folder = _refuse_protected(os.path.dirname(exposure_log) or ".", "The exposure log")
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

    # Record the out-of-sample looks only now that the report is saved.
    looks = [d for d in report.datasets
             if not report.in_sample_only and d.oos_evaluated and d.oos_key]
    if looks:
        try:
            os.makedirs(log_folder, exist_ok=True)
            with open(exposure_log, "a") as f:
                for d in looks:
                    record = dict(d.oos_key, recorded_at=report.generated_at.isoformat(),
                                  report=base + ".json", plan_name=report.plan.name,
                                  code_fingerprint=report.code_fingerprint["combined"])
                    f.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError as error:
            raise EvaluationError(
                f"The report was saved ({base}.json), but the exposure log {exposure_log} "
                f"could not be updated: {error}. Add the look to the log manually.") from None
    return base + ".json", base + ".txt"


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
    parser.add_argument("--exposure-log", default=settings.OOS_EXPOSURE_LOG,
                        help="append-only log of saved out-of-sample evaluations")
    args = parser.parse_args(argv)

    report = evaluate(load_plan(args.plan), in_sample_only=args.in_sample_only,
                      exposure_log=args.exposure_log)
    paths = save_report(report, args.reports_dir, args.exposure_log)
    print(report.to_text())
    print(f"\nSaved: {paths[0]}\n       {paths[1]}")
    return paths


if __name__ == "__main__":
    main()
