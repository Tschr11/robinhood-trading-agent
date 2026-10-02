# Robinhood Trading Agent

A Python research and **paper-trading** agent for learning day trading with a
small account. It analyzes simulated market data, decides on trades under strict
rules, simulates them, and writes down every decision in plain English.

> **Paper trading only.** This project does not connect to Robinhood or any
> other brokerage, holds no API keys or credentials, and never moves real money.
> All trades are simulated.

## Goals

| Item | Value |
|---|---|
| Starting capital | $25 |
| Weekly contribution | $25 |
| Long-term milestone | $1,000 |
| Trading style | Day trading |
| Mode | Paper trading only |
| Autonomy | Level 3: analyzes and executes on its own, but only within the rules in `config/settings.py` |

## Project layout

```
robinhood-trading-agent/
├── README.md            This file
├── requirements.txt     Python packages (none needed yet)
├── .gitignore           Files Git should not track (secrets, logs, data)
├── config/
│   ├── settings.py      All goals and risk rules in one place
│   └── market_calendar.json  NYSE trading calendar (versioned, per-year verification)
├── data/                paper_account.db + market/<SYMBOL>.csv (git-ignored)
├── historical_data/     Imported historical market data (git-ignored; importer coming)
├── logs/                Trade journal and run logs
├── plans/
│   └── example_plan.json  Example evaluation plan
├── reports/             Evaluation reports (created on first run, git-ignored)
├── src/
│   ├── backtest.py      Historical backtesting engine (separate from paper trading)
│   ├── evaluation.py    Repeatable in-sample / out-of-sample evaluation
│   ├── main.py          Prints one round of strategy signals (no trades)
│   ├── market_data/     Market Data Engine
│   │   ├── candles.py       Candle (OHLCV), DataKind (historical/live), errors
│   │   ├── validation.py    Rejects bad candles - never repairs them
│   │   ├── dataset.py       MarketDataSet: validated + labelled candles
│   │   ├── indicators.py    SMA 20/50, RSI 14, VWAP, average volume
│   │   ├── sessions.py      New York time, NYSE calendar, regular-session bar grid
│   │   └── providers.py     Provider interface + offline CSV provider
│   ├── strategy.py      Rules-based BUY / SELL / HOLD with explanations
│   ├── risk_manager.py  Approves or blocks each trade against the rules
│   ├── paper_trader.py  Simulated account: buys, sells, P&L, daily losses
│   ├── storage.py       Saves the paper account to a local SQLite file
│   └── journal.py       Records every decision to a CSV file
└── tests/
    ├── test_risk_manager.py             Approved and rejected trades
    ├── test_risk_manager_safeguards.py  Each safeguard, including bad data
    ├── test_paper_trader.py             Buys, sells, P&L, daily losses, journal
    ├── test_exits.py                    Automatic stop-loss / take-profit exits
    ├── test_persistence.py              Restarts, saved trades, crash safety
    ├── test_market_data.py              Validation, indicators, providers
    ├── test_strategy.py                 Every entry, exit, hold and data rule
    ├── test_backtest.py                 Chronology, look-ahead, costs, exits, metrics
    ├── test_evaluation.py               Date splits, reproducibility, benchmark, reports
    ├── test_sessions.py                 Calendar, time zones, DST, session grid
    ├── market_fixtures.py               Locally generated test candles
    └── test_no_brokerage_access.py      Proves the code cannot reach a broker
```

## How the modules work together

```
market_data -> strategy -> risk_manager -> paper_trader -> journal
 (candles)    (suggest)    (allow/block)    (simulate)     (record)

The paper trader never imports market_data. Prices reach it as a plain
dict such as {"SPY": 494.00}, built by market_data.latest_prices().
```

- **`config/settings.py`** - The rulebook. Starting capital, weekly deposit,
  watchlist, and risk limits (max risk per trade, max daily loss, max trades per
  day, stop-loss, take-profit). `PAPER_TRADING = True` is the safety switch.
- **`src/market_data/`** - The Market Data Engine (built and tested):
  - **Candles (OHLCV):** timestamp, open, high, low, close, volume
  - **Validation** rejects missing values, NaN/infinity, zero or negative
    prices, negative volume, impossible bars (e.g. high below close),
    duplicate or out-of-order timestamps, timestamps without a time zone,
    and too few candles. Every problem is listed; nothing is ever filled in
  - **Labels:** every dataset and indicator result says whether it is
    `historical` or `live` data and where it came from (`source`).
    `require_kind(DataKind.LIVE)` refuses the wrong kind
  - **Indicators:** SMA 20, SMA 50, RSI 14 (Wilder), VWAP (latest session),
    20-candle average volume. Too little data raises an error - no guesses
  - **Providers:** one interface (`MarketDataProvider`), so the data source
    can change later. Validation is built into the interface, so no provider
    can skip it. Included: `CSVHistoricalProvider` (your CSV files) and
    `InMemoryProvider` (tests/backtests). There is no live provider yet
- **`src/strategy.py`** - The rules-based Strategy Engine (built and tested).
  `evaluate(dataset, expected_kind=..., has_open_position=...)` returns a
  `StrategySignal`: symbol, BUY/SELL/HOLD, timestamp, data source,
  historical/live label, and every rule marked met or not met with its
  numbers. It only *suggests*; it never places orders or touches the account.
  See "The strategy" below.
- **`src/risk_manager.py`** - The safety gate (fully built and tested). Every
  proposed BUY must include a stop-loss and pass these rules, or it is rejected
  with a reason a beginner can read:
  - paper trading only
  - the cost must fit within available cash
  - at most `MAX_OPEN_POSITIONS` open positions (1 for now)
  - the loss if the stop-loss hits must be at most `MAX_RISK_PER_TRADE_PCT` of the account
  - no trading once `MAX_DAILY_LOSS_PCT` is lost today, and no trade whose
    worst case would push past that limit
- **`src/paper_trader.py`** - The paper trading engine (built and tested). A
  pretend account that starts with $25 and is saved to disk:
  - `buy()` asks the Risk Manager first; a rejected buy changes nothing
  - `sell()` / `close_position()` lock in realized P&L and refuse to sell
    shares you don't hold
  - tracks cash, positions (shares, average entry price, stop-loss),
    realized P&L, and today's losses (reset each new day)
  - `unrealized_pnl(prices)` and `total_equity(prices)` use prices you supply
  - `account_state()` hands the real numbers to the Risk Manager
  - `check_exits(prices)` automatically closes any position whose price is at
    or below its stop-loss, or at or above its take-profit (default 2% above
    entry). Exits work even after the daily loss limit is hit. Missing or
    invalid prices are skipped (never guessed) and logged
  - every buy, sell, deposit and rejection is written to the journal
- **`src/storage.py`** - Saves the paper account in `data/paper_account.db`
  using Python's built-in SQLite (a local file, no network, no credentials):
  - cash, realized P&L, and today's loss tracking (so a restart cannot
    dodge the daily loss limit)
  - open positions: symbol, shares, average entry price, stop-loss, take-profit
  - a `transactions` ledger of every buy, sell, automatic exit and deposit
  - every change is one all-or-nothing database transaction: if anything
    fails, nothing is saved and the account reloads from disk
  - an optional `order_id` on buy/sell/deposit blocks the same order from
    being applied twice, even after a restart
- **`src/backtest.py`** - Replays historical candles through the strategy in
  time order with its own in-memory pretend account. It never touches the
  paper account, its database or the journal. See "Backtesting" below.
- **`src/evaluation.py`** - Runs a JSON plan: splits each CSV dataset into
  in-sample and out-of-sample periods by date, backtests each period
  separately, compares it with buy-and-hold over the same candles, and saves
  JSON + text reports in `reports/`. See "Evaluating the strategy" below.
- **`src/journal.py`** - Appends each decision and transaction to
  `logs/trade_journal.csv` (with realized P&L and cash afterwards) so you can
  review what the agent did and why.
- **`src/main.py`** - Loads your CSV data, prints a signal with its full
  explanation for each watchlist symbol, and writes it to the journal.
  Because the data is historical, it places **no** trades.

## Running it

Everything uses only the Python standard library. From the project root:

```
python -m src.main
```

This prints one signal per watchlist symbol from your CSV files in
`data/market/`. It does not trade.

## The strategy (`trend_vwap_v1`)

A simple, deterministic trend-following rule set. There is no machine
learning, AI model or randomness: the same data always gives the same
signal. All thresholds live in `config/settings.py`.

**Entry - BUY only if every rule passes and no position is open:**

| Rule | Condition | Idea |
|---|---|---|
| E1 | SMA 20 > SMA 50 | short-term trend is above the longer-term trend |
| E2 | close > VWAP | price is above the average paid today |
| E3 | close > SMA 20 | price is above its recent average |
| E4 | 50 <= RSI 14 <= 70 | momentum is positive but not overbought |
| E5 | latest volume >= 20-candle average volume x 1.0 | real trading interest |

**Exit - SELL if any rule passes and a position is open:**

| Rule | Condition | Idea |
|---|---|---|
| X1 | SMA 20 < SMA 50 | the trend has turned down |
| X2 | close < VWAP | price fell below today's average |
| X3 | RSI 14 >= 75 | overbought - lock in the move |

Stop-loss and take-profit exits are separate: the paper trader handles them.

**Otherwise HOLD.** The strategy also returns HOLD, never BUY or SELL, when:
- there are fewer than 50 candles (SMA 50 needs 50)
- indicator values are invalid (NaN, infinity, out of range)
- live data is older than 120 seconds or time-stamped in the future

**Historical is never treated as live.** The caller must say which kind of
data it expects (`expected_kind`); a mismatch is an error. Signals from
historical data are marked "for research only, not a live trading signal".

### Limitations - please read

- **No profitability claim.** These are common textbook indicators combined
  in a simple way. Nothing here has been shown to make money, and no result
  in this project should be read as evidence that it will.
- **Not backtested yet.** The rules have only been checked for correct
  behaviour, not for performance on real market history.
- **Indicators lag.** Moving averages and RSI describe the past; signals
  often arrive after much of a move has happened.
- **Choppy markets cause whipsaws.** In sideways markets price crosses VWAP
  and the averages often, which can produce many small losing trades.
- **The thresholds are arbitrary starting points** (50/70/75, 1.0x volume),
  not optimized or validated values.
- **Long only, one position.** It never shorts and doesn't size positions;
  the risk manager does sizing.
- **The volume rule compares the latest candle with an average that includes
  it**, and the latest candle may still be forming.
- **VWAP uses the calendar day**, so pre-market candles in your data count.
- **No news, earnings, spreads, fees or market hours** are considered.

## Market data files

The agent **never makes up prices**. It reads historical candles from CSV
files that you provide, one per symbol, in `data/market/`:

```
data/market/SPY.csv
timestamp,open,high,low,close,volume
2026-01-05T09:30:00-05:00,500.10,500.80,499.90,500.50,120000
2026-01-05T09:35:00-05:00,500.50,501.20,500.30,501.00,95000
```

- Timestamps must include a time zone (`-05:00` or `Z`) and go oldest first.
- Use the exchange's local time zone so VWAP resets on the right day.
- If a file is missing or has any invalid row, that symbol is skipped and
  the reason is written to the journal. Nothing is substituted.
- Indicators need at least 50 candles (for SMA 50).

**Historical is not live.** CSV data is labelled `historical`. Prices in a
file can be minutes or years old; never treat them as current market quotes.

## Historical data conventions (in progress)

Historical market data will be imported **offline** from files you obtain
yourself (no downloader, no API keys, no network access in this project) into
a separate, git-ignored `historical_data/` folder. The importer is not built
yet; the time and calendar rules it will enforce are:

- **Regular trading hours only:** 09:30-16:00 America/New_York, and the
  official NYSE early closes (13:00).
- **Bar-start timestamps with New York's own UTC offset:** the 09:30-09:35 bar
  is `2026-01-05T09:30:00-05:00` in winter and `2026-07-06T09:30:00-04:00` in
  summer. A timestamp whose offset doesn't match New York's offset at that
  moment is rejected, as is one without a time zone.
- **Bars must sit on the grid** (1, 5, 15 or 30 minutes from 09:30) and fit
  entirely inside the session: with 5-minute bars the last bar is 15:55 (12:55
  on an early close). Bars on weekends, holidays or unscheduled closures are
  rejected.

**Market calendar.** `config/market_calendar.json` lists, per year, the NYSE
holidays, unscheduled closures (such as 9 January 2025) and early closes, with
links to the official NYSE / ICE announcements. It is versioned, and it covers
only the years it lists (currently 2023-2026). Any other year is refused -
never treated as a normal year.

Every year starts as `"verified": false`, because the dates were transcribed
from search excerpts of the official documents (the documents could not be
opened from the build environment). **Unverified years are refused** unless a
caller explicitly passes `allow_unverified=True`. To verify a year: open the
links in its `sources`, compare every holiday, closure and early close, then
set `"verified": true`, `"verified_by"` and `"verified_on"`.

## Your saved paper account

The first run creates `data/paper_account.db` with the configured starting
capital ($25). Every later run **restores** that account - cash, open
positions, P&L and today's losses - instead of starting over.

To start fresh with a new $25 account, stop the program and delete
`data/paper_account.db`. (The CSV journal in `logs/` is separate and is kept.)

## Running the tests

The tests use Python's built-in `unittest`, so nothing needs installing:

```
python -m unittest discover tests -v
```

## Backtesting

```
python -m src.backtest
```

This replays each watchlist symbol's CSV file in `data/market/` through
`trend_vwap_v1` and prints a report labelled **BACKTEST - historical
simulation**. It uses its own pretend money; your paper account in
`data/paper_account.db` is never read or changed. From Python:

```python
from src.backtest import BacktestConfig, run_backtest
from src.market_data import CSVHistoricalProvider

data = CSVHistoricalProvider().get_candles("SPY")
result = run_backtest(data, BacktestConfig(commission_per_trade=0.0, slippage_pct=0.001))
print(result.report())
```

### How each candle is processed (no look-ahead)

1. **Open** - an order decided at the *previous* candle's close fills at this
   candle's open.
2. **High/low** - if a position is open, check the stop-loss and take-profit.
3. **Close** - the strategy is shown only the candles up to and including this
   one. A BUY or SELL becomes an order for the *next* open.
4. **Last candle of the day** - exit at the close and cancel pending orders
   (day trading: nothing is held overnight).
5. Record equity (cash + position value at the close).

Tests confirm that changing future candles never changes an earlier decision,
trade, or equity value.

### Assumptions (all configurable in `BacktestConfig`)

| Setting | Default | Meaning |
|---|---|---|
| `starting_capital` | $25 | pretend starting cash |
| `risk_per_trade_pct` | 2% | size so a stop-loss loses at most this much (uses the Risk Manager) |
| `max_position_pct` | 100% | most of the cash one position may use |
| `daily_loss_pct` | 5% | no new entries after losing this much in a day |
| `stop_loss_pct` / `take_profit_pct` | 1% / 2% | measured from the actual entry fill |
| `commission_per_trade` | $0.00 | charged on entry and on exit |
| `slippage_pct` | 0.05% | market orders fill this much worse |
| `flatten_end_of_day` | on | exit at each day's last close |
| `warmup_candles` | 50 | history needed before the first decision (minimum 50) |
| `lookback_candles` | 500 | how many recent candles the strategy sees each step (minimum 50) |

Values below 50 are rejected: the strategy's SMA 50 needs 50 candles, so a
smaller warm-up or lookback would only ever produce HOLD.

**History before the test period.** `run_backtest(..., trade_start=...)` treats
candles before `trade_start` as history only: they warm up indicators such as
RSI, but the strategy is never asked on them, nothing is bought or sold, and
they are not part of the equity curve or metrics. The first decision is at
the close of the first candle on or after `trade_start`, so the first possible
fill is the next candle's open. The backtester and the buy-and-hold benchmark
use the same rule (`first_tradable_index`) to find that first tradable candle.

Fill rules:
- **Entries and strategy exits** are market orders at the next open, with slippage.
- **Stop-loss** fills at the stop price, or at the open if the price gapped
  below it (worse), with slippage.
- **Take-profit** is a limit order: it fills at the target, or at the open if
  the price gapped above it, without slippage.
- **If a candle touches both** the stop and the target, the backtest assumes
  the stop came first (the worse outcome), because the order of prices inside
  a candle is unknown.

### Metrics

Total return, win rate, average winning trade, average losing trade, profit
factor (total won / total lost), and number of trades. Drawdown is reported
two ways, each describing ONE fall from one peak to one trough:
- **Max $ drawdown** - the largest fall in dollars, with that same fall as a
  % of its own peak;
- **Max % drawdown** - the largest fall in %, with that same fall in dollars.
  This can be a different fall (for example an early 40% drop on a small
  balance vs. a later, bigger-dollar but smaller-% drop).

**Costs are already included.** Commission and slippage are already included
in every return, P&L and drawdown figure, because the simulated fills and cash
include them. The commission and slippage totals only show how much they took;
they must not be subtracted again.

Every trade records entry/exit time and price, size, commission, slippage
cost, realized P&L and exit reason. A metric that can't be calculated (e.g. profit factor with no losing trades)
is shown as `n/a`, never as an invented number.

### Limitations - please read

- **One backtest proves nothing about the future.** It shows how fixed rules
  behaved on one stretch of past data. Good results are often luck, or the
  result of rules that happen to fit that period ("overfitting"). Test on many
  symbols and time periods, and expect real results to be worse.
- **Fills are idealized.** Real orders can fill worse than the next open, the
  stop price or the target, especially in fast or thin markets. A limit order
  touching its price may not fill at all.
- **Only candle data is used.** The path inside a candle is unknown, so stop
  vs. target ordering is a (conservative) guess.
- **Costs are estimates.** Spreads, fees and slippage vary; the defaults are
  not measurements.
- **Missing candles are not detected.** A gap in your file (e.g. a missing
  9:35 candle) is treated as if no time passed.
- **"End of day" means the last candle of each calendar day in your data,**
  not the exchange's official close.
- **Your data must be accurate.** Splits, dividends, bad prints and survivorship
  bias in the CSV files will distort results.
- **RSI is computed over the lookback window** (500 candles), so it can differ
  slightly from an RSI computed over the full history.
- **Speed.** Each candle re-validates and re-computes over its window, which
  is simple and safe but slow for very long histories.
- **Pattern-day-trader rules, settlement and taxes are not modeled.**

## Evaluating the strategy

The evaluation measures the **current** `trend_vwap_v1` exactly as configured.
It does not tune, search or change any threshold, and a plan file is not
allowed to set strategy thresholds.

```
python -m src.evaluation plans/example_plan.json --in-sample-only
python -m src.evaluation plans/example_plan.json
```

A plan lists datasets (one CSV per symbol and folder), optional date ranges,
a split date, and optional backtest *assumptions* such as costs:

```json
{
  "name": "example_baseline",
  "datasets": [
    {"symbol": "SPY", "folder": "data/market",
     "start": "2025-01-02", "end": "2025-12-31", "split_date": "2025-10-01"}
  ],
  "backtest": {"commission_per_trade": 0.0, "slippage_pct": 0.0005}
}
```

**In-sample vs. out-of-sample.** Each dataset is cut to `[start, end]`
(inclusive calendar dates). Dates **before** `split_date` are in-sample;
dates **on or after** it are out-of-sample. A trading day is never split and
the periods never overlap. Each period is backtested separately. Up to
`lookback_candles` candles from **before** the period (same file) are used only
to warm up indicators - for the out-of-sample period these are in-sample
candles, which are earlier in time, so there is no look-ahead and nothing is
fitted to them. No decision, trade or equity value happens before the
period's first candle. If no earlier candles exist (e.g. at the start of the
file), the period's own first 50 candles are the warm-up.

How to use the split honestly:
1. Explore with `--in-sample-only`. The out-of-sample candles are never put
   into a backtest (the whole CSV file is still read and validated).
2. Run the full evaluation **once** when you are done deciding.
3. If you then change the strategy because of out-of-sample results, that
   period is no longer out-of-sample.

**Out-of-sample count.** Every saved report that evaluated a dataset's
out-of-sample period adds a line to an append-only exposure log,
`reports/oos_exposure_log.jsonl` (`--exposure-log` to change it). Each report
shows, per dataset, how many earlier **saved** out-of-sample evaluations of
that exact dataset period exist. A dataset period is identified by its
symbol, the SHA-256 fingerprint of the CSV file, the split date and the end
date (the plan's `end`, or the file's last candle date).

What the count **can** detect: repeated saved evaluations of the same dataset
period, even if the plan is renamed, costs change, other datasets are added,
or reports go to a different folder.

What it **cannot** detect:
- evaluations that were never saved (e.g. calling `evaluate()` from Python
  without `save_report()`), or saved with a different exposure log;
- looks at overlapping but different periods (another split or end date);
- the same prices in an edited or re-saved CSV (it gets a new fingerprint);
- looking at the data any other way (charts, spreadsheets);
- deleting or editing the log. A damaged log line stops the evaluation with
  an error rather than silently resetting the count.

**Benchmark.** Buy-and-hold buys at the open of the first candle on which the
strategy is permitted to trade (the same `first_tradable_index` rule the
backtester uses) and sells at the period's last close, with the same starting
capital, slippage, commission and fractional-share rule.

**Report contents** (for each dataset and period, strategy vs. buy-and-hold):
total return, max $ drawdown, max % drawdown, win rate, profit factor,
number of trades, average winning and losing trade, total commission and
total slippage. Commission and slippage are already included in the returns
and must not be subtracted again. The summary counts in how many periods the
strategy beat buy-and-hold; it does not combine returns across datasets.

**Reproducibility.** Every report records the full plan, the strategy and
backtest settings, a SHA-256 fingerprint of each CSV file, and a **code
fingerprint**: SHA-256 hashes of `src/strategy.py`, `src/backtest.py`,
`src/evaluation.py`, `src/risk_manager.py` and `src/market_data/*.py`. The code
fingerprint identifies those source files only - not the Python version,
operating system, installed packages or `config/settings.py` (the relevant
settings are recorded separately). The same plan on the same files and code
gives the same results. Between runs, the generation time and the
out-of-sample counts differ: the counts grow each time a report with an
out-of-sample evaluation is saved.

**Where reports go.** `reports/<plan>_<time>.json` and `.txt`. Existing reports
are never overwritten. Saving into `data/` (paper account) or `logs/`
(journal) is refused. A missing or invalid dataset is reported as an error
while the other datasets still run.

### Assumptions and limitations of the comparison

- Buy-and-hold stays invested overnight; the strategy exits every day and is
  out of the market most of the time. They take **different risks**, and the
  returns shown are **not risk-adjusted**.
- No statistical significance test is done. A difference between the strategy
  and buy-and-hold may be pure chance, especially with few trades.
- A short out-of-sample period is very noisy. One evaluation on one stretch of
  history is **not evidence of future profitability**.
- Costs and fills follow the backtester's assumptions (see "Backtesting").
- Results are only as good as your CSV data (splits, dividends, missing
  candles and survivorship bias are not adjusted).
- Each period re-starts with fresh capital. With earlier candles available,
  the out-of-sample period can trade from its second candle; without them
  (e.g. an in-sample period at the start of the file), its first 50 candles
  are warm-up only.

## Simulated fills vs. real market execution

Every simulated order - including automatic stop-loss and take-profit
exits - fills at **exactly the price supplied** to the paper trader. Real
trading is usually worse:

- **Gaps.** Prices can jump past your stop-loss without trading in between
  (for example overnight, or after news). A stop at $495 can fill at $480.
  The simulator only models this if the supplied price itself has gapped.
- **Spread.** You buy at the higher *ask* price and sell at the lower *bid*
  price. The difference is a cost on every round trip that the simulator
  does not charge.
- **Slippage.** By the time an order reaches the market, the price may have
  moved. A real stop-loss usually becomes a market order once triggered, and
  it fills at whatever price is available, not at the stop price.
- **Timing.** The simulator only notices a stop-loss or take-profit when
  `check_exits()` is called with a new price. Between checks, the price can
  move a long way.

So paper-trading results are a **best case**. Losses in real trading can be
larger than the planned risk, and profits smaller.

## Important notes for a small account

- With $25, fractional shares are required for most stocks and ETFs.
- Day-trading rules for small accounts (such as pattern day trader limits and
  cash-settlement rules) depend on the account type and current regulations.
  These should be researched and encoded in `risk_manager.py` before the agent's
  behavior is trusted.
- Simulated results do not include slippage, spreads, or emotions, so real
  results would differ.

This project is for education and research. It is not financial advice.
