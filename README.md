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
│   └── settings.py      All goals and risk rules in one place
├── data/                paper_account.db - the saved paper account (git-ignored)
├── logs/                Trade journal and run logs
├── src/
│   ├── main.py          Runs one decision cycle, start to finish
│   ├── market_data.py   Provides prices (simulated for now)
│   ├── strategy.py      Suggests BUY / SELL / HOLD with a reason
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
    └── test_no_brokerage_access.py      Proves the code cannot reach a broker
```

## How the modules work together

```
market_data -> strategy -> risk_manager -> paper_trader -> journal
 (prices)     (suggest)    (allow/block)    (simulate)     (record)
```

- **`config/settings.py`** - The rulebook. Starting capital, weekly deposit,
  watchlist, and risk limits (max risk per trade, max daily loss, max trades per
  day, stop-loss, take-profit). `PAPER_TRADING = True` is the safety switch.
- **`src/market_data.py`** - Supplies prices. Right now it generates random
  prices so everything works offline.
- **`src/strategy.py`** - A simple placeholder strategy: buy when the price is
  above its recent average. It only *suggests* trades.
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
- **`src/journal.py`** - Appends each decision and transaction to
  `logs/trade_journal.csv` (with realized P&L and cash afterwards) so you can
  review what the agent did and why.
- **`src/main.py`** - Ties the steps together and runs one cycle.

## Running it (later)

The skeleton uses only the Python standard library. From the project root:

```
python -m src.main
```

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
