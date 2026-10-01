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
├── data/                Saved price data (future)
├── logs/                Trade journal and run logs
├── src/
│   ├── main.py          Runs one decision cycle, start to finish
│   ├── market_data.py   Provides prices (simulated for now)
│   ├── strategy.py      Suggests BUY / SELL / HOLD with a reason
│   ├── risk_manager.py  Approves or blocks each trade against the rules
│   ├── paper_trader.py  Simulated account: buys, sells, P&L, daily losses
│   └── journal.py       Records every decision to a CSV file
└── tests/
    ├── test_risk_manager.py             Approved and rejected trades
    ├── test_risk_manager_safeguards.py  Each safeguard, including bad data
    ├── test_paper_trader.py             Buys, sells, P&L, daily losses, journal
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
  pretend account that starts with $25 and lives only in memory:
  - `buy()` asks the Risk Manager first; a rejected buy changes nothing
  - `sell()` / `close_position()` lock in realized P&L and refuse to sell
    shares you don't hold
  - tracks cash, positions (shares, average entry price, stop-loss),
    realized P&L, and today's losses (reset each new day)
  - `unrealized_pnl(prices)` and `total_equity(prices)` use prices you supply
  - `account_state()` hands the real numbers to the Risk Manager
  - every buy, sell, deposit and rejection is written to the journal
- **`src/journal.py`** - Appends each decision and transaction to
  `logs/trade_journal.csv` (with realized P&L and cash afterwards) so you can
  review what the agent did and why.
- **`src/main.py`** - Ties the steps together and runs one cycle.

## Running it (later)

The skeleton uses only the Python standard library. From the project root:

```
python -m src.main
```

## Running the tests

The tests use Python's built-in `unittest`, so nothing needs installing:

```
python -m unittest discover tests -v
```

## Important notes for a small account

- With $25, fractional shares are required for most stocks and ETFs.
- Day-trading rules for small accounts (such as pattern day trader limits and
  cash-settlement rules) depend on the account type and current regulations.
  These should be researched and encoded in `risk_manager.py` before the agent's
  behavior is trusted.
- Simulated results do not include slippage, spreads, or emotions, so real
  results would differ.

This project is for education and research. It is not financial advice.
