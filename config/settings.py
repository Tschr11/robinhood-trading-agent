"""
Project settings for the Robinhood Trading Agent.

Every rule the agent follows lives here, in one place, so it is easy to
read and change. There are NO API keys, passwords, or brokerage settings
in this file - and there never should be.
"""

# --- Safety switch -----------------------------------------------------------
# The agent only ever simulates trades. There is no live-trading code path.
PAPER_TRADING = True

# --- Account goals -----------------------------------------------------------
STARTING_CAPITAL = 25.00       # dollars we start with
WEEKLY_CONTRIBUTION = 25.00    # dollars added every week
ACCOUNT_MILESTONE = 1_000.00   # long-term goal for the account

# --- Trading style -----------------------------------------------------------
TRADING_STYLE = "day_trading"  # positions are opened and closed the same day
WATCHLIST = ["SPY", "QQQ"]     # symbols the agent is allowed to look at

# --- Risk rules (the agent must obey these before every trade) ---------------
# Percentages are written as decimals: 0.02 means 2%.
MAX_RISK_PER_TRADE_PCT = 0.02  # lose at most 2% of the account if a stop-loss hits
MAX_DAILY_LOSS_PCT = 0.05      # stop trading for the day after losing 5%
MAX_OPEN_POSITIONS = 1         # hold only one position at a time (for now)
MAX_TRADES_PER_DAY = 3         # keep the number of trades small
STOP_LOSS_PCT = 0.01           # exit a trade if it falls 1% below entry
TAKE_PROFIT_PCT = 0.02         # exit a trade if it rises 2% above entry
ALLOW_FRACTIONAL_SHARES = True # needed to trade with a small balance

# --- Files -------------------------------------------------------------------
DATA_DIR = "data"
# Historical candles, one CSV per symbol: data/market/SPY.csv
# (columns: timestamp,open,high,low,close,volume). You supply these files;
# the agent never invents prices.
MARKET_DATA_DIR = "data/market"
LOG_DIR = "logs"
JOURNAL_FILE = "logs/trade_journal.csv"
# The saved paper account (cash, positions, P&L). Delete this file to start
# over with a fresh $25 account. It holds no credentials - only pretend money.
DATABASE_FILE = "data/paper_account.db"
