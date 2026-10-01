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
MAX_RISK_PER_TRADE_PCT = 0.02  # risk at most 2% of the account on one trade
MAX_POSITION_SIZE_PCT = 0.50   # never put more than 50% of cash in one position
MAX_DAILY_LOSS_PCT = 0.05      # stop trading for the day after a 5% loss
MAX_TRADES_PER_DAY = 3         # keep the number of trades small
STOP_LOSS_PCT = 0.01           # exit a trade if it falls 1% below entry
TAKE_PROFIT_PCT = 0.02         # exit a trade if it rises 2% above entry
ALLOW_FRACTIONAL_SHARES = True # needed to trade with a small balance

# --- Files -------------------------------------------------------------------
DATA_DIR = "data"
LOG_DIR = "logs"
JOURNAL_FILE = "logs/trade_journal.csv"
