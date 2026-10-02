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

# --- Strategy rules (see src/strategy.py and README.md) ----------------------
# A deterministic trend-following strategy. These numbers ARE the strategy:
# changing them changes the signals. They are a starting point for paper
# research, not a recommendation, and nothing here implies profitability.
STRATEGY_NAME = "trend_vwap_v1"
# Entry (BUY) - every rule must pass, and no position may be open:
#   E1  SMA 20 > SMA 50                     (short-term trend above long-term)
#   E2  close > VWAP                        (price above today's average paid)
#   E3  close > SMA 20                      (price above its recent average)
#   E4  RSI_ENTRY_MIN <= RSI 14 <= RSI_ENTRY_MAX   (momentum, not overbought)
#   E5  latest volume >= average volume x VOLUME_MULTIPLIER (real interest)
RSI_ENTRY_MIN = 50.0
RSI_ENTRY_MAX = 70.0
VOLUME_MULTIPLIER = 1.0
# Exit (SELL) - any one rule is enough, and a position must be open:
#   X1  SMA 20 < SMA 50                     (trend has turned down)
#   X2  close < VWAP                        (price fell below today's average)
#   X3  RSI 14 >= RSI_EXIT                  (overbought - lock in the move)
# (Stop-loss and take-profit exits are handled separately by the paper trader.)
RSI_EXIT = 75.0
# Live data older than this is treated as unusable (HOLD, never trade).
LIVE_DATA_MAX_AGE_SECONDS = 120

# --- Backtesting (see src/backtest.py and README.md) -------------------------
# Costs used when replaying history. Real costs vary; set these to match your
# own expectations. Commission is per order (entry and exit each pay it).
BACKTEST_COMMISSION_PER_TRADE = 0.00   # dollars per order
BACKTEST_SLIPPAGE_PCT = 0.0005         # 0.05% worse fill on market orders
BACKTEST_LOOKBACK_CANDLES = 500        # candles shown to the strategy each step

# --- Files -------------------------------------------------------------------
DATA_DIR = "data"
# Historical candles, one CSV per symbol: data/market/SPY.csv
# (columns: timestamp,open,high,low,close,volume). You supply these files;
# the agent never invents prices.
MARKET_DATA_DIR = "data/market"
# Imported historical market data (raw source files, canonical CSVs and their
# manifests) lives in its own top-level folder, apart from the paper account
# (data/), the journal (logs/) and evaluation reports (reports/). Git-ignored.
HISTORICAL_DATA_DIR = "historical_data"
# Exchange calendar for regular trading hours (America/New_York). Every year
# in it is versioned and must be marked verified before it can be used.
MARKET_CALENDAR_FILE = "config/market_calendar.json"
LOG_DIR = "logs"
# Evaluation reports (src/evaluation.py) - kept apart from the paper account
# in data/ and the journal in logs/.
REPORTS_DIR = "reports"
# Append-only record of SAVED out-of-sample evaluations, used to count how
# often each dataset's out-of-sample period has been looked at. It does not
# depend on the plan name or the report folder.
OOS_EXPOSURE_LOG = "reports/oos_exposure_log.jsonl"
JOURNAL_FILE = "logs/trade_journal.csv"
# The saved paper account (cash, positions, P&L). Delete this file to start
# over with a fresh $25 account. It holds no credentials - only pretend money.
DATABASE_FILE = "data/paper_account.db"
