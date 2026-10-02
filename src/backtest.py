"""
backtest.py - Replays HISTORICAL candles to see how the strategy's rules
would have behaved. This is a BACKTEST: a simulation on past data.

It is completely separate from paper trading:
    - it keeps its own pretend cash and position in memory
    - it never imports or touches the paper trader, its SQLite database,
      or the trading journal
    - it only accepts data labelled HISTORICAL, and never connects anywhere

How each candle (bar) is processed, in time order:

    1. OPEN   fill any order decided at the PREVIOUS candle's close,
              at this candle's open price (plus slippage)
    2. RANGE  if holding, check the stop-loss / take-profit against this
              candle's low and high
    3. CLOSE  ask the strategy what to do, showing it ONLY the candles up to
              and including this one. A BUY/SELL becomes an order for the
              NEXT candle's open - never this candle's close.
    4. If this was the day's last candle (day trading): exit at the close
       and cancel pending orders, so nothing is held overnight.
    5. Record equity (cash + shares x close) for the drawdown calculation.

This ordering is what prevents "look-ahead bias": no decision can use a
price that would not have been known at that moment.

History before the test period (trade_start):
    Candles before `trade_start` are HISTORY ONLY. They are shown to the
    strategy so indicators such as RSI start from a realistic value, but on
    those candles the strategy is never asked, nothing is bought or sold, and
    no equity is recorded. Trading decisions begin at the close of the first
    candle on or after `trade_start`.

IMPORTANT: a backtest shows how fixed rules interacted with ONE stretch of
past data. It is not a prediction, and a good result in one backtest is
not evidence that the strategy will make money.

Run it on your CSV files with:
    python -m src.backtest
"""

import math
from dataclasses import dataclass, field
from datetime import date, datetime

from config import settings
from src import strategy
from src.market_data import (MIN_CANDLES_FOR_INDICATORS, Candle,
                             CSVHistoricalProvider, DataKind,
                             InsufficientDataError, MarketDataError,
                             MarketDataSet)
from src.risk_manager import AccountState, RiskManager, TradeRequest
from src.strategy import Signal, StrategyConfig

LABEL = "BACKTEST - historical simulation (not live trading, not the paper account)"
DISCLAIMER = ("A single backtest only shows how these rules behaved on this "
              "particular stretch of past data. It is NOT evidence that the "
              "strategy will be profitable in the future.")

# Exit reasons
STOP_LOSS = "STOP_LOSS"
TAKE_PROFIT = "TAKE_PROFIT"
STRATEGY_EXIT = "STRATEGY_EXIT"
SESSION_END = "SESSION_END"
END_OF_DATA = "END_OF_DATA"


class BacktestError(ValueError):
    """Raised when the backtest is set up incorrectly."""


def _is_real(value) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


# --- Configuration --------------------------------------------------------------

@dataclass(frozen=True)
class BacktestConfig:
    """Every assumption the backtest makes. Defaults come from settings.py."""
    starting_capital: float = settings.STARTING_CAPITAL
    risk_per_trade_pct: float = settings.MAX_RISK_PER_TRADE_PCT   # sizing
    max_position_pct: float = 1.0          # most of the cash one position may use
    daily_loss_pct: float = settings.MAX_DAILY_LOSS_PCT
    stop_loss_pct: float = settings.STOP_LOSS_PCT
    take_profit_pct: float = settings.TAKE_PROFIT_PCT
    commission_per_trade: float = settings.BACKTEST_COMMISSION_PER_TRADE
    slippage_pct: float = settings.BACKTEST_SLIPPAGE_PCT
    flatten_end_of_day: bool = True        # day trading: no overnight positions
    allow_fractional_shares: bool = settings.ALLOW_FRACTIONAL_SHARES
    warmup_candles: int = MIN_CANDLES_FOR_INDICATORS   # history before 1st decision
    lookback_candles: int = settings.BACKTEST_LOOKBACK_CANDLES

    def __post_init__(self):
        problems = []

        def number(label, value, low, high, low_ok=False, high_ok=True):
            if not _is_real(value):
                problems.append(f"{label} must be a real number (got {value!r}).")
                return
            above = value >= low if low_ok else value > low
            below = value <= high if high_ok else value < high
            if not (above and below):
                problems.append(f"{label} must be {'>=' if low_ok else '>'} {low} "
                                f"and {'<=' if high_ok else '<'} {high} (got {value}).")

        number("starting_capital", self.starting_capital, 0, math.inf, high_ok=False)
        number("risk_per_trade_pct", self.risk_per_trade_pct, 0, 1)
        number("max_position_pct", self.max_position_pct, 0, 1)
        number("daily_loss_pct", self.daily_loss_pct, 0, 1)
        number("stop_loss_pct", self.stop_loss_pct, 0, 1, high_ok=False)
        number("take_profit_pct", self.take_profit_pct, 0, math.inf, high_ok=False)
        number("commission_per_trade", self.commission_per_trade, 0, math.inf,
               low_ok=True, high_ok=False)
        number("slippage_pct", self.slippage_pct, 0, 0.1, low_ok=True, high_ok=False)
        for label in ["flatten_end_of_day", "allow_fractional_shares"]:
            if not isinstance(getattr(self, label), bool):
                problems.append(f"{label} must be True or False.")
        for label in ["warmup_candles", "lookback_candles"]:
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int):
                problems.append(f"{label} must be a whole number (got {value!r}).")
            elif value < MIN_CANDLES_FOR_INDICATORS:
                problems.append(
                    f"{label} must be at least {MIN_CANDLES_FOR_INDICATORS} (got {value}): "
                    f"the strategy's SMA {MIN_CANDLES_FOR_INDICATORS} needs that many "
                    "candles, so a smaller value would only produce HOLD signals.")
        if not problems and self.lookback_candles < self.warmup_candles:
            problems.append("lookback_candles must be at least warmup_candles.")
        if problems:
            raise BacktestError("Invalid backtest settings: " + " ".join(problems))


DEFAULT_CONFIG = BacktestConfig()


# --- Results ------------------------------------------------------------------------

@dataclass(frozen=True)
class BacktestTrade:
    """One completed simulated round trip (buy, then sell)."""
    symbol: str
    signal_time: datetime    # close of the candle where the BUY was decided
    entry_time: datetime
    entry_price: float       # actual simulated fill, including slippage
    exit_time: datetime
    exit_price: float        # actual simulated fill, including slippage
    shares: float
    commission: float        # entry + exit
    realized_pnl: float      # after slippage and commission
    exit_reason: str
    slippage_cost: float = 0.0   # dollars lost to slippage (entry + exit)

    @property
    def return_pct(self) -> float:
        return self.realized_pnl / (self.shares * self.entry_price)


@dataclass(frozen=True)
class BacktestMetrics:
    """
    Summary numbers. A value of None means "undefined" (for example, there
    were no losing trades, so profit factor can't be calculated) - it is
    never replaced with a made-up number.
    """
    starting_capital: float
    final_equity: float
    total_return_pct: float          # 0.10 = +10%
    number_of_trades: int
    wins: int
    losses: int
    win_rate: float | None           # wins / trades
    average_win: float | None        # dollars
    average_loss: float | None       # dollars (negative)
    profit_factor: float | None      # total won / total lost
    # The largest fall in DOLLARS, and that SAME fall as a % of its own peak.
    max_drawdown: float
    max_drawdown_pct: float          # 0.25 = 25% below that fall's peak
    total_commission: float
    total_slippage: float = 0.0      # dollars lost to slippage
    # The largest fall in PERCENT, and that SAME fall in dollars. This can be
    # a different fall from max_drawdown (e.g. an early -40% on a small
    # balance vs. a later, bigger-dollar but smaller-% drop).
    largest_pct_drawdown: float = 0.0
    largest_pct_drawdown_dollars: float = 0.0


def drawdowns(equity_values, starting_capital: float) -> tuple[float, float, float, float]:
    """
    Peak-to-trough falls in the equity curve, measured from the starting
    capital. Returns (largest $ fall, its %, largest % fall, its $), where
    each pair describes ONE fall from one peak to one trough.
    """
    peak = starting_capital
    max_dd, max_dd_pct, top_pct, top_pct_dollars = 0.0, 0.0, 0.0, 0.0
    for value in equity_values:
        peak = max(peak, value)
        drop = peak - value
        drop_pct = drop / peak if peak > 0 else 0.0
        if drop > max_dd:
            max_dd, max_dd_pct = drop, drop_pct
        if drop_pct > top_pct:
            top_pct, top_pct_dollars = drop_pct, drop
    return max_dd, max_dd_pct, top_pct, top_pct_dollars


def compute_metrics(trades, equity_values, starting_capital: float) -> BacktestMetrics:
    """Calculate the summary metrics from trades and the equity curve."""
    pnls = [t.realized_pnl for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]          # exactly $0 counts as neither
    final_equity = equity_values[-1] if equity_values else starting_capital

    max_dd, max_dd_pct, top_pct, top_pct_dollars = drawdowns(equity_values, starting_capital)

    return BacktestMetrics(
        starting_capital=starting_capital,
        final_equity=final_equity,
        total_return_pct=(final_equity - starting_capital) / starting_capital,
        number_of_trades=len(pnls),
        wins=len(winners),
        losses=len(losers),
        win_rate=len(winners) / len(pnls) if pnls else None,
        average_win=sum(winners) / len(winners) if winners else None,
        average_loss=sum(losers) / len(losers) if losers else None,
        profit_factor=sum(winners) / -sum(losers) if losers else None,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        total_commission=sum(t.commission for t in trades),
        total_slippage=sum(t.slippage_cost for t in trades),
        largest_pct_drawdown=top_pct,
        largest_pct_drawdown_dollars=top_pct_dollars,
    )


@dataclass(frozen=True)
class BacktestResult:
    label: str
    symbol: str
    source: str
    strategy: str
    config: BacktestConfig
    first_candle: datetime
    last_candle: datetime
    candles: int
    trades: tuple[BacktestTrade, ...]
    equity_curve: tuple[tuple[datetime, float], ...]
    metrics: BacktestMetrics
    rejected_entries: tuple[tuple[datetime, str], ...]   # (time, why)
    expired_orders: tuple[tuple[datetime, str, str], ...]  # (time, signal, why)
    disclaimer: str = DISCLAIMER

    def report(self) -> str:
        m, c = self.metrics, self.config

        def money(v):
            return "n/a" if v is None else f"${v:,.2f}"

        def pct(v):
            return "n/a" if v is None else f"{v * 100:.2f}%"

        lines = [
            LABEL,
            f"Symbol {self.symbol} | strategy {self.strategy} | data {self.source}",
            f"Period {self.first_candle.isoformat()} to {self.last_candle.isoformat()} "
            f"({self.candles} historical candles)",
            f"Assumptions: start {money(c.starting_capital)}, risk "
            f"{pct(c.risk_per_trade_pct)}/trade, max position {pct(c.max_position_pct)} "
            f"of cash, stop {pct(c.stop_loss_pct)}, target {pct(c.take_profit_pct)}, "
            f"commission {money(c.commission_per_trade)}/order, slippage "
            f"{pct(c.slippage_pct)}, flatten end of day: {c.flatten_end_of_day}",
            "",
            f"  Final equity      {money(m.final_equity)}",
            f"  Total return      {pct(m.total_return_pct)}",
            f"  Trades            {m.number_of_trades} ({m.wins} won, {m.losses} lost)",
            f"  Win rate          {pct(m.win_rate)}",
            f"  Average win       {money(m.average_win)}",
            f"  Average loss      {money(m.average_loss)}",
            f"  Profit factor     "
            + ("n/a (no losing trades)" if m.profit_factor is None and m.number_of_trades
               else "n/a" if m.profit_factor is None else f"{m.profit_factor:.2f}"),
            f"  Max $ drawdown    {money(m.max_drawdown)} "
            f"({pct(m.max_drawdown_pct)} of that fall's peak)",
            f"  Max % drawdown    {pct(m.largest_pct_drawdown)} "
            f"({money(m.largest_pct_drawdown_dollars)})",
            f"  Commission paid   {money(m.total_commission)} (already in the return)",
            f"  Slippage cost     {money(m.total_slippage)} (already in the return)",
            f"  Entries rejected  {len(self.rejected_entries)}",
            f"  Orders expired    {len(self.expired_orders)}",
        ]
        if self.trades:
            lines += ["", "  Trades:"]
            for t in self.trades:
                lines.append(
                    f"    {t.entry_time.isoformat()} buy {t.shares:g} @ {t.entry_price:.4f}"
                    f" -> {t.exit_time.isoformat()} sell @ {t.exit_price:.4f}"
                    f"  P&L {t.realized_pnl:+.4f}  [{t.exit_reason}]")
        lines += ["", f"NOTE: {self.disclaimer}"]
        return "\n".join(lines)


# --- Running a backtest -----------------------------------------------------------------

def first_tradable_index(candles, warmup_candles: int, trade_start: datetime | None = None) -> int:
    """
    Index of the first candle on which a simulated order can FILL.

    The first decision happens at the close of candle d, where d is the later
    of (a) the first candle on or after `trade_start` and (b) the candle that
    completes `warmup_candles` of history. Orders fill at the next open, so
    the answer is d + 1. Warm-up is never treated as less than the strategy's
    minimum, even if a caller bypasses validation.

    The backtester and the buy-and-hold benchmark both use this, so they
    always start on the same candle.
    """
    warmup = max(warmup_candles, MIN_CANDLES_FOR_INDICATORS)
    first_period_candle = 0
    if trade_start is not None:
        first_period_candle = next((i for i, c in enumerate(candles)
                                    if c.timestamp >= trade_start), len(candles))
    return max(first_period_candle, warmup - 1) + 1


def run_backtest(dataset: MarketDataSet, config: BacktestConfig = DEFAULT_CONFIG,
                 strategy_config: StrategyConfig = strategy.DEFAULT_CONFIG,
                 strategy_fn=None, trade_start: datetime | None = None) -> BacktestResult:
    """
    Replay `dataset` (HISTORICAL candles, oldest first) through the strategy.

    trade_start  optional: candles before this time are indicator history
                 only - never traded, never in the equity curve or metrics.

    strategy_fn  optional replacement for strategy.evaluate - mainly for
                 tests. It is called exactly like strategy.evaluate and must
                 return an object with a `.signal` (BUY / SELL / HOLD).
    """
    if not isinstance(dataset, MarketDataSet):
        raise MarketDataError(
            f"A backtest needs a validated MarketDataSet (got {type(dataset).__name__}).")
    dataset.require_kind(DataKind.HISTORICAL)          # never backtest "live" data
    if not isinstance(config, BacktestConfig):
        raise BacktestError("config must be a BacktestConfig.")
    if trade_start is not None and (not isinstance(trade_start, datetime)
                                    or trade_start.utcoffset() is None):
        raise BacktestError("trade_start must be a datetime with a time zone.")
    if first_tradable_index(dataset.candles, config.warmup_candles,
                            trade_start) >= len(dataset):
        raise InsufficientDataError(
            f"{dataset.symbol}: a backtest needs at least {config.warmup_candles} "
            "candles of warm-up history plus at least one more candle to trade on "
            f"(in the trading period), but {dataset.source} has {len(dataset)} "
            "candles in total.")

    if strategy_fn is None:
        def strategy_fn(view, **kwargs):
            return strategy.evaluate(view, config=strategy_config, **kwargs)
        name = strategy_config.name
    else:
        name = getattr(strategy_fn, "name", "custom")

    return _Simulation(dataset, config, strategy_fn, name, trade_start).run()


@dataclass
class _Position:
    shares: float
    entry_price: float
    entry_time: datetime
    signal_time: datetime
    stop_loss_price: float
    take_profit_price: float
    entry_commission: float
    entry_slippage: float = 0.0


@dataclass
class _Simulation:
    """The backtest's own private books. Nothing here is saved anywhere."""
    dataset: MarketDataSet
    config: BacktestConfig
    strategy_fn: object
    strategy_name: str
    trade_start: datetime | None = None
    cash: float = 0.0
    position: _Position | None = None
    pending: tuple | None = None              # (Signal, decided_at)
    day: date | None = None
    loss_today: float = 0.0
    trades: list = field(default_factory=list)
    equity: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    expired: list = field(default_factory=list)

    def __post_init__(self):
        self.cash = float(self.config.starting_capital)
        self.risk = RiskManager(max_risk_per_trade_pct=self.config.risk_per_trade_pct,
                                max_daily_loss_pct=self.config.daily_loss_pct,
                                max_open_positions=1)

    # -- Main loop ---------------------------------------------------------------

    def run(self) -> BacktestResult:
        candles = self.dataset.candles
        last = len(candles) - 1
        trading = []                                     # candles inside the period
        for i, bar in enumerate(candles):
            if self.trade_start is not None and bar.timestamp < self.trade_start:
                continue        # history only: no decisions, fills or equity
            trading.append(bar)
            if bar.timestamp.date() != self.day:          # a new trading day
                self.day = bar.timestamp.date()
                self.loss_today = 0.0

            # 1. OPEN: fill the order decided at the previous close.
            if self.pending:
                self._fill_pending(bar)

            # 2. RANGE: stop-loss / take-profit during this candle.
            if self.position:
                self._check_stop_and_target(bar)

            # 3. CLOSE: decide, using ONLY candles[0 .. i].
            if i + 1 >= self.config.warmup_candles:
                self._decide(candles, i)

            # 4. End of data / end of day.
            if i == last:
                if self.position:
                    self._exit(bar.timestamp, bar.close, END_OF_DATA)
                self._expire_pending("no later candle to fill it")
            elif self.config.flatten_end_of_day and \
                    candles[i + 1].timestamp.date() != self.day:
                # Only the next candle's DATE is used (the market's closing
                # time is known in advance) - never its prices.
                if self.position:
                    self._exit(bar.timestamp, bar.close, SESSION_END)
                self._expire_pending("day ended; no overnight orders")

            # 5. Equity at this close.
            held = self.position.shares * bar.close if self.position else 0.0
            self.equity.append((bar.timestamp, self.cash + held))

        return BacktestResult(
            label=LABEL, symbol=self.dataset.symbol, source=self.dataset.source,
            strategy=self.strategy_name, config=self.config,
            first_candle=trading[0].timestamp, last_candle=trading[-1].timestamp,
            candles=len(trading), trades=tuple(self.trades),
            equity_curve=tuple(self.equity),
            metrics=compute_metrics(self.trades, [v for _, v in self.equity],
                                    self.config.starting_capital),
            rejected_entries=tuple(self.rejected), expired_orders=tuple(self.expired))

    # -- Step 3: asking the strategy ---------------------------------------------

    def _decide(self, candles: tuple[Candle, ...], i: int) -> None:
        start = max(0, i + 1 - self.config.lookback_candles)
        # The strategy sees a dataset that ENDS at the current candle.
        view = MarketDataSet(self.dataset.symbol, DataKind.HISTORICAL,
                             self.dataset.source, candles[start:i + 1])
        result = self.strategy_fn(view, expected_kind=DataKind.HISTORICAL,
                                  has_open_position=self.position is not None)
        signal = getattr(result, "signal", None)
        if not isinstance(signal, Signal):
            raise BacktestError(f"The strategy returned {result!r} instead of a signal.")
        if signal is Signal.BUY and self.position is None:
            self.pending = (Signal.BUY, candles[i].timestamp)
        elif signal is Signal.SELL and self.position is not None:
            self.pending = (Signal.SELL, candles[i].timestamp)

    # -- Step 1: filling orders at the next open ----------------------------------

    def _fill_pending(self, bar: Candle) -> None:
        signal, decided_at = self.pending
        self.pending = None
        if signal is Signal.BUY:
            self._enter(bar, decided_at)
        elif self.position:
            self._exit(bar.timestamp, bar.open, STRATEGY_EXIT)

    def _enter(self, bar: Candle, decided_at: datetime) -> None:
        c = self.config
        fill = bar.open * (1 + c.slippage_pct)                 # pay a bit more
        stop = fill * (1 - c.stop_loss_pct)
        target = fill * (1 + c.take_profit_pct)
        # Flat when entering, so equity = cash.
        state = AccountState(cash=self.cash, equity=self.cash, open_positions=0,
                             realized_loss_today=self.loss_today, paper_trading=True)

        shares = self.risk.max_shares(state, fill, stop)       # per-trade risk
        shares = min(shares, c.max_position_pct * self.cash / fill)
        spendable = self.cash - c.commission_per_trade
        shares = min(shares, spendable / fill) if spendable > 0 else 0.0
        shares = (math.floor(shares) if not c.allow_fractional_shares
                  else math.floor(shares * 10_000 + 1e-9) / 10_000)
        if shares <= 0:
            self.rejected.append((bar.timestamp, "Position size rounds to zero "
                                  "(too little cash for the risk/size limits)."))
            return

        decision = self.risk.evaluate(TradeRequest(self.dataset.symbol, shares, fill, stop),
                                      state)
        if not decision.approved:
            self.rejected.append((bar.timestamp, decision.explain()))
            return

        self.cash -= shares * fill + c.commission_per_trade
        self.position = _Position(shares, fill, bar.timestamp, decided_at, stop, target,
                                  c.commission_per_trade,
                                  entry_slippage=shares * (fill - bar.open))

    # -- Step 2: stop-loss and take-profit ------------------------------------------

    def _check_stop_and_target(self, bar: Candle) -> None:
        p = self.position
        if bar.low <= p.stop_loss_price:
            # If the low reached both levels we can't know which came first,
            # so assume the WORSE outcome: the stop.
            # A gap below the stop fills at the (worse) open price.
            base = bar.open if bar.open <= p.stop_loss_price else p.stop_loss_price
            self._exit(bar.timestamp, base, STOP_LOSS)
        elif bar.high >= p.take_profit_price:
            # A limit order: fills at the target (or a better gap-up open),
            # with no slippage.
            base = bar.open if bar.open >= p.take_profit_price else p.take_profit_price
            self._exit(bar.timestamp, base, TAKE_PROFIT, market_order=False)

    # -- Closing a position --------------------------------------------------------------

    def _sell_fill(self, price: float) -> float:
        """A market sell receives a bit less than the quoted price."""
        return price * (1 - self.config.slippage_pct)

    def _exit(self, when: datetime, quoted: float, reason: str,
              market_order: bool = True) -> None:
        """Close the position. Market orders pay slippage; limit orders don't."""
        p, commission = self.position, self.config.commission_per_trade
        fill = self._sell_fill(quoted) if market_order else quoted
        self.cash += p.shares * fill - commission
        pnl = p.shares * (fill - p.entry_price) - p.entry_commission - commission
        if pnl < 0:
            self.loss_today += -pnl
        self.trades.append(BacktestTrade(
            symbol=self.dataset.symbol, signal_time=p.signal_time,
            entry_time=p.entry_time, entry_price=p.entry_price, exit_time=when,
            exit_price=fill, shares=p.shares,
            commission=p.entry_commission + commission, realized_pnl=pnl,
            exit_reason=reason,
            slippage_cost=p.entry_slippage + p.shares * (quoted - fill)))
        self.position = None

    def _expire_pending(self, why: str) -> None:
        if self.pending:
            signal, decided_at = self.pending
            self.expired.append((decided_at, signal.value, why))
            self.pending = None


# --- Command line ------------------------------------------------------------------------

if __name__ == "__main__":
    provider = CSVHistoricalProvider()
    for symbol in settings.WATCHLIST:
        try:
            data = provider.get_candles(symbol)
            print(run_backtest(data).report())
        except MarketDataError as error:
            print(f"{symbol}: backtest skipped - {error}")
        print()
