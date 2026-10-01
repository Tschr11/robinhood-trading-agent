"""
paper_trader.py - The simulated (paper) brokerage account.

What it does:
    - starts with $25 of pretend cash
    - simulates BUY and SELL orders
    - keeps the books: cash, open positions, entry prices, profit and loss (P&L)
    - asks the Risk Manager to approve every BUY before it happens
    - automatically closes positions that hit their stop-loss or
      take-profit price, using simulated prices you supply (check_exits)
    - writes every transaction (and every rejection) to the trading journal

What it does NOT do:
    - It never connects to Robinhood, any brokerage, or the internet.
      There is no code here that can send an order anywhere. Everything
      happens in this Python object's memory, and no real money moves.

Two kinds of profit and loss:
    realized P&L   = profit or loss "locked in" by selling
                     (shares sold x (sell price - entry price))
    unrealized P&L = profit or loss on paper for positions still held
                     (shares held x (current price - entry price))

Simulated fills happen at EXACTLY the price supplied. Real markets are
worse: prices can gap past a stop-loss, and the spread and slippage mean a
real order often fills at a less favorable price. See README.md.
"""

from dataclasses import dataclass
from datetime import date

from config import settings
from src import journal
from src.risk_manager import (TOLERANCE, AccountState, RiskManager,
                              TradeRequest, is_real_number)


# --- Small data holders -------------------------------------------------------

@dataclass
class Position:
    """Shares of one symbol that we currently hold."""
    symbol: str
    shares: float
    entry_price: float        # average price paid per share
    stop_loss_price: float    # sell automatically at or below this price
    take_profit_price: float  # sell automatically at or above this price

    @property
    def cost_basis(self) -> float:
        """What we paid in total for the shares we still hold."""
        return self.shares * self.entry_price

    def unrealized_pnl(self, market_price: float) -> float:
        """Profit (+) or loss (-) if we sold everything at `market_price`."""
        return self.shares * (market_price - self.entry_price)


@dataclass
class TradeResult:
    """What happened when we asked for a buy or sell."""
    success: bool
    action: str            # e.g. "BUY", "SELL", "BUY REJECTED"
    symbol: object
    shares: object
    price: object
    reason: str
    realized_pnl: float = 0.0


# --- The paper trading engine ------------------------------------------------

class PaperTrader:
    """
    A pretend trading account.

    Example:
        trader = PaperTrader()                         # $25 of pretend cash
        trader.buy("SPY", 0.04, 500.00, stop_loss_price=495.00)
        trader.unrealized_pnl({"SPY": 502.00})         # -> 0.08
        trader.sell("SPY", 0.04, 502.00)               # locks in +$0.08
    """

    def __init__(self, starting_cash: float = settings.STARTING_CAPITAL,
                 risk_manager: RiskManager | None = None,
                 journal_file: str = settings.JOURNAL_FILE,
                 today=date.today):
        if not is_real_number(starting_cash) or starting_cash < 0:
            raise ValueError(f"Starting cash must be $0 or more (got {starting_cash!r}).")

        self.cash = float(starting_cash)
        self.positions: dict[str, Position] = {}   # symbol -> Position
        self.realized_pnl = 0.0                    # all-time locked-in P&L

        self.risk_manager = risk_manager or RiskManager()
        self.journal_file = journal_file

        # Daily tracking. `today` is a function that returns today's date;
        # tests can swap in a fake one to pretend a new day has started.
        self._today = today
        self.current_day = today()
        self.realized_pnl_today = 0.0   # net P&L locked in today
        self.realized_loss_today = 0.0  # sum of today's LOSING sales only

    # -- Daily bookkeeping ----------------------------------------------------

    def _start_new_day_if_needed(self) -> None:
        """When the date changes, today's loss counters reset to zero."""
        now = self._today()
        if now != self.current_day:
            self.current_day = now
            self.realized_pnl_today = 0.0
            self.realized_loss_today = 0.0

    # -- Information for the Risk Manager --------------------------------------

    def account_state(self) -> AccountState:
        """
        A snapshot of this account in the format the Risk Manager expects.

        Equity here uses what we PAID for open positions (cost basis), because
        the paper trader has no live prices. That keeps the numbers predictable.
        """
        self._start_new_day_if_needed()
        holdings = sum(p.cost_basis for p in self.positions.values())
        return AccountState(
            cash=self.cash,
            equity=self.cash + holdings,
            open_positions=len(self.positions),
            realized_loss_today=self.realized_loss_today,
            paper_trading=settings.PAPER_TRADING,
        )

    # -- Buying ----------------------------------------------------------------

    def buy(self, symbol: str, shares: float, price: float,
            stop_loss_price: float, reason: str = "",
            take_profit_price: float | None = None) -> TradeResult:
        """
        Simulate buying `shares` of `symbol` at `price`.
        The Risk Manager must approve first; nothing changes if it says no.

        `take_profit_price` is optional. If left out, it is set
        TAKE_PROFIT_PCT above the entry price (2%: $500 -> $510).
        """
        symbol = self._clean_symbol(symbol)

        # Step 1: ask the Risk Manager. It checks the values (no NaN, no
        # negatives, valid stop-loss), cash, position limit, per-trade risk,
        # the daily loss limit, and that we are in paper-trading mode.
        trade = TradeRequest(symbol, shares, price, stop_loss_price)
        decision = self.risk_manager.evaluate(trade, self.account_state())
        if not decision.approved:
            return self._reject("BUY", symbol, shares, price, decision.explain())

        # Step 2: a second, independent cash check. Even if a future change
        # breaks the Risk Manager, the account can never spend money it lacks.
        cost = shares * price
        if cost > self.cash + TOLERANCE:
            return self._reject("BUY", symbol, shares, price,
                                f"Rejected: Trade costs ${cost:.2f} but only "
                                f"${self.cash:.2f} cash is available.")

        # Step 3: work out the take-profit price. Buying more of something
        # we already hold makes the entry price the AVERAGE paid for all
        # shares, so the take-profit must sit above that average.
        held = self.positions.get(symbol)
        if held is None:
            new_entry = price
        else:
            new_entry = (held.cost_basis + cost) / (held.shares + shares)
        if take_profit_price is None:
            take_profit_price = new_entry * (1 + settings.TAKE_PROFIT_PCT)
        elif (not is_real_number(take_profit_price)
              or take_profit_price <= max(price, new_entry)):
            return self._reject("BUY", symbol, shares, price,
                                "Rejected: Take-profit price must be a number "
                                f"above the entry price ${max(price, new_entry):.2f} "
                                f"(got {take_profit_price!r}).")

        # Step 4: update the books.
        self.cash = max(0.0, self.cash - cost)  # max() hides tiny rounding dust
        if held is None:
            self.positions[symbol] = Position(symbol, shares, price,
                                              stop_loss_price, take_profit_price)
        else:
            held.shares += shares
            held.entry_price = new_entry
            held.stop_loss_price = stop_loss_price
            held.take_profit_price = take_profit_price

        message = (f"Simulated buy of {shares} {symbol} at ${price:.2f} "
                   f"(stop-loss ${stop_loss_price:.2f}, take-profit "
                   f"${take_profit_price:.2f}). {reason}").strip()
        self._record("BUY", symbol, shares, price, message)
        return TradeResult(True, "BUY", symbol, shares, price, message)

    # -- Selling ---------------------------------------------------------------

    def sell(self, symbol: str, shares: float, price: float,
             reason: str = "") -> TradeResult:
        """
        Simulate selling `shares` of `symbol` at `price`.
        Selling reduces risk, so it does not need Risk Manager approval,
        but it must be valid and we must actually own the shares.
        """
        self._start_new_day_if_needed()
        symbol = self._clean_symbol(symbol)

        problems = []
        if settings.PAPER_TRADING is not True:
            problems.append("Only paper trading is allowed.")
        if not isinstance(symbol, str) or not symbol:
            problems.append("Symbol must be a ticker such as 'SPY'.")
        if not is_real_number(shares) or shares <= 0:
            problems.append(f"Share quantity must be a number greater than zero (got {shares!r}).")
        if not is_real_number(price) or price <= 0:
            problems.append(f"Price must be a number greater than zero (got {price!r}).")
        if problems:
            return self._reject("SELL", symbol, shares, price,
                                "Rejected: " + " ".join(problems))

        held = self.positions.get(symbol)
        if held is None:
            return self._reject("SELL", symbol, shares, price,
                                f"Rejected: You don't hold any {symbol} to sell.")
        if shares > held.shares + TOLERANCE:
            return self._reject("SELL", symbol, shares, price,
                                f"Rejected: Tried to sell {shares} {symbol} but "
                                f"only {held.shares:g} are held.")

        return self._fill_sell(held, shares, price, "SELL", reason)

    def _fill_sell(self, held: Position, shares: float, price: float,
                   action: str, reason: str) -> TradeResult:
        """
        The one place where a sale actually updates the books. Used by both
        manual sells and automatic stop-loss / take-profit exits.
        Callers must already have checked that the inputs are valid.
        """
        symbol = held.symbol
        # Never sell more than we own (this also absorbs rounding dust).
        shares = min(shares, held.shares)

        # Update the books.
        pnl = shares * (price - held.entry_price)
        self.cash += shares * price
        held.shares -= shares
        if held.shares <= TOLERANCE:          # sold everything -> close it
            del self.positions[symbol]

        self.realized_pnl += pnl
        self.realized_pnl_today += pnl
        if pnl < 0:
            # Losses add up even if a later trade wins. This is the stricter
            # (safer) choice: a $1 win does not "erase" a $1 loss for the
            # purpose of the daily loss limit.
            self.realized_loss_today += -pnl

        message = (f"Simulated sell of {shares:g} {symbol} at ${price:.2f}, "
                   f"realized P&L ${pnl:+.2f}. {reason}").strip()
        self._record(action, symbol, shares, price, message, realized_pnl=pnl,
                     signal="SELL")
        return TradeResult(True, action, symbol, shares, price, message, pnl)

    def close_position(self, symbol: str, price: float, reason: str = "") -> TradeResult:
        """Sell every share we hold of `symbol`."""
        symbol = self._clean_symbol(symbol)
        held = self.positions.get(symbol) if isinstance(symbol, str) else None
        if held is None:
            return self._reject("SELL", symbol, 0, price,
                                f"Rejected: You don't hold any {symbol} to sell.")
        return self.sell(symbol, held.shares, price, reason)

    # -- Automatic exits: stop-loss and take-profit ---------------------------

    def check_exits(self, market_prices: dict[str, float]) -> list[TradeResult]:
        """
        Compare each open position with its current simulated price, e.g.
        check_exits({"SPY": 494.00}), and close it automatically if:

            price <= stop-loss    -> "STOP-LOSS SELL"   (cut the loss)
            price >= take-profit  -> "TAKE-PROFIT SELL" (lock in the gain)

        Anything in between: do nothing.

        Exits do NOT ask the Risk Manager, so they still work after the daily
        loss limit is reached - we must always be able to get OUT of a trade.

        If a price is missing or invalid, that position is SKIPPED (never
        guessed) and the skip is written to the journal, because it means the
        position is unprotected until a good price arrives.

        Returns a list describing every exit and every skip.
        """
        self._start_new_day_if_needed()
        results = []

        if settings.PAPER_TRADING is not True:
            return [self._skip("ALL", "Only paper trading is allowed.")]
        if not self.positions:
            return results
        if not isinstance(market_prices, dict):
            return [self._skip("ALL", "Market prices must be a dict like "
                                      f"{{'SPY': 500.00}} (got {market_prices!r}).")]

        # Accept " spy " as well as "SPY" for the keys.
        prices = {self._clean_symbol(k): v for k, v in market_prices.items()
                  if isinstance(k, str)}

        # list(...) makes a copy, because closing a position removes it
        # from self.positions while we are looping.
        for held in list(self.positions.values()):
            price = prices.get(held.symbol)
            if not is_real_number(price) or price <= 0:
                results.append(self._skip(
                    held.symbol, f"No valid price for {held.symbol} "
                                 f"(got {price!r}); stop-loss and take-profit "
                                 "were not checked."))
            elif price <= held.stop_loss_price + TOLERANCE:
                results.append(self._fill_sell(
                    held, held.shares, price, "STOP-LOSS SELL",
                    f"Stop-loss hit: price ${price:.2f} is at or below the "
                    f"stop of ${held.stop_loss_price:.2f}."))
            elif price >= held.take_profit_price - TOLERANCE:
                results.append(self._fill_sell(
                    held, held.shares, price, "TAKE-PROFIT SELL",
                    f"Take-profit hit: price ${price:.2f} is at or above the "
                    f"target of ${held.take_profit_price:.2f}."))
        return results

    def _skip(self, symbol, reason: str) -> TradeResult:
        """Journal an exit check that could not be done. Books unchanged."""
        self._record("EXIT CHECK SKIPPED", symbol, "", "", reason, signal="CHECK")
        return TradeResult(False, "EXIT CHECK SKIPPED", symbol, "", "", reason)

    # -- Deposits --------------------------------------------------------------

    def deposit(self, amount: float = settings.WEEKLY_CONTRIBUTION,
                reason: str = "Weekly contribution") -> TradeResult:
        """Add pretend cash, e.g. the $25 weekly contribution."""
        if not is_real_number(amount) or amount <= 0:
            return self._reject("DEPOSIT", "", "", amount,
                                f"Rejected: Deposit must be more than $0 (got {amount!r}).")
        self.cash += amount
        self._record("DEPOSIT", "", "", amount, f"{reason}: +${amount:.2f}")
        return TradeResult(True, "DEPOSIT", "", "", amount, reason)

    # -- Profit and loss with supplied market prices --------------------------

    def unrealized_pnl(self, market_prices: dict[str, float]) -> float:
        """
        Total paper profit/loss on open positions.
        `market_prices` is a dict like {"SPY": 502.00} supplied by the caller
        (simulated prices for now - this module never fetches prices itself).
        """
        return sum(held.unrealized_pnl(self._market_price(held.symbol, market_prices))
                   for held in self.positions.values())

    def total_equity(self, market_prices: dict[str, float]) -> float:
        """Cash plus what open positions are worth at `market_prices`."""
        return self.cash + sum(
            held.shares * self._market_price(held.symbol, market_prices)
            for held in self.positions.values())

    @staticmethod
    def _market_price(symbol: str, market_prices: dict[str, float]) -> float:
        """Look up a price, refusing to guess if it is missing or invalid."""
        if not isinstance(market_prices, dict):
            raise ValueError("Market prices must be a dict like {'SPY': 500.00}.")
        price = market_prices.get(symbol)
        if not is_real_number(price) or price <= 0:
            raise ValueError(f"Need a valid market price for {symbol} (got {price!r}).")
        return price

    # -- Helpers ----------------------------------------------------------------

    @staticmethod
    def _clean_symbol(symbol):
        """' spy ' -> 'SPY'. Non-text values are passed through to be rejected."""
        return symbol.strip().upper() if isinstance(symbol, str) else symbol

    def _reject(self, side: str, symbol, shares, price, reason: str) -> TradeResult:
        """Record a refused order in the journal and report it. Books unchanged."""
        action = f"{side} REJECTED"
        self._record(action, symbol, shares, price, reason, signal=side)
        return TradeResult(False, action, symbol, shares, price, reason)

    def _record(self, action: str, symbol, shares, price, reason: str,
                realized_pnl: float | str = "", signal: str | None = None) -> None:
        """Write one row to the trading journal."""
        journal.log_decision(
            symbol=symbol, signal=signal or action, action=action,
            shares=shares, price=price, reason=reason,
            realized_pnl=round(realized_pnl, 4) if realized_pnl != "" else "",
            cash_after=round(self.cash, 4), path=self.journal_file)
