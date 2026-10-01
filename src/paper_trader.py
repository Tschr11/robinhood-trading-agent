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
    - saves the account to a local SQLite file (src/storage.py), so cash,
      positions, P&L and today's losses survive a restart

What it does NOT do:
    - It never connects to Robinhood, any brokerage, or the internet.
      There is no code here that can send an order anywhere. Everything
      happens in this Python object and a local database file, and no real
      money moves.

How saving works (the database is the source of truth):
    Every buy, sell, deposit and exit check is ONE operation:
        lock the database -> reload the account -> check rules -> change the
        books -> save books + ledger row -> commit -> write the CSV journal
    If anything fails in the middle, the database rolls back and the
    account is reloaded, so memory and disk always agree.

Two kinds of profit and loss:
    realized P&L   = profit or loss "locked in" by selling
                     (shares sold x (sell price - entry price))
    unrealized P&L = profit or loss on paper for positions still held
                     (shares held x (current price - entry price))

Simulated fills happen at EXACTLY the price supplied. Real markets are
worse: prices can gap past a stop-loss, and the spread and slippage mean a
real order often fills at a less favorable price. See README.md.
"""

import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date

from config import settings
from src import journal
from src.storage import AccountStore
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
        trader = PaperTrader()      # new: $25 of pretend cash; else restored
        trader.buy("SPY", 0.04, 500.00, stop_loss_price=495.00)
        trader.unrealized_pnl({"SPY": 502.00})         # -> 0.08
        trader.sell("SPY", 0.04, 502.00)               # locks in +$0.08

    `starting_cash` is only used when the database has no account yet.
    Pass `order_id="..."` to buy/sell/deposit to make sure the same order
    can never be applied twice, even across restarts.
    """

    def __init__(self, starting_cash: float = settings.STARTING_CAPITAL,
                 risk_manager: RiskManager | None = None,
                 journal_file: str = settings.JOURNAL_FILE,
                 today=date.today,
                 db_path: str = settings.DATABASE_FILE):
        if not is_real_number(starting_cash) or starting_cash < 0:
            raise ValueError(f"Starting cash must be $0 or more (got {starting_cash!r}).")

        self.risk_manager = risk_manager or RiskManager()
        self.journal_file = journal_file
        # `today` is a function that returns today's date; tests can swap in
        # a fake one to pretend a new day has started.
        self._today = today

        self._in_operation = False
        self._pending_journal: list[dict] = []

        # Open the database. A brand-new file gets a fresh account with the
        # starting capital; an existing file is RESTORED, not reset.
        self.store = AccountStore(db_path)
        with self.store.transaction():
            if self.store.load_account() is None:
                self.store.create_account(float(starting_cash), today().isoformat())
                self.restored = False
            else:
                self.restored = True
            self._load()

    def close(self) -> None:
        """Close the database file (everything is already saved)."""
        self.store.close()

    # -- Loading and saving ------------------------------------------------------

    def _load(self) -> None:
        """Copy the saved account from the database into this object."""
        saved = self.store.load_account()
        self.cash = saved["cash"]
        self.starting_capital = saved["starting_capital"]
        self.realized_pnl = saved["realized_pnl"]             # all-time
        self.current_day = date.fromisoformat(saved["current_day"])
        self.realized_pnl_today = saved["realized_pnl_today"]    # net, today
        self.realized_loss_today = saved["realized_loss_today"]  # losses only
        self.positions: dict[str, Position] = {
            row["symbol"]: Position(**row) for row in self.store.load_positions()}

    def _save(self) -> None:
        """Copy this object's books into the database (inside a transaction)."""
        self.store.save_account(self.cash, self.realized_pnl,
                                self.current_day.isoformat(),
                                self.realized_pnl_today, self.realized_loss_today)
        self.store.save_positions([asdict(p) for p in self.positions.values()])

    @contextmanager
    def _operation(self):
        """
        Wrap one buy / sell / deposit / exit check so it is all-or-nothing:
            lock + reload -> (the operation runs) -> save -> commit -> journal
        """
        if self._in_operation:
            raise RuntimeError("Operations cannot be nested.")
        self._in_operation = True
        self._pending_journal = []
        try:
            with self.store.transaction():
                self._load()                      # never act on stale numbers
                self._start_new_day_if_needed()
                yield
                self._save()
        except BaseException:
            # The database rolled back. Reload so memory matches it again,
            # and throw away journal rows for things that did not happen.
            self._pending_journal = []
            self._load()
            raise
        finally:
            self._in_operation = False
        self._flush_journal()                     # only after a good commit

    def _refresh(self) -> None:
        """For read-only questions: pick up changes saved by anyone else."""
        if not self._in_operation:
            self._load()

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
        self._refresh()
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
            take_profit_price: float | None = None,
            order_id: str | None = None) -> TradeResult:
        """
        Simulate buying `shares` of `symbol` at `price`.
        The Risk Manager must approve first; nothing changes if it says no.

        `take_profit_price` is optional. If left out, it is set
        TAKE_PROFIT_PCT above the entry price (2%: $500 -> $510).
        """
        with self._operation():
            return self._buy_locked(symbol, shares, price, stop_loss_price,
                                    reason, take_profit_price, order_id)

    def _buy_locked(self, symbol, shares, price, stop_loss_price, reason,
                    take_profit_price, order_id) -> TradeResult:
        symbol = self._clean_symbol(symbol)

        # Step 0: refuse an order we have already processed.
        problem = self._order_id_problem(order_id)
        if problem:
            return self._reject("BUY", symbol, shares, price, problem)

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
        self._record("BUY", symbol, shares, price, message,
                     ledger=True, order_id=order_id)
        return TradeResult(True, "BUY", symbol, shares, price, message)

    # -- Selling ---------------------------------------------------------------

    def sell(self, symbol: str, shares: float, price: float,
             reason: str = "", order_id: str | None = None) -> TradeResult:
        """
        Simulate selling `shares` of `symbol` at `price`.
        Selling reduces risk, so it does not need Risk Manager approval,
        but it must be valid and we must actually own the shares.
        """
        with self._operation():
            return self._sell_locked(symbol, shares, price, reason, order_id)

    def _sell_locked(self, symbol, shares, price, reason, order_id) -> TradeResult:
        symbol = self._clean_symbol(symbol)

        problems = []
        order_problem = self._order_id_problem(order_id)
        if order_problem:
            problems.append(order_problem.removeprefix("Rejected: "))
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

        return self._fill_sell(held, shares, price, "SELL", reason, order_id)

    def _fill_sell(self, held: Position, shares: float, price: float,
                   action: str, reason: str,
                   order_id: str | None = None) -> TradeResult:
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
                     signal="SELL", ledger=True, order_id=order_id)
        return TradeResult(True, action, symbol, shares, price, message, pnl)

    def close_position(self, symbol: str, price: float, reason: str = "",
                       order_id: str | None = None) -> TradeResult:
        """Sell every share we hold of `symbol`."""
        with self._operation():
            symbol = self._clean_symbol(symbol)
            held = self.positions.get(symbol) if isinstance(symbol, str) else None
            if held is None:
                return self._reject("SELL", symbol, 0, price,
                                    f"Rejected: You don't hold any {symbol} to sell.")
            return self._sell_locked(symbol, held.shares, price, reason, order_id)

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
        All exits from one call are saved together (all or nothing).
        """
        with self._operation():
            return self._check_exits_locked(market_prices)

    def _check_exits_locked(self, market_prices) -> list[TradeResult]:
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
                reason: str = "Weekly contribution",
                order_id: str | None = None) -> TradeResult:
        """Add pretend cash, e.g. the $25 weekly contribution."""
        with self._operation():
            problem = self._order_id_problem(order_id)
            if problem:
                return self._reject("DEPOSIT", "", "", amount, problem)
            if not is_real_number(amount) or amount <= 0:
                return self._reject("DEPOSIT", "", "", amount,
                                    f"Rejected: Deposit must be more than $0 (got {amount!r}).")
            self.cash += amount
            self._record("DEPOSIT", "", "", amount, f"{reason}: +${amount:.2f}",
                         ledger=True, order_id=order_id)
            return TradeResult(True, "DEPOSIT", "", "", amount, reason)

    # -- Profit and loss with supplied market prices --------------------------

    def unrealized_pnl(self, market_prices: dict[str, float]) -> float:
        """
        Total paper profit/loss on open positions.
        `market_prices` is a dict like {"SPY": 502.00} supplied by the caller
        (simulated prices for now - this module never fetches prices itself).
        """
        self._refresh()
        return sum(held.unrealized_pnl(self._market_price(held.symbol, market_prices))
                   for held in self.positions.values())

    def total_equity(self, market_prices: dict[str, float]) -> float:
        """Cash plus what open positions are worth at `market_prices`."""
        self._refresh()
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

    def _order_id_problem(self, order_id) -> str | None:
        """A rejection reason if `order_id` is unusable or already used."""
        if order_id is None:
            return None
        if not isinstance(order_id, str) or not order_id.strip():
            return f"Rejected: order_id must be non-empty text (got {order_id!r})."
        if self.store.order_exists(order_id):
            return (f"Rejected: Duplicate order - order_id '{order_id}' was "
                    "already processed, so it was not applied again.")
        return None

    def _reject(self, side: str, symbol, shares, price, reason: str) -> TradeResult:
        """Record a refused order in the journal and report it. Books unchanged."""
        action = f"{side} REJECTED"
        self._record(action, symbol, shares, price, reason, signal=side)
        return TradeResult(False, action, symbol, shares, price, reason)

    def _record(self, action: str, symbol, shares, price, reason: str,
                realized_pnl: float | str = "", signal: str | None = None,
                ledger: bool = False, order_id: str | None = None) -> None:
        """
        Note one event for the trading journal.

        ledger=True (real money movements) also writes a row to the
        database's transactions table, inside the same all-or-nothing
        transaction as the change itself.
        """
        if ledger:
            self.store.add_transaction(
                action, symbol, shares if shares != "" else None, price,
                realized_pnl if realized_pnl != "" else None,
                self.cash, reason, order_id)
        row = dict(symbol=symbol, signal=signal or action, action=action,
                   shares=shares, price=price, reason=reason,
                   realized_pnl=round(realized_pnl, 4) if realized_pnl != "" else "",
                   cash_after=round(self.cash, 4), path=self.journal_file)
        self._pending_journal.append(row)
        if not self._in_operation:
            self._flush_journal()

    def _flush_journal(self) -> None:
        """
        Write waiting rows to the CSV journal. This runs only AFTER the
        database commit, so the CSV never shows something that was undone.
        If the CSV cannot be written, the trade still stands (it is safely
        in the database ledger) and a warning is shown instead of an error,
        so nobody is tempted to "retry" and trade twice.
        """
        rows, self._pending_journal = self._pending_journal, []
        for row in rows:
            try:
                journal.log_decision(**row)
            except OSError as error:
                warnings.warn(f"Could not write the CSV journal ({error}). The "
                              "transaction WAS saved in the database ledger.")
