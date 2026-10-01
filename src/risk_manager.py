"""
risk_manager.py - Decides WHETHER a trade is allowed.

This is the agent's safety gate. Before any (simulated) trade happens, the
agent describes it as a TradeRequest and asks the RiskManager to check it.
The RiskManager answers with a RiskDecision: approved or rejected, plus a
plain-English reason for every rule that failed.

Key idea - "risk" is not the same as "cost":
    cost = shares x entry price          (how much cash the trade uses)
    risk = shares x (entry - stop-loss)  (how much we lose if the stop-loss hits)

Example: buy 0.05 shares at $500 with a stop-loss at $495.
    cost = 0.05 x 500       = $25.00
    risk = 0.05 x (500-495) = $0.25

This module only does math and checks rules. It never places orders and
never connects to Robinhood or any other brokerage.
"""

from dataclasses import dataclass, field

from config import settings

# Computers store decimals slightly imprecisely (0.1 + 0.2 = 0.30000000000000004).
# This tiny tolerance stops a trade that sits EXACTLY on a limit from being
# rejected because of that rounding noise.
TOLERANCE = 1e-9


# --- Inputs ------------------------------------------------------------------

@dataclass
class TradeRequest:
    """A proposed new BUY (long) position that the agent would like to open."""
    symbol: str
    shares: float           # fractional shares are allowed, e.g. 0.05
    entry_price: float      # price we expect to buy at
    stop_loss_price: float  # price where we would sell to cut the loss


@dataclass
class AccountState:
    """A snapshot of the simulated account at the moment of the check."""
    cash: float                  # cash available to spend right now
    equity: float                # cash + value of open positions
    open_positions: int = 0      # how many positions are currently held
    realized_loss_today: float = 0.0  # dollars lost today (a positive number)
    paper_trading: bool = settings.PAPER_TRADING


# --- Output ------------------------------------------------------------------

@dataclass
class RiskDecision:
    """The answer from the risk manager."""
    approved: bool
    reasons: list[str] = field(default_factory=list)

    def explain(self) -> str:
        """One readable sentence (or a few) for logs and the dashboard."""
        if self.approved:
            return "Approved: the trade is within all risk limits."
        return "Rejected: " + " ".join(self.reasons)


# --- The risk manager --------------------------------------------------------

class RiskManager:
    """
    Checks trades against configurable limits.

    By default the limits come from config/settings.py, but you can pass
    different values, for example: RiskManager(max_risk_per_trade_pct=0.01)
    """

    def __init__(self,
                 max_risk_per_trade_pct: float = settings.MAX_RISK_PER_TRADE_PCT,
                 max_daily_loss_pct: float = settings.MAX_DAILY_LOSS_PCT,
                 max_open_positions: int = settings.MAX_OPEN_POSITIONS):
        # Refuse obviously broken settings right away, so a typo like 2
        # (meaning 200%) instead of 0.02 can never slip through silently.
        for name, pct in [("max_risk_per_trade_pct", max_risk_per_trade_pct),
                          ("max_daily_loss_pct", max_daily_loss_pct)]:
            if not 0 < pct <= 1:
                raise ValueError(f"{name} must be between 0 and 1 (got {pct}).")
        if max_open_positions < 1:
            raise ValueError("max_open_positions must be at least 1.")

        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_open_positions = max_open_positions

    # -- Helper math ----------------------------------------------------------

    def max_risk_dollars(self, account: AccountState) -> float:
        """Most we may lose on a single trade, e.g. 2% of $25 = $0.50."""
        return account.equity * self.max_risk_per_trade_pct

    def daily_loss_limit_dollars(self, account: AccountState) -> float:
        """Most we may lose in one day, e.g. 5% of $25 = $1.25."""
        return account.equity * self.max_daily_loss_pct

    @staticmethod
    def trade_cost(trade: TradeRequest) -> float:
        return trade.shares * trade.entry_price

    @staticmethod
    def trade_risk(trade: TradeRequest) -> float:
        return trade.shares * (trade.entry_price - trade.stop_loss_price)

    def max_shares(self, account: AccountState, entry_price: float,
                   stop_loss_price: float) -> float:
        """
        The largest position that passes BOTH the risk limit and the cash limit.
        Rounded DOWN to 4 decimals so it never goes over either limit.
        """
        risk_per_share = entry_price - stop_loss_price
        if entry_price <= 0 or risk_per_share <= 0:
            return 0.0
        by_risk = self.max_risk_dollars(account) / risk_per_share
        by_cash = account.cash / entry_price
        return int(min(by_risk, by_cash) * 10_000) / 10_000

    # -- The main check -------------------------------------------------------

    def evaluate(self, trade: TradeRequest, account: AccountState) -> RiskDecision:
        """
        Run every rule and collect ALL failures (not just the first one),
        so the reason tells the whole story.
        """
        reasons = []

        # Rule 1: simulation only. No real-money trading, ever.
        if not account.paper_trading:
            reasons.append("Only paper trading is allowed.")

        # Rule 2: the trade itself must make sense.
        if trade.shares <= 0:
            reasons.append("Share quantity must be greater than zero.")
        if trade.entry_price <= 0:
            reasons.append("Entry price must be greater than zero.")
        if trade.stop_loss_price <= 0 or trade.stop_loss_price >= trade.entry_price:
            reasons.append("Stop-loss must be above $0 and below the entry price.")

        # If the inputs are invalid, the dollar math below would be meaningless.
        if reasons:
            return RiskDecision(approved=False, reasons=reasons)

        cost = self.trade_cost(trade)
        risk = self.trade_risk(trade)

        # Rule 3: we can't spend money we don't have.
        if cost > account.cash + TOLERANCE:
            reasons.append(f"Trade costs ${cost:.2f} but only "
                           f"${account.cash:.2f} cash is available.")

        # Rule 4: only a limited number of open positions at once.
        if account.open_positions >= self.max_open_positions:
            reasons.append(f"Already holding {account.open_positions} open "
                           f"position(s); the limit is {self.max_open_positions}.")

        # Rule 5: the loss if the stop-loss hits must be small.
        max_risk = self.max_risk_dollars(account)
        if risk > max_risk + TOLERANCE:
            reasons.append(f"Trade risks ${risk:.2f}, more than the "
                           f"${max_risk:.2f} per-trade limit "
                           f"({self.max_risk_per_trade_pct:.0%} of the account).")

        # Rule 6: daily loss limit. Stop once the limit is reached, and don't
        # take a trade whose worst case would push us past it.
        daily_limit = self.daily_loss_limit_dollars(account)
        if account.realized_loss_today >= daily_limit - TOLERANCE:
            reasons.append(f"Daily loss limit reached (lost "
                           f"${account.realized_loss_today:.2f} of "
                           f"${daily_limit:.2f} allowed). No more trades today.")
        elif account.realized_loss_today + risk > daily_limit + TOLERANCE:
            room = daily_limit - account.realized_loss_today
            reasons.append(f"If the stop-loss hits, today's loss would exceed "
                           f"the ${daily_limit:.2f} daily limit (only "
                           f"${room:.2f} of room left).")

        return RiskDecision(approved=not reasons, reasons=reasons)
