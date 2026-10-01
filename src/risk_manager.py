"""
risk_manager.py - Decides WHETHER a trade is allowed.

This is the agent's safety gate. Every proposed trade is checked against
the rules in config/settings.py. If any rule fails, the trade is blocked.
"""

from config import settings


def position_size(cash: float, price: float) -> float:
    """How many (possibly fractional) shares we may buy with our size limit."""
    budget = cash * settings.MAX_POSITION_SIZE_PCT
    shares = budget / price
    if not settings.ALLOW_FRACTIONAL_SHARES:
        shares = int(shares)
    return round(shares, 4)


def approve_trade(cash: float, price: float, trades_today: int,
                  loss_today: float, starting_equity: float) -> tuple[bool, str]:
    """Return (approved, reason). Reasons are written for beginners."""
    if not settings.PAPER_TRADING:
        return False, "Blocked: only paper trading is allowed."
    if trades_today >= settings.MAX_TRADES_PER_DAY:
        return False, "Blocked: daily trade limit reached."
    if loss_today >= starting_equity * settings.MAX_DAILY_LOSS_PCT:
        return False, "Blocked: daily loss limit reached. Done for today."
    if position_size(cash, price) <= 0:
        return False, "Blocked: not enough cash for this trade."
    return True, "Approved: trade is within all risk limits."
