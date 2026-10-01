"""
paper_trader.py - Pretends to place trades.

This is a simulated account. It tracks cash and positions in memory only.
It does NOT connect to Robinhood or any other broker, and no real money moves.
"""

from config import settings


class PaperAccount:
    def __init__(self, cash: float = settings.STARTING_CAPITAL):
        self.cash = cash
        self.positions = {}  # symbol -> number of shares

    def buy(self, symbol: str, shares: float, price: float) -> None:
        cost = shares * price
        if cost > self.cash:
            raise ValueError("Not enough simulated cash.")
        self.cash -= cost
        self.positions[symbol] = self.positions.get(symbol, 0) + shares

    def sell(self, symbol: str, shares: float, price: float) -> None:
        if self.positions.get(symbol, 0) < shares:
            raise ValueError("Not enough simulated shares.")
        self.cash += shares * price
        self.positions[symbol] -= shares
        if self.positions[symbol] == 0:
            del self.positions[symbol]

    def add_contribution(self, amount: float = settings.WEEKLY_CONTRIBUTION) -> None:
        """Simulate the weekly deposit."""
        self.cash += amount

    def total_value(self, prices: dict[str, float]) -> float:
        """Cash plus the current value of every position."""
        holdings = sum(shares * prices[s] for s, shares in self.positions.items())
        return round(self.cash + holdings, 2)
