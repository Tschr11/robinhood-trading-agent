"""
main.py - The starting point. Runs one simulated decision cycle.

Run from the project root with:
    python -m src.main

Flow: get prices -> strategy suggests -> risk manager checks
      -> paper trader simulates -> journal records.
"""

from config import settings
from src import journal, market_data, risk_manager, strategy
from src.paper_trader import PaperAccount


def run_once(account: PaperAccount) -> None:
    for symbol in settings.WATCHLIST:
        prices = market_data.get_recent_prices(symbol)
        price = prices[-1]
        signal, reason = strategy.generate_signal(prices)
        action, shares = "NONE", 0.0

        if signal == "BUY":
            approved, risk_reason = risk_manager.approve_trade(
                cash=account.cash, price=price, trades_today=0,
                loss_today=0.0, starting_equity=account.cash)
            reason = f"{reason} {risk_reason}"
            if approved:
                shares = risk_manager.position_size(account.cash, price)
                account.buy(symbol, shares, price)
                action = "SIMULATED BUY"

        journal.log_decision(symbol, signal, action, shares, price, reason)
        print(f"{symbol}: {signal} -> {action}. {reason}")

    print(f"Simulated cash remaining: ${account.cash:.2f}")


if __name__ == "__main__":
    print("Robinhood Trading Agent - PAPER TRADING ONLY (no real money)")
    run_once(PaperAccount())
