"""
main.py - The starting point. Runs one simulated decision cycle.

Run from the project root with:
    python -m src.main

Flow: get prices -> strategy suggests -> risk manager checks
      -> paper trader simulates -> journal records.
"""

from config import settings
from src import journal, market_data, strategy
from src.paper_trader import PaperTrader


def run_once(trader: PaperTrader) -> None:
    for symbol in settings.WATCHLIST:
        prices = market_data.get_recent_prices(symbol)
        price = prices[-1]
        signal, reason = strategy.generate_signal(prices)

        if signal == "BUY":
            # The paper trader asks the risk manager and writes the journal.
            stop = round(price * (1 - settings.STOP_LOSS_PCT), 2)
            shares = trader.risk_manager.max_shares(
                trader.account_state(), price, stop)
            result = trader.buy(symbol, shares, price, stop, reason)
            print(f"{symbol}: BUY -> {result.action}. {result.reason}")
        else:
            journal.log_decision(symbol, signal, "NONE", 0.0, price, reason)
            print(f"{symbol}: {signal} -> NONE. {reason}")

    print(f"Simulated cash remaining: ${trader.cash:.2f}")


if __name__ == "__main__":
    print("Robinhood Trading Agent - PAPER TRADING ONLY (no real money)")
    run_once(PaperTrader())
