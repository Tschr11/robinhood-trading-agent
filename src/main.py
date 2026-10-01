"""
main.py - The starting point. Runs one paper-trading decision cycle.

Run from the project root with:
    python -m src.main

Flow: load historical candles -> strategy suggests -> risk manager checks
      -> paper trader simulates -> journal records.

Prices come from your own CSV files in data/market/ (see README.md).
If a file is missing or invalid, that symbol is skipped - the agent never
makes up prices. These are HISTORICAL prices, not live market quotes.
"""

from config import settings
from src import journal, strategy
from src.market_data import CSVHistoricalProvider, MarketDataError
from src.paper_trader import PaperTrader


def run_once(trader: PaperTrader, provider=None) -> None:
    provider = provider or CSVHistoricalProvider()
    for symbol in settings.WATCHLIST:
        try:
            data = provider.get_candles(symbol, min_candles=5)
        except MarketDataError as error:
            journal.log_decision(symbol, "NONE", "SKIPPED", 0.0, "",
                                 f"No usable market data: {error}")
            print(f"{symbol}: SKIPPED - no usable market data. {error}")
            continue

        prices = data.closes[-20:]
        price = data.latest_close
        signal, reason = strategy.generate_signal(prices)
        reason = f"{reason} ({data.kind.value} data from {data.source})"

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
