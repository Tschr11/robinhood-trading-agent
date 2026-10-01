"""
main.py - The starting point. Prints one round of strategy signals.

Run from the project root with:
    python -m src.main

Flow: load historical candles -> strategy evaluates its rules
      -> signal is printed and written to the journal.

Because the only data source so far is HISTORICAL (your CSV files in
data/market/), this script does NOT place any trades, not even paper ones.
Acting on signals needs live data and the (not yet built) trading loop.
If a file is missing or invalid, that symbol is skipped - the agent never
makes up prices.
"""

from config import settings
from src import journal, strategy
from src.market_data import CSVHistoricalProvider, DataKind, MarketDataError
from src.paper_trader import PaperTrader


def run_once(trader: PaperTrader, provider=None) -> list:
    """Evaluate every watchlist symbol and return the signals (no trades)."""
    provider = provider or CSVHistoricalProvider()
    signals = []
    for symbol in settings.WATCHLIST:
        try:
            data = provider.get_candles(symbol)
        except MarketDataError as error:
            journal.log_decision(symbol, "NONE", "SKIPPED", 0.0, "",
                                 f"No usable market data: {error}")
            print(f"{symbol}: SKIPPED - no usable market data. {error}")
            continue

        # Reading the account only - the strategy never changes it.
        result = strategy.evaluate(
            data, expected_kind=DataKind.HISTORICAL,
            has_open_position=symbol in trader.positions)
        signals.append(result)
        journal.log_decision(symbol, result.signal.value, "SIGNAL ONLY", 0.0,
                             data.latest_close, result.explain().replace("\n", " | "))
        print(result.explain())
    return signals


if __name__ == "__main__":
    print("Robinhood Trading Agent - PAPER TRADING ONLY (no real money)")
    print("Signals from historical data are for research only; no trades are placed.\n")
    trader = PaperTrader()
    try:
        run_once(trader)
    finally:
        trader.close()
