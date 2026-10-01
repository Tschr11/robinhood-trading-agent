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
    risks = risk_manager.RiskManager()
    for symbol in settings.WATCHLIST:
        prices = market_data.get_recent_prices(symbol)
        price = prices[-1]
        signal, reason = strategy.generate_signal(prices)
        action, shares = "NONE", 0.0

        if signal == "BUY":
            state = risk_manager.AccountState(
                cash=account.cash,
                equity=account.total_value(
                    {s: market_data.get_latest_price(s) for s in account.positions}),
                open_positions=len(account.positions))
            stop = round(price * (1 - settings.STOP_LOSS_PCT), 2)
            shares = risks.max_shares(state, price, stop)
            decision = risks.evaluate(
                risk_manager.TradeRequest(symbol, shares, price, stop), state)
            reason = f"{reason} {decision.explain()}"
            if decision.approved:
                account.buy(symbol, shares, price)
                action = "SIMULATED BUY"
            else:
                shares = 0.0

        journal.log_decision(symbol, signal, action, shares, price, reason)
        print(f"{symbol}: {signal} -> {action}. {reason}")

    print(f"Simulated cash remaining: ${account.cash:.2f}")


if __name__ == "__main__":
    print("Robinhood Trading Agent - PAPER TRADING ONLY (no real money)")
    run_once(PaperAccount())
