"""
Verification tests for the Risk Manager's safeguards.

Each test class covers one safeguard. Besides the normal cases, these tests
also feed in "bad data" - negative numbers, NaN ("not a number"), infinity,
and wrong types - because a safety gate must reject garbage, not approve it.

Run from the project root with:
    python -m unittest discover tests -v

Default limits for a $25 account:
    per-trade risk limit = 2% of $25 = $0.50
    daily loss limit     = 5% of $25 = $1.25
    max open positions   = 1
"""

import math
import unittest

from src.risk_manager import AccountState, RiskManager, TradeRequest

NAN = math.nan
INF = math.inf


def account(**changes) -> AccountState:
    """A brand-new $25 paper account, with optional overrides."""
    values = dict(cash=25.00, equity=25.00, open_positions=0,
                  realized_loss_today=0.0, paper_trading=True)
    values.update(changes)
    return AccountState(**values)


def good_trade(**changes) -> TradeRequest:
    """A trade that passes every rule: costs $20.00, risks $0.20."""
    values = dict(symbol="SPY", shares=0.04, entry_price=500.00,
                  stop_loss_price=495.00)
    values.update(changes)
    return TradeRequest(**values)


class SafeguardTestCase(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager()

    def assertApproved(self, trade, acct):
        decision = self.rm.evaluate(trade, acct)
        self.assertTrue(decision.approved, decision.explain())
        self.assertEqual(decision.reasons, [])

    def assertRejected(self, trade, acct, words):
        """Must be rejected, and the explanation must contain `words`."""
        decision = self.rm.evaluate(trade, acct)
        self.assertFalse(decision.approved,
                         f"Expected rejection for {trade} / {acct}")
        self.assertTrue(decision.reasons, "A rejection must give a reason.")
        self.assertIn(words, decision.explain())


# 1. A valid trade is approved ------------------------------------------------

class ValidTradeIsApproved(SafeguardTestCase):
    def test_normal_trade(self):
        self.assertApproved(good_trade(), account())

    def test_trade_exactly_on_every_limit(self):
        # cost $25.00 = all cash; risk 0.05 x $10 = $0.50 = risk limit
        trade = good_trade(shares=0.05, stop_loss_price=490.00)
        self.assertApproved(trade, account())

    def test_whole_share_trade(self):
        # 1 share of a $20 stock, stop at $19.50 -> risk $0.50
        trade = good_trade(shares=1, entry_price=20, stop_loss_price=19.5)
        self.assertApproved(trade, account())

    def test_decision_explains_approval(self):
        decision = self.rm.evaluate(good_trade(), account())
        self.assertEqual(decision.explain(),
                         "Approved: the trade is within all risk limits.")


# 2. Exceeding the risk limit is rejected -------------------------------------

class RiskLimitIsEnforced(SafeguardTestCase):
    def test_just_over_risk_limit(self):
        # risk 0.0501 x $10 = $0.501 > $0.50 (cost $25.05 also > cash)
        trade = good_trade(shares=0.0501, stop_loss_price=490.00)
        self.assertRejected(trade, account(), "per-trade limit")

    def test_affordable_but_too_risky(self):
        # cost $20 fits in cash, but risk 0.04 x $50 = $2.00 > $0.50
        trade = good_trade(stop_loss_price=450.00)
        self.assertRejected(trade, account(), "Trade risks $2.00")

    def test_risk_limit_follows_configuration(self):
        strict = RiskManager(max_risk_per_trade_pct=0.005)  # $0.125
        decision = strict.evaluate(good_trade(), account())  # risk $0.20
        self.assertFalse(decision.approved)
        self.assertIn("per-trade limit", decision.explain())


# 3. Exceeding available cash is rejected -------------------------------------

class CashLimitIsEnforced(SafeguardTestCase):
    def test_one_cent_over_cash(self):
        # cost $25.01, risk only $0.05
        trade = good_trade(shares=1, entry_price=25.01, stop_loss_price=24.96)
        self.assertRejected(trade, account(), "cash is available")

    def test_low_cash_with_higher_equity(self):
        # $20 cost but only $5 cash left (the rest is tied up elsewhere)
        self.assertRejected(good_trade(), account(cash=5.00), "cash is available")

    def test_zero_cash(self):
        self.assertRejected(good_trade(), account(cash=0.0), "cash is available")


# 4. Exceeding max open positions is rejected ---------------------------------

class OpenPositionLimitIsEnforced(SafeguardTestCase):
    def test_one_position_already_open(self):
        self.assertRejected(good_trade(), account(open_positions=1),
                            "the limit is 1")

    def test_many_positions_already_open(self):
        self.assertRejected(good_trade(), account(open_positions=5),
                            "the limit is 1")


# 5. Reaching the daily loss limit is rejected --------------------------------

class DailyLossLimitIsEnforced(SafeguardTestCase):
    def test_exactly_at_daily_limit(self):
        self.assertRejected(good_trade(), account(realized_loss_today=1.25),
                            "Daily loss limit reached")

    def test_past_daily_limit(self):
        self.assertRejected(good_trade(), account(realized_loss_today=3.00),
                            "Daily loss limit reached")

    def test_trade_would_cross_daily_limit(self):
        # $1.10 lost, $0.15 of room; trade risks $0.20
        self.assertRejected(good_trade(), account(realized_loss_today=1.10),
                            "would exceed the $1.25 daily limit")

    def test_just_under_limit_with_small_trade_is_ok(self):
        # $1.00 lost, $0.25 of room; trade risks $0.20
        self.assertApproved(good_trade(), account(realized_loss_today=1.00))


# 6. Invalid or negative values are rejected ----------------------------------

class InvalidValuesAreRejected(SafeguardTestCase):
    def test_bad_share_quantities(self):
        for shares in [0, -0.04, -1, NAN, INF, -INF]:
            with self.subTest(shares=shares):
                self.assertRejected(good_trade(shares=shares), account(), "Share")

    def test_bad_entry_prices(self):
        for price in [0, -500.00, NAN, INF]:
            with self.subTest(entry_price=price):
                self.assertRejected(good_trade(entry_price=price), account(),
                                    "Entry price")

    def test_bad_stop_loss_prices(self):
        # zero, negative, above entry, equal to entry, NaN
        for stop in [0, -1, 505.00, 500.00, NAN, -INF]:
            with self.subTest(stop_loss_price=stop):
                self.assertRejected(good_trade(stop_loss_price=stop), account(),
                                    "Stop-loss")

    def test_wrong_types_are_rejected_not_crashed(self):
        bad_trades = [good_trade(shares="0.04"), good_trade(entry_price=None),
                      good_trade(stop_loss_price="cheap"), good_trade(shares=True)]
        for trade in bad_trades:
            with self.subTest(trade=trade):
                self.assertRejected(trade, account(), "must be a number")

    def test_empty_symbol(self):
        for symbol in ["", "   ", None]:
            with self.subTest(symbol=symbol):
                self.assertRejected(good_trade(symbol=symbol), account(), "Symbol")

    def test_bad_account_values(self):
        bad_accounts = {
            "negative cash": account(cash=-5.00),
            "NaN cash": account(cash=NAN),
            "negative equity": account(equity=-25.00),
            "NaN equity": account(equity=NAN),
            "infinite equity": account(equity=INF),
            "equity below cash": account(cash=25.00, equity=10.00),
            "negative open positions": account(open_positions=-1),
            "fractional open positions": account(open_positions=0.5),
            "negative loss today": account(realized_loss_today=-10.00),
            "NaN loss today": account(realized_loss_today=NAN),
        }
        for label, acct in bad_accounts.items():
            with self.subTest(label):
                self.assertRejected(good_trade(), acct, "Account")

    def test_bad_limits_refused_at_setup(self):
        bad_settings = [dict(max_risk_per_trade_pct=0),
                        dict(max_risk_per_trade_pct=-0.02),
                        dict(max_risk_per_trade_pct=2),
                        dict(max_risk_per_trade_pct=NAN),
                        dict(max_daily_loss_pct=0),
                        dict(max_daily_loss_pct=NAN),
                        dict(max_open_positions=0),
                        dict(max_open_positions=1.5),
                        dict(max_risk_per_trade_pct="0.02")]
        for kwargs in bad_settings:
            with self.subTest(**{k: repr(v) for k, v in kwargs.items()}):
                with self.assertRaises(ValueError):
                    RiskManager(**kwargs)

    def test_position_sizing_never_suggests_bad_sizes(self):
        cases = [(500.00, NAN), (NAN, 495.00), (500.00, 500.00), (-1, -2),
                 (500.00, -5.00)]
        for entry, stop in cases:
            with self.subTest(entry=entry, stop=stop):
                self.assertEqual(self.rm.max_shares(account(), entry, stop), 0.0)
        self.assertEqual(
            self.rm.max_shares(account(cash=NAN), 500.00, 495.00), 0.0)


if __name__ == "__main__":
    unittest.main()
