"""
Tests for src/risk_manager.py.

Run from the project root with:
    python -m unittest discover tests -v

Every test builds a small, made-up situation and checks that the risk
manager approves or rejects it - and, for rejections, that it says why.

Numbers used in most tests (default settings, $25 account):
    per-trade risk limit = 2% of $25 = $0.50
    daily loss limit     = 5% of $25 = $1.25
"""

import unittest

from src.risk_manager import AccountState, RiskManager, TradeRequest


def fresh_account(**changes) -> AccountState:
    """A brand-new $25 paper account. Override any field, e.g. cash=10."""
    values = dict(cash=25.00, equity=25.00, open_positions=0,
                  realized_loss_today=0.0, paper_trading=True)
    values.update(changes)
    return AccountState(**values)


class ApprovedTradeTests(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager()  # defaults from config/settings.py

    def test_small_trade_within_all_limits(self):
        # cost 0.04 x $500 = $20.00, risk 0.04 x $5 = $0.20
        trade = TradeRequest("SPY", 0.04, 500.00, 495.00)
        decision = self.rm.evaluate(trade, fresh_account())
        self.assertTrue(decision.approved)
        self.assertEqual(decision.reasons, [])
        self.assertIn("Approved", decision.explain())

    def test_risk_exactly_at_limit_is_allowed(self):
        # risk 0.1 x $5 = $0.50, exactly the 2% limit
        trade = TradeRequest("SPY", 0.1, 200.00, 195.00)
        self.assertTrue(self.rm.evaluate(trade, fresh_account()).approved)

    def test_using_exactly_all_cash_is_allowed(self):
        # cost 0.05 x $500 = $25.00, exactly the cash available
        trade = TradeRequest("SPY", 0.05, 500.00, 495.00)
        self.assertTrue(self.rm.evaluate(trade, fresh_account()).approved)

    def test_trade_allowed_when_daily_loss_has_room(self):
        # lost $0.75 today; this trade risks $0.50 -> worst case $1.25 = limit
        trade = TradeRequest("SPY", 0.1, 200.00, 195.00)
        account = fresh_account(realized_loss_today=0.75)
        self.assertTrue(self.rm.evaluate(trade, account).approved)

    def test_looser_custom_limit_allows_bigger_risk(self):
        # risk $1.00 is too much at 2%, but fine at a custom 4% ($1.00)
        rm = RiskManager(max_risk_per_trade_pct=0.04)
        trade = TradeRequest("SPY", 0.1, 200.00, 190.00)
        self.assertTrue(rm.evaluate(trade, fresh_account()).approved)

    def test_suggested_max_shares_is_always_approved(self):
        account = fresh_account()
        shares = self.rm.max_shares(account, 500.00, 495.00)
        self.assertGreater(shares, 0)
        trade = TradeRequest("SPY", shares, 500.00, 495.00)
        self.assertTrue(self.rm.evaluate(trade, account).approved)


class RejectedTradeTests(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager()

    def assertRejectedFor(self, decision, words):
        """The trade must be rejected AND the reason must mention `words`."""
        self.assertFalse(decision.approved)
        self.assertTrue(decision.explain().startswith("Rejected:"))
        self.assertIn(words, decision.explain())

    def test_rejects_trade_costing_more_than_cash(self):
        # cost 0.06 x $500 = $30.00 > $25.00 cash (risk only $0.30)
        trade = TradeRequest("SPY", 0.06, 500.00, 495.00)
        self.assertRejectedFor(self.rm.evaluate(trade, fresh_account()),
                               "cash is available")

    def test_rejects_trade_risking_too_much(self):
        # risk 0.1 x $10 = $1.00 > $0.50 limit (cost $20 is affordable)
        trade = TradeRequest("SPY", 0.1, 200.00, 190.00)
        self.assertRejectedFor(self.rm.evaluate(trade, fresh_account()),
                               "per-trade limit")

    def test_rejects_second_open_position(self):
        trade = TradeRequest("QQQ", 0.04, 400.00, 396.00)
        account = fresh_account(open_positions=1)
        self.assertRejectedFor(self.rm.evaluate(trade, account),
                               "open position(s); the limit is 1")

    def test_rejects_after_daily_loss_limit_reached(self):
        trade = TradeRequest("SPY", 0.01, 500.00, 495.00)
        account = fresh_account(realized_loss_today=1.25)
        self.assertRejectedFor(self.rm.evaluate(trade, account),
                               "Daily loss limit reached")

    def test_rejects_trade_that_could_break_daily_limit(self):
        # lost $1.00 today, only $0.25 room left; this trade risks $0.40
        trade = TradeRequest("SPY", 0.08, 200.00, 195.00)
        account = fresh_account(realized_loss_today=1.00)
        self.assertRejectedFor(self.rm.evaluate(trade, account),
                               "would exceed the $1.25 daily limit")

    def test_rejects_when_not_paper_trading(self):
        trade = TradeRequest("SPY", 0.04, 500.00, 495.00)
        account = fresh_account(paper_trading=False)
        self.assertRejectedFor(self.rm.evaluate(trade, account),
                               "Only paper trading is allowed")

    def test_rejects_zero_shares(self):
        trade = TradeRequest("SPY", 0, 500.00, 495.00)
        self.assertRejectedFor(self.rm.evaluate(trade, fresh_account()),
                               "greater than zero")

    def test_rejects_negative_price(self):
        trade = TradeRequest("SPY", 0.04, -5.00, -6.00)
        self.assertRejectedFor(self.rm.evaluate(trade, fresh_account()),
                               "Entry price must be greater than zero")

    def test_rejects_stop_loss_above_entry(self):
        trade = TradeRequest("SPY", 0.04, 500.00, 505.00)
        self.assertRejectedFor(self.rm.evaluate(trade, fresh_account()),
                               "Stop-loss must be above $0 and below")

    def test_rejects_missing_stop_loss(self):
        trade = TradeRequest("SPY", 0.04, 500.00, 0)
        self.assertRejectedFor(self.rm.evaluate(trade, fresh_account()),
                               "Stop-loss")

    def test_lists_every_broken_rule(self):
        # too expensive, too risky, a position is already open,
        # and the worst-case loss would break the daily limit
        trade = TradeRequest("SPY", 1.0, 500.00, 450.00)
        account = fresh_account(open_positions=1)
        decision = self.rm.evaluate(trade, account)
        self.assertFalse(decision.approved)
        self.assertEqual(len(decision.reasons), 4)

    def test_stricter_custom_limit_rejects(self):
        # risk $0.20 is fine at 2% but not at a custom 0.5% ($0.125)
        rm = RiskManager(max_risk_per_trade_pct=0.005)
        trade = TradeRequest("SPY", 0.04, 500.00, 495.00)
        self.assertRejectedFor(rm.evaluate(trade, fresh_account()),
                               "per-trade limit")


class ConfigurationTests(unittest.TestCase):
    def test_rejects_percent_written_as_whole_number(self):
        with self.assertRaises(ValueError):
            RiskManager(max_risk_per_trade_pct=2)  # meant 0.02

    def test_rejects_zero_daily_loss_limit(self):
        with self.assertRaises(ValueError):
            RiskManager(max_daily_loss_pct=0)

    def test_rejects_zero_open_positions(self):
        with self.assertRaises(ValueError):
            RiskManager(max_open_positions=0)

    def test_allows_more_positions_when_configured(self):
        rm = RiskManager(max_open_positions=2)
        trade = TradeRequest("QQQ", 0.04, 400.00, 396.00)
        self.assertTrue(rm.evaluate(trade, fresh_account(open_positions=1)).approved)


class PositionSizingTests(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager()

    def test_size_limited_by_cash(self):
        # risk allows 0.1 shares ($0.50 / $5), but cash allows only 0.05
        self.assertEqual(self.rm.max_shares(fresh_account(), 500.00, 495.00), 0.05)

    def test_size_limited_by_risk(self):
        # risk allows $0.50 / $10 = 0.05 shares; cash would allow 0.25
        self.assertEqual(self.rm.max_shares(fresh_account(), 100.00, 90.00), 0.05)

    def test_size_zero_for_bad_stop(self):
        self.assertEqual(self.rm.max_shares(fresh_account(), 100.00, 100.00), 0.0)


if __name__ == "__main__":
    unittest.main()
