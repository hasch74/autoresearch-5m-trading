from datetime import datetime, timezone

from src.risk.manager import RiskManager
from src.types import Direction, Order


def test_risk_manager_closes_losing_fill() -> None:
    mgr = RiskManager(equity=10_000.0)
    mgr.record_entry("SPY")
    assert mgr.state.open_positions["SPY"] == 1

    mgr.record_fill("SPY", pnl=-25.0)
    assert "SPY" not in mgr.state.open_positions


def test_risk_manager_activates_kill_switch_after_repeated_rejections() -> None:
    mgr = RiskManager(equity=10_000.0, kill_switch_after_rejections=2)
    created_at = datetime(2026, 1, 2, tzinfo=timezone.utc)

    # Build a valid order object first, then break the bracket to exercise manager logic.
    valid_order = Order(
        order_id="o-2",
        hypothesis_id="h_0001",
        symbol="SPY",
        direction=Direction.LONG,
        quantity=1,
        stop_price=99.0,
        take_profit_price=102.0,
        created_at=created_at,
    )
    valid_order.stop_price = None

    assert mgr.validate_order(valid_order) is False
    assert mgr.state.kill_switch_active is False
    assert mgr.validate_order(valid_order) is False
    assert mgr.state.kill_switch_active is True
    assert mgr.last_rejection_reason == "bracket order required: stop_price and take_profit_price must be set"
