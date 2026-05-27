from src.risk.manager import RiskManager


def test_risk_manager_closes_losing_fill() -> None:
    mgr = RiskManager(equity=10_000.0)
    mgr.record_entry("SPY")
    assert mgr.state.open_positions["SPY"] == 1

    mgr.record_fill("SPY", pnl=-25.0)
    assert "SPY" not in mgr.state.open_positions
