from src.evaluator.scoring import check_gates
from src.types import EvalResult, HypothesisStatus


def _eval_result(*, net_pnl: float, max_drawdown: float) -> EvalResult:
    return EvalResult(
        hypothesis_id="h_test",
        run_id="run-1",
        status=HypothesisStatus.DRAFT,
        net_pnl=net_pnl,
        profit_factor=1.5,
        win_rate=0.55,
        avg_win=3.0,
        avg_loss=-2.0,
        max_drawdown=max_drawdown,
        max_intraday_drawdown=50.0,
        worst_day=-25.0,
        longest_losing_streak=2,
        trades_per_day=1.0,
        exposure_time=0.2,
        total_trades=100,
        slippage_sensitivity=0.4,
        composite_score=0.0,
    )


def test_drawdown_gate_does_not_depend_on_net_pnl_scale() -> None:
    thresholds = {"starting_equity": 2_000.0}

    low_pnl = check_gates(_eval_result(net_pnl=200.0, max_drawdown=150.0), thresholds)
    high_pnl = check_gates(_eval_result(net_pnl=2_000.0, max_drawdown=150.0), thresholds)

    assert low_pnl.passed is True
    assert high_pnl.passed is True
    assert all("max_drawdown" not in failure for failure in low_pnl.failed_gates)
    assert all("max_drawdown" not in failure for failure in high_pnl.failed_gates)


def test_drawdown_gate_uses_starting_equity_limit() -> None:
    gate = check_gates(
        _eval_result(net_pnl=2_000.0, max_drawdown=250.0),
        {"starting_equity": 2_000.0},
    )

    assert gate.passed is False
    assert any("max_drawdown" in failure for failure in gate.failed_gates)