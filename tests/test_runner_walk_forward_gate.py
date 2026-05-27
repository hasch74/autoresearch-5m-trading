from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.agent_runner.runner import ResearchRunner
from src.types import EvalResult, HypothesisStatus


class _DummyHypothesis:
    hypothesis_id = "h_dummy_wf"


def _make_df(symbol: str) -> pd.DataFrame:
    ts = pd.Timestamp("2026-01-02T14:30:00Z")
    return pd.DataFrame(
        [
            {
                "event_time": ts,
                "available_time": ts + pd.Timedelta(seconds=5),
                "symbol": symbol,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000,
                "atr_14": 1.0,
            }
        ]
    )


def _make_eval(hyp_id: str, net_pnl: float, trades: int = 5) -> EvalResult:
    return EvalResult(
        hypothesis_id=hyp_id,
        run_id="",
        status=HypothesisStatus.DRAFT,
        net_pnl=net_pnl,
        profit_factor=1.2,
        win_rate=0.6,
        avg_win=1.0,
        avg_loss=-0.8,
        max_drawdown=1.0,
        max_intraday_drawdown=1.0,
        worst_day=-1.0,
        longest_losing_streak=1,
        trades_per_day=0.7,
        exposure_time=0.2,
        total_trades=trades,
        slippage_sensitivity=0.5,
        composite_score=0.0,
    )


def test_runner_fails_when_walk_forward_is_weak(tmp_path: Path, monkeypatch) -> None:
    features_dir = tmp_path / "features"
    reports_dir = tmp_path / "reports"
    features_dir.mkdir(parents=True)

    _make_df("SPY").to_parquet(features_dir / "SPY_5m.parquet", index=False)

    monkeypatch.setattr("src.agent_runner.runner.load_all", lambda _p: {"h_dummy_wf": _DummyHypothesis()})

    def fake_run_backtest(df: pd.DataFrame, hyp: _DummyHypothesis) -> EvalResult:
        marker = float(df.iloc[0].get("wf_marker", 1.0))
        return _make_eval(hyp.hypothesis_id, marker)

    def fake_walk_forward_splits(df: pd.DataFrame, **_kwargs):
        t = df.copy()
        t["wf_marker"] = -2.0
        yield df.copy(), df.copy(), t

    monkeypatch.setattr("src.agent_runner.runner.run_backtest", fake_run_backtest)
    monkeypatch.setattr("src.agent_runner.runner.walk_forward_splits", fake_walk_forward_splits)
    monkeypatch.setattr("src.agent_runner.runner.score", lambda r: r)
    monkeypatch.setattr("src.agent_runner.runner.check_gates", lambda _r: SimpleNamespace(passed=True, failed_gates=[]))

    report = ResearchRunner(features_dir=features_dir, reports_dir=reports_dir).run_once()
    result = report["results"]["h_dummy_wf"]

    assert result["gate_passed"] is False
    assert result["walk_forward"]["passed"] is False
    assert any(g.startswith("walk_forward") for g in result["failed_gates"])
