from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.agent_runner.runner import ResearchRunner
from src.types import EvalResult, HypothesisStatus


class _DummyHypothesis:
    hypothesis_id = "h_dummy"


def _feature_df(symbol: str) -> pd.DataFrame:
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
                "rvol_20": 1.2,
                "vwap": 100.4,
                "or_high": 100.8,
                "or_low": 99.2,
                "gap_pct": 0.01,
                "session_gap_pct": 0.01,
                "ret_open_to_now": 0.001,
            }
        ]
    )


def test_runner_backtests_each_symbol_separately(tmp_path: Path, monkeypatch) -> None:
    features_dir = tmp_path / "features"
    reports_dir = tmp_path / "reports"
    features_dir.mkdir(parents=True)

    _feature_df("SPY").to_parquet(features_dir / "SPY_5m.parquet", index=False)
    _feature_df("QQQ").to_parquet(features_dir / "QQQ_5m.parquet", index=False)

    def fake_load_all(_):
        return {"h_dummy": _DummyHypothesis()}

    def fake_run_backtest(df, _hyp):
        assert df["symbol"].nunique() == 1
        symbol = df["symbol"].iloc[0]
        trades = 2 if symbol == "SPY" else 3
        net = 5.0 if symbol == "SPY" else -1.0
        return EvalResult(
            hypothesis_id="h_dummy",
            run_id="",
            status=HypothesisStatus.DRAFT,
            net_pnl=net,
            profit_factor=1.1,
            win_rate=0.5,
            avg_win=1.0,
            avg_loss=-1.0,
            max_drawdown=1.0,
            max_intraday_drawdown=1.0,
            worst_day=-1.0,
            longest_losing_streak=1,
            trades_per_day=0.5,
            exposure_time=0.2,
            total_trades=trades,
            slippage_sensitivity=0.2,
            composite_score=0.0,
        )

    monkeypatch.setattr("src.agent_runner.runner.load_all", fake_load_all)
    monkeypatch.setattr("src.agent_runner.runner.run_backtest", fake_run_backtest)
    monkeypatch.setattr("src.agent_runner.runner.score", lambda x: x)
    monkeypatch.setattr(
        "src.agent_runner.runner.check_gates",
        lambda _x: SimpleNamespace(passed=False, failed_gates=["test_gate"]),
    )

    report = ResearchRunner(features_dir=features_dir, reports_dir=reports_dir).run_once()

    assert report["hypotheses_evaluated"] == 1
    result = report["results"]["h_dummy"]
    assert result["n_trades"] == 5
    assert result["net_pnl"] == 4.0


def test_backtester_rejects_multi_symbol_frame() -> None:
    from src.backtester.engine import run_backtest

    class _AnyHyp:
        hypothesis_id = "h_any"

        def generate_signals(self, bars, features):
            return []

    ts = pd.Timestamp("2026-01-02T14:30:00Z")
    df = pd.DataFrame(
        [
            {
                "event_time": ts,
                "available_time": ts + pd.Timedelta(seconds=5),
                "symbol": "SPY",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000,
            },
            {
                "event_time": ts + pd.Timedelta(minutes=5),
                "available_time": ts + pd.Timedelta(minutes=5, seconds=5),
                "symbol": "QQQ",
                "open": 200.0,
                "high": 201.0,
                "low": 199.0,
                "close": 200.5,
                "volume": 1500,
            },
        ]
    )

    import pytest

    with pytest.raises(ValueError, match="single-symbol"):
        run_backtest(df, _AnyHyp())
