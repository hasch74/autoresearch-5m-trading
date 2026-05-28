from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.agent_runner import runner
from src.agent_runner.runner import ResearchRunner
from src.types import EvalResult, HypothesisStatus


class _HypA:
    hypothesis_id = "h_a"


class _HypB:
    hypothesis_id = "h_b"


def _feature_df(symbol: str, days: int = 5) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-01T14:30:00Z")
    for i in range(days):
        ts = base + pd.Timedelta(days=i)
        rows.append(
            {
                "event_time": ts,
                "available_time": ts + pd.Timedelta(seconds=5),
                "symbol": symbol,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 1000 + i,
                "atr_14": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _eval_for(hypothesis_id: str, net_pnl: float = 1.0, trades: int = 3) -> EvalResult:
    return EvalResult(
        hypothesis_id=hypothesis_id,
        run_id="",
        status=HypothesisStatus.DRAFT,
        net_pnl=net_pnl,
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


def test_runner_applies_hypothesis_include_exclude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    features_dir = tmp_path / "features"
    reports_dir = tmp_path / "reports"
    features_dir.mkdir(parents=True)
    _feature_df("SPY", days=1).to_parquet(features_dir / "SPY_5m.parquet", index=False)

    monkeypatch.setattr(
        "src.agent_runner.runner.load_all",
        lambda _p: {"h_a": _HypA(), "h_b": _HypB()},
    )
    monkeypatch.setattr(
        "src.agent_runner.runner.run_backtest",
        lambda _df, hyp: _eval_for(getattr(hyp, "hypothesis_id", "unknown")),
    )
    monkeypatch.setattr("src.agent_runner.runner.score", lambda r: r)
    monkeypatch.setattr(
        "src.agent_runner.runner.check_gates",
        lambda _r: SimpleNamespace(passed=True, failed_gates=[]),
    )
    monkeypatch.setattr(
        "src.agent_runner.runner.walk_forward_splits",
        lambda df, **_kwargs: iter([(df.copy(), df.copy(), df.copy())]),
    )

    report = ResearchRunner(
        features_dir=features_dir,
        reports_dir=reports_dir,
        include_hypotheses=["h_a", "h_b"],
        exclude_hypotheses=["h_b"],
    ).run_once()

    assert report["hypotheses_evaluated"] == 1
    assert list(report["results"].keys()) == ["h_a"]
    assert report["provenance"]["execution"]["include_hypotheses"] == ["h_a", "h_b"]
    assert report["provenance"]["execution"]["exclude_hypotheses"] == ["h_b"]


def test_runner_raises_on_unknown_included_hypothesis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    features_dir = tmp_path / "features"
    reports_dir = tmp_path / "reports"
    features_dir.mkdir(parents=True)

    monkeypatch.setattr("src.agent_runner.runner.load_all", lambda _p: {"h_a": _HypA()})

    with pytest.raises(ValueError, match="Unknown hypotheses requested"):
        ResearchRunner(
            features_dir=features_dir,
            reports_dir=reports_dir,
            include_hypotheses=["h_missing"],
        ).run_once()


def test_runner_applies_symbol_filter_and_max_days(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    features_dir = tmp_path / "features"
    reports_dir = tmp_path / "reports"
    features_dir.mkdir(parents=True)

    _feature_df("SPY", days=5).to_parquet(features_dir / "SPY_5m.parquet", index=False)
    _feature_df("QQQ", days=5).to_parquet(features_dir / "QQQ_5m.parquet", index=False)

    monkeypatch.setattr("src.agent_runner.runner.load_all", lambda _p: {"h_a": _HypA()})

    seen_lengths: list[int] = []

    def fake_run_backtest(df: pd.DataFrame, hyp: _HypA) -> EvalResult:
        assert df["symbol"].nunique() == 1
        assert df["symbol"].iloc[0] == "SPY"
        seen_lengths.append(len(df))
        return _eval_for(hyp.hypothesis_id)

    monkeypatch.setattr("src.agent_runner.runner.run_backtest", fake_run_backtest)
    monkeypatch.setattr("src.agent_runner.runner.score", lambda r: r)
    monkeypatch.setattr(
        "src.agent_runner.runner.check_gates",
        lambda _r: SimpleNamespace(passed=True, failed_gates=[]),
    )
    monkeypatch.setattr(
        "src.agent_runner.runner.walk_forward_splits",
        lambda df, **_kwargs: iter([(df.copy(), df.copy(), df.copy())]),
    )

    report = ResearchRunner(
        features_dir=features_dir,
        reports_dir=reports_dir,
        symbols=["SPY"],
        max_days=2,
    ).run_once()

    assert report["provenance"]["data"]["symbol_count"] == 1
    assert set(report["provenance"]["data"]["symbol_coverage"]) == {"SPY"}
    assert report["provenance"]["execution"]["symbols"] == ["SPY"]
    assert report["provenance"]["execution"]["max_days"] == 2
    assert seen_lengths
    assert all(length <= 3 for length in seen_lengths)


def test_main_applies_fast_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_once(self):
            return {
                "run_at": "2026-01-01T00:00:00+00:00",
                "hypotheses_evaluated": 0,
                "timings": {"total_seconds": 0.0, "workers": captured.get("workers")},
            }

    monkeypatch.setattr(runner, "ResearchRunner", _FakeRunner)
    monkeypatch.setattr("sys.argv", ["runner.py", "--fast"])

    runner.main()

    assert captured["fast_mode"] is True
    assert captured["workers"] == 4
    assert captured["max_days"] == 120
    assert captured["symbols"] == ["SPY", "QQQ"]


def test_main_respects_explicit_overrides_in_fast_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_once(self):
            return {
                "run_at": "2026-01-01T00:00:00+00:00",
                "hypotheses_evaluated": 0,
                "timings": {"total_seconds": 0.0, "workers": captured.get("workers")},
            }

    monkeypatch.setattr(runner, "ResearchRunner", _FakeRunner)
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner.py",
            "--fast",
            "--workers",
            "8",
            "--symbols",
            "NVDA",
            "--max-days",
            "30",
            "--hypotheses",
            "h_0006",
            "--exclude-hypotheses",
            "h_0005",
        ],
    )

    runner.main()

    assert captured["fast_mode"] is True
    assert captured["workers"] == 8
    assert captured["max_days"] == 30
    assert captured["symbols"] == ["NVDA"]
    assert captured["include_hypotheses"] == ["h_0006"]
    assert captured["exclude_hypotheses"] == ["h_0005"]
