from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import Future

import pandas as pd
import pytest

from strategies.hypotheses.h_0007_opening_range_continuation_confirmed import OpeningRangeContinuationConfirmed
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


class _InlineProcessPoolExecutor:
    """Test executor that runs submitted tasks inline for deterministic assertions."""

    def __init__(self, max_workers=None, mp_context=None):
        self.max_workers = max_workers
        self.mp_context = mp_context

    def submit(self, fn, *args, **kwargs):
        fut: Future = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001
            fut.set_exception(exc)
        return fut

    def shutdown(self, wait=False, cancel_futures=False):
        return None


def test_runner_serial_parallel_outputs_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    features_dir = tmp_path / "features"
    reports_dir = tmp_path / "reports"
    features_dir.mkdir(parents=True)

    _feature_df("SPY", days=6).to_parquet(features_dir / "SPY_5m.parquet", index=False)
    _feature_df("QQQ", days=6).to_parquet(features_dir / "QQQ_5m.parquet", index=False)

    monkeypatch.setattr("src.agent_runner.runner.load_all", lambda _p: {"h_a": _HypA()})
    monkeypatch.setattr("src.agent_runner.runner.score", lambda r: r)
    monkeypatch.setattr(
        "src.agent_runner.runner.check_gates",
        lambda _r: SimpleNamespace(passed=False, failed_gates=["gate_example"]),
    )
    monkeypatch.setattr(
        "src.agent_runner.runner.walk_forward_splits",
        lambda df, **_kwargs: iter([(df.iloc[:2].copy(), df.iloc[2:4].copy(), df.iloc[4:].copy())]),
    )
    monkeypatch.setattr("src.agent_runner.runner.ProcessPoolExecutor", _InlineProcessPoolExecutor)
    monkeypatch.setattr(ResearchRunner, "_git_commit_hash", lambda self: "c" * 40)
    monkeypatch.setattr(ResearchRunner, "_git_is_dirty", lambda self: False)

    def fake_run_backtest(df: pd.DataFrame, hyp: _HypA) -> EvalResult:
        net = float(df["close"].sum())
        trades = int(len(df))
        return _eval_for(getattr(hyp, "hypothesis_id", "unknown"), net_pnl=net, trades=trades)

    monkeypatch.setattr("src.agent_runner.runner.run_backtest", fake_run_backtest)

    report_serial = ResearchRunner(
        features_dir=features_dir,
        reports_dir=reports_dir,
        workers=1,
    ).run_once()
    report_parallel = ResearchRunner(
        features_dir=features_dir,
        reports_dir=reports_dir,
        workers=4,
    ).run_once()

    serial_result = report_serial["results"]["h_a"]
    parallel_result = report_parallel["results"]["h_a"]
    assert serial_result["recommended_status"] == parallel_result["recommended_status"]
    assert serial_result["gate_passed"] == parallel_result["gate_passed"]
    assert serial_result["n_trades"] == parallel_result["n_trades"]
    assert serial_result["net_pnl"] == parallel_result["net_pnl"]
    assert serial_result["failed_gates"] == parallel_result["failed_gates"]
    assert serial_result["walk_forward"]["folds"] == parallel_result["walk_forward"]["folds"]
    assert serial_result["walk_forward"]["passed"] == parallel_result["walk_forward"]["passed"]

    serial_symbols = list(report_serial["timings"]["hypotheses"]["h_a"]["symbols"].keys())
    parallel_symbols = list(report_parallel["timings"]["hypotheses"]["h_a"]["symbols"].keys())
    assert serial_symbols == sorted(serial_symbols)
    assert parallel_symbols == sorted(parallel_symbols)
    assert serial_symbols == parallel_symbols


def _feature_df_h0007(symbol: str = "SPY") -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-02T14:30:00Z")
    closes = [100.00, 100.05, 100.10, 100.15, 100.18, 100.35, 100.30]
    minutes = [0, 5, 10, 15, 20, 35, 40]
    rets = [0.0000, 0.0005, 0.0010, 0.0015, 0.0018, 0.0035, 0.0030]
    for i, close in enumerate(closes):
        ts = base + pd.Timedelta(minutes=5 * i)
        rows.append(
            {
                "event_time": ts,
                "available_time": ts + pd.Timedelta(seconds=5),
                "symbol": symbol,
                "open": close - 0.05,
                "high": close + 0.10,
                "low": close - 0.10,
                "close": close,
                "volume": 1000 + i * 10,
                "or_high": 100.20,
                "atr_14": 1.0,
                "atr_pct": 0.006,
                "rvol_20": 1.6,
                "minutes_since_open": float(minutes[i]),
                "session_gap_pct": 0.001,
                "ret_open_to_now": rets[i],
            }
        )
    return pd.DataFrame(rows)


def test_runner_h0007_report_includes_signal_funnel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    features_dir = tmp_path / "features"
    reports_dir = tmp_path / "reports"
    features_dir.mkdir(parents=True)
    _feature_df_h0007("SPY").to_parquet(features_dir / "SPY_5m.parquet", index=False)

    monkeypatch.setattr(
        "src.agent_runner.runner.load_all",
        lambda _p: {"h_0007": OpeningRangeContinuationConfirmed()},
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
        include_hypotheses=["h_0007"],
        workers=1,
    ).run_once()

    result = report["results"]["h_0007"]
    assert "signal_funnel" in result
    funnel = result["signal_funnel"]
    required_keys = {
        "bars_total",
        "opening_range_available",
        "final_signals",
        "executed_trades",
    }
    assert required_keys.issubset(set(funnel))
    assert funnel["executed_trades"] == result["n_trades"]
