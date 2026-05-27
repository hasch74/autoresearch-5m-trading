from datetime import datetime

import pandas as pd

from src.data_ingest.coverage import build_coverage_report, summarize_symbol


def _full_rth_day(day: str) -> pd.DataFrame:
    # 09:30-15:55 Eastern, encoded as UTC timestamps.
    idx_local = pd.date_range(f"{day} 09:30", f"{day} 15:55", freq="5min", tz="America/New_York")
    idx_utc = idx_local.tz_convert("UTC")
    return pd.DataFrame({"event_time": idx_utc})


def test_symbol_summary_no_missing_bars_for_full_day() -> None:
    df = _full_rth_day("2026-01-05")

    summary = summarize_symbol("SPY", df)

    assert summary.bars_observed == 78
    assert summary.observed_trading_days == 1
    assert summary.missing_bars_observed_days == 0
    assert summary.out_of_rth_bars == 0
    assert summary.coverage_ratio_observed_days == 1.0


def test_symbol_summary_detects_missing_bar() -> None:
    df = _full_rth_day("2026-01-06").iloc[:-1].copy()

    summary = summarize_symbol("QQQ", df)

    assert summary.bars_observed == 77
    assert summary.expected_bars_observed_days == 78
    assert summary.missing_bars_observed_days == 1
    assert summary.coverage_ratio_observed_days == 0.9872


def test_symbol_summary_detects_missing_business_day_between_observed_days() -> None:
    df = pd.concat([_full_rth_day("2026-01-05"), _full_rth_day("2026-01-07")], ignore_index=True)

    summary = summarize_symbol("IWM", df)

    assert summary.observed_trading_days == 2
    assert summary.missing_business_days == 1
    assert summary.expected_bars_business_span == 234
    assert summary.missing_bars_business_span == 78


def test_build_report_aggregates_symbols(tmp_path) -> None:
    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True)

    _full_rth_day("2026-01-08").to_parquet(features_dir / "SPY_5m.parquet", index=False)
    _full_rth_day("2026-01-08").iloc[:-2].to_parquet(features_dir / "QQQ_5m.parquet", index=False)

    report = build_coverage_report(features_dir)

    assert report["totals"]["symbol_count"] == 2
    assert report["totals"]["bars_observed"] == 154
    assert report["totals"]["expected_bars_observed_days"] == 156
    assert report["totals"]["missing_bars_observed_days"] == 2
    assert isinstance(datetime.fromisoformat(report["generated_at"].replace("Z", "+00:00")), datetime)
