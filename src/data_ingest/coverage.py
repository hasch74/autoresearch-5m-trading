"""Data coverage report for 5-minute feature files.

Computes per-symbol coverage metrics from ``data/features/*_5m.parquet``:
- observed bars
- expected bars for observed trading dates (78 bars/day)
- missing bars within observed dates
- missing business days in the date span (holiday-sensitive heuristic)
- out-of-RTH bar count

Usage:
    python -m src.data_ingest.coverage --features-dir data/features
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_RTH_TZ = "America/New_York"
_BARS_PER_RTH_DAY = 78


@dataclass(frozen=True)
class SymbolCoverage:
    symbol: str
    bars_observed: int
    date_start: str | None
    date_end: str | None
    observed_trading_days: int
    expected_bars_observed_days: int
    missing_bars_observed_days: int
    expected_bars_business_span: int
    missing_bars_business_span: int
    missing_business_days: int
    out_of_rth_bars: int
    coverage_ratio_observed_days: float


def build_coverage_report(features_dir: Path) -> dict:
    files = sorted(features_dir.glob("*_5m.parquet"))
    symbols: list[SymbolCoverage] = []

    for path in files:
        symbol = path.name.replace("_5m.parquet", "")
        frame = pd.read_parquet(path)
        symbols.append(summarize_symbol(symbol, frame))

    total_observed = sum(s.bars_observed for s in symbols)
    total_expected_observed_days = sum(s.expected_bars_observed_days for s in symbols)
    total_missing_observed_days = sum(s.missing_bars_observed_days for s in symbols)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features_dir": str(features_dir),
        "bars_per_rth_day": _BARS_PER_RTH_DAY,
        "symbols": [asdict(s) for s in symbols],
        "totals": {
            "symbol_count": len(symbols),
            "bars_observed": total_observed,
            "expected_bars_observed_days": total_expected_observed_days,
            "missing_bars_observed_days": total_missing_observed_days,
            "coverage_ratio_observed_days": _safe_ratio(total_observed, total_expected_observed_days),
        },
        "notes": [
            "missing_business_days uses pandas business-day calendar and may count market holidays.",
            "missing_bars_observed_days is stricter and checks only dates with at least one bar.",
        ],
    }


def summarize_symbol(symbol: str, frame: pd.DataFrame) -> SymbolCoverage:
    if frame.empty or "event_time" not in frame.columns:
        return SymbolCoverage(
            symbol=symbol,
            bars_observed=0,
            date_start=None,
            date_end=None,
            observed_trading_days=0,
            expected_bars_observed_days=0,
            missing_bars_observed_days=0,
            expected_bars_business_span=0,
            missing_bars_business_span=0,
            missing_business_days=0,
            out_of_rth_bars=0,
            coverage_ratio_observed_days=0.0,
        )

    event_time = pd.to_datetime(frame["event_time"], utc=True, errors="coerce").dropna().drop_duplicates()
    if event_time.empty:
        return SymbolCoverage(
            symbol=symbol,
            bars_observed=0,
            date_start=None,
            date_end=None,
            observed_trading_days=0,
            expected_bars_observed_days=0,
            missing_bars_observed_days=0,
            expected_bars_business_span=0,
            missing_bars_business_span=0,
            missing_business_days=0,
            out_of_rth_bars=0,
            coverage_ratio_observed_days=0.0,
        )

    eastern = event_time.dt.tz_convert(_RTH_TZ)
    observed_days = sorted(eastern.dt.date.unique())
    observed_day_count = len(observed_days)

    expected_on_observed_days = observed_day_count * _BARS_PER_RTH_DAY
    out_of_rth = _count_out_of_rth(eastern)
    missing_on_observed_days = max(0, expected_on_observed_days - len(event_time) + out_of_rth)

    start_day = min(observed_days)
    end_day = max(observed_days)
    business_days = pd.bdate_range(start_day, end_day)
    expected_on_business_span = len(business_days) * _BARS_PER_RTH_DAY
    missing_on_business_span = max(0, expected_on_business_span - len(event_time) + out_of_rth)
    missing_business_days = max(0, len(business_days) - observed_day_count)

    return SymbolCoverage(
        symbol=symbol,
        bars_observed=int(len(event_time)),
        date_start=pd.Timestamp(event_time.min()).isoformat(),
        date_end=pd.Timestamp(event_time.max()).isoformat(),
        observed_trading_days=observed_day_count,
        expected_bars_observed_days=expected_on_observed_days,
        missing_bars_observed_days=missing_on_observed_days,
        expected_bars_business_span=expected_on_business_span,
        missing_bars_business_span=missing_on_business_span,
        missing_business_days=missing_business_days,
        out_of_rth_bars=out_of_rth,
        coverage_ratio_observed_days=round(_safe_ratio(len(event_time) - out_of_rth, expected_on_observed_days), 4),
    )


def _count_out_of_rth(eastern_times: pd.Series) -> int:
    tod = eastern_times.dt.time
    open_t = pd.Timestamp("09:30").time()
    close_t = pd.Timestamp("16:00").time()
    in_rth = (tod >= open_t) & (tod < close_t)
    return int((~in_rth).sum())


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data coverage report for 5-minute feature files.")
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write JSON output.")
    args = parser.parse_args()

    report = build_coverage_report(args.features_dir)
    text = json.dumps(report, indent=2)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
