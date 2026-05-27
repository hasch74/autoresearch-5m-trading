"""CLI wrapper for syncing the EODHD universe into parquet caches.

This module loads an API token from the environment or from a local `.env`
file, reads the trading universe from `configs/universe.yaml`, downloads raw
and normalized 5-minute bars, and then materializes features for each symbol.
"""

from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

from src.data_ingest.eodhd import sync_universe
from src.feature_store.features import compute_features

_DEFAULT_ENV_PATH = Path(".env")
_DEFAULT_UNIVERSE_PATH = Path("configs/universe.yaml")


def load_env_file(path: Path = _DEFAULT_ENV_PATH) -> dict[str, str]:
    """Load KEY=VALUE pairs from a local env file without extra dependencies."""
    if not path.exists():
        return {}

    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def load_api_key(env_path: Path = _DEFAULT_ENV_PATH) -> str:
    """Resolve the EODHD API token from env vars or a local .env file."""
    env = load_env_file(env_path)
    return (
        os.environ.get("EODHD_API_TOKEN")
        or os.environ.get("EODHD_API_KEY")
        or env.get("EODHD_API_TOKEN")
        or env.get("EODHD_API_KEY")
        or ""
    )


def load_universe(path: Path = _DEFAULT_UNIVERSE_PATH) -> list[str]:
    """Load the MVP universe symbols from YAML."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    symbols = list(data.get("etfs", [])) + list(data.get("stocks", []))
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(symbols))


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def sync_and_build_features(
    *,
    start: date | None,
    end: date | None,
    universe_path: Path,
    raw_dir: Path,
    norm_dir: Path,
    feat_dir: Path,
    env_path: Path,
    rebuild: bool = False,
) -> dict[str, object]:
    """Sync universe data and compute features for each symbol."""
    api_key = load_api_key(env_path)
    if not api_key:
        raise RuntimeError("EODHD API token not found in environment or .env")

    symbols = load_universe(universe_path)
    if rebuild:
        _clear_symbol_caches(symbols, raw_dir=raw_dir, norm_dir=norm_dir, feat_dir=feat_dir)

    sync_results = sync_universe(
        symbols,
        api_key,
        start=start,
        end=end,
        raw_dir=raw_dir,
        norm_dir=norm_dir,
    )

    feature_results: dict[str, int] = {}
    for symbol, frame in sync_results.items():
        if frame.empty:
            feature_results[symbol] = 0
            continue
        feature_frame = compute_features(frame, feat_dir=feat_dir, save=True)
        feature_results[symbol] = int(len(feature_frame))

    return {
        "symbols": symbols,
        "synced_symbols": len(sync_results),
        "feature_rows": feature_results,
    }


def _clear_symbol_caches(symbols: list[str], *, raw_dir: Path, norm_dir: Path, feat_dir: Path) -> None:
    for symbol in symbols:
        for path in (
            raw_dir / f"{symbol}_5m_raw.parquet",
            norm_dir / f"{symbol}_5m.parquet",
            feat_dir / f"{symbol}_5m.parquet",
        ):
            if path.exists():
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync EODHD universe into parquet caches.")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD.")
    parser.add_argument("--universe", type=Path, default=_DEFAULT_UNIVERSE_PATH)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--norm-dir", type=Path, default=Path("data/normalized"))
    parser.add_argument("--feat-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--env-path", type=Path, default=_DEFAULT_ENV_PATH)
    parser.add_argument("--rebuild", action="store_true", help="Delete existing symbol parquet caches first.")
    args = parser.parse_args()

    summary = sync_and_build_features(
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        universe_path=args.universe,
        raw_dir=args.raw_dir,
        norm_dir=args.norm_dir,
        feat_dir=args.feat_dir,
        env_path=args.env_path,
        rebuild=args.rebuild,
    )
    print(summary)


if __name__ == "__main__":
    main()
