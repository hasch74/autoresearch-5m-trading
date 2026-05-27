from pathlib import Path

import pandas as pd

from src.data_ingest import eodhd_sync
from src.data_ingest.eodhd import _normalise


def test_load_env_file_parses_token(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("EODHD_API_TOKEN=abc123\n# comment\n", encoding="utf-8")

    env = eodhd_sync.load_env_file(env_path)

    assert env["EODHD_API_TOKEN"] == "abc123"


def test_normalise_uses_bar_open_timestamps_and_delays_availability() -> None:
    raw = pd.DataFrame(
        [
            {
                "datetime": 1735828200,  # 2025-01-02 14:30:00 UTC / 09:30 ET
                "open": 100.0,
                "high": 101.0,
                "low": 99.5,
                "close": 100.5,
                "volume": 1000,
            }
        ]
    )

    norm = _normalise(raw, "SPY")

    assert norm.loc[0, "event_time"] == pd.Timestamp("2025-01-02T14:30:00Z")
    assert norm.loc[0, "available_time"] == pd.Timestamp("2025-01-02T14:35:05Z")
    assert norm.loc[0, "symbol"] == "SPY"


def test_load_universe_combines_and_deduplicates(tmp_path: Path) -> None:
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text(
        """
etfs: [SPY, QQQ]
stocks: [AAPL, SPY]
""",
        encoding="utf-8",
    )

    symbols = eodhd_sync.load_universe(universe_path)

    assert symbols == ["SPY", "QQQ", "AAPL"]


def test_sync_and_build_features_orchestrates_sync_and_features(tmp_path: Path, monkeypatch) -> None:
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text(
        """
etfs: [SPY]
stocks: [AAPL]
""",
        encoding="utf-8",
    )
    env_path = tmp_path / ".env"
    env_path.write_text("EODHD_API_TOKEN=abc123\n", encoding="utf-8")

    frame = pd.DataFrame(
        [
            {
                "event_time": pd.Timestamp("2026-01-05T14:30:00Z"),
                "available_time": pd.Timestamp("2026-01-05T14:30:05Z"),
                "symbol": "SPY",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000,
            }
        ]
    )

    sync_calls = {}

    def fake_sync_universe(symbols, api_key, start=None, end=None, raw_dir=None, norm_dir=None):
        sync_calls["symbols"] = list(symbols)
        sync_calls["api_key"] = api_key
        sync_calls["start"] = start
        sync_calls["end"] = end
        return {symbol: frame.copy() for symbol in symbols}

    feature_calls = []

    def fake_compute_features(df, *, feat_dir, save):
        feature_calls.append((int(len(df)), feat_dir, save))
        return df.assign(feature_flag=1)

    monkeypatch.setattr(eodhd_sync, "sync_universe", fake_sync_universe)
    monkeypatch.setattr(eodhd_sync, "compute_features", fake_compute_features)

    summary = eodhd_sync.sync_and_build_features(
        start=eodhd_sync._parse_date("2025-01-01"),
        end=eodhd_sync._parse_date("2025-01-31"),
        universe_path=universe_path,
        raw_dir=tmp_path / "raw",
        norm_dir=tmp_path / "normalized",
        feat_dir=tmp_path / "features",
        env_path=env_path,
    )

    assert sync_calls["symbols"] == ["SPY", "AAPL"]
    assert sync_calls["api_key"] == "abc123"
    assert summary["symbols"] == ["SPY", "AAPL"]
    assert summary["synced_symbols"] == 2
    assert feature_calls == [(1, tmp_path / "features", True), (1, tmp_path / "features", True)]


def test_sync_and_build_features_rebuild_clears_existing_files(tmp_path: Path, monkeypatch) -> None:
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text(
        """
etfs: [SPY]
stocks: []
""",
        encoding="utf-8",
    )
    env_path = tmp_path / ".env"
    env_path.write_text("EODHD_API_TOKEN=abc123\n", encoding="utf-8")

    raw_dir = tmp_path / "raw"
    norm_dir = tmp_path / "normalized"
    feat_dir = tmp_path / "features"
    raw_dir.mkdir()
    norm_dir.mkdir()
    feat_dir.mkdir()
    (raw_dir / "SPY_5m_raw.parquet").write_text("old", encoding="utf-8")
    (norm_dir / "SPY_5m.parquet").write_text("old", encoding="utf-8")
    (feat_dir / "SPY_5m.parquet").write_text("old", encoding="utf-8")

    frame = pd.DataFrame(
        [
            {
                "event_time": pd.Timestamp("2026-01-05T14:30:00Z"),
                "available_time": pd.Timestamp("2026-01-05T14:30:05Z"),
                "symbol": "SPY",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000,
            }
        ]
    )

    def fake_sync_universe(symbols, api_key, start=None, end=None, raw_dir=None, norm_dir=None):
        return {"SPY": frame.copy()}

    monkeypatch.setattr(eodhd_sync, "sync_universe", fake_sync_universe)
    monkeypatch.setattr(eodhd_sync, "compute_features", lambda df, *, feat_dir, save: df)

    eodhd_sync.sync_and_build_features(
        start=eodhd_sync._parse_date("2025-01-01"),
        end=eodhd_sync._parse_date("2025-01-31"),
        universe_path=universe_path,
        raw_dir=raw_dir,
        norm_dir=norm_dir,
        feat_dir=feat_dir,
        env_path=env_path,
        rebuild=True,
    )

    assert not (raw_dir / "SPY_5m_raw.parquet").exists()
    assert not (norm_dir / "SPY_5m.parquet").exists()
    assert not (feat_dir / "SPY_5m.parquet").exists()
