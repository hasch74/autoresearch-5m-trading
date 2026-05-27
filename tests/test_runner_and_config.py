from pathlib import Path

import yaml

from src.hypothesis_engine.registry import load_all


def test_runner_loads_hypotheses() -> None:
    registry = load_all(Path("strategies/hypotheses"))
    assert "h_0001" in registry
    assert "h_0002" in registry


def test_yaml_configs_parse() -> None:
    for path in Path("configs").glob("*.yaml"):
        yaml.safe_load(path.read_text(encoding="utf-8"))
