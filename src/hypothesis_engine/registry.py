"""Hypothesis registry: discovers, loads, and manages hypothesis objects.

Scans strategies/hypotheses/ for .py files matching h_XXXX_*.py,
imports each module, finds the class implementing the Hypothesis Protocol,
and returns a registry mapping hypothesis_id → instance.

The agent creates new files in strategies/hypotheses/. The registry
picks them up on the next scan without any manual registration.
"""

from __future__ import annotations

import importlib.util
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HYPOTHESES_DIR = Path("strategies/hypotheses")
_HYP_PATTERN = re.compile(r"^(h_\d{4}_\w+)\.py$")


def load_all(hypotheses_dir: Path = _HYPOTHESES_DIR) -> dict[str, Any]:
    """Scan *hypotheses_dir* and return {hypothesis_id: instance} for all valid files.

    A valid hypothesis module must:
    - match h_XXXX_*.py filename pattern
    - define a class with a ``hypothesis_id`` class attribute
    - define ``generate_signals(bars, features)`` method
    """
    registry: dict[str, Any] = {}
    for path in sorted(hypotheses_dir.glob("h_*.py")):
        m = _HYP_PATTERN.match(path.name)
        if not m:
            continue
        try:
            instance = _load_hypothesis(path)
            if instance is not None:
                hyp_id = instance.hypothesis_id
                registry[hyp_id] = instance
                logger.debug("Loaded hypothesis: %s from %s", hyp_id, path.name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load hypothesis from %s: %s", path.name, exc)
    return registry


def _load_hypothesis(path: Path) -> Any | None:
    """Import *path* as a module and return an instance of its hypothesis class."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    for name in dir(mod):
        obj = getattr(mod, name)
        if (
            isinstance(obj, type)
            and hasattr(obj, "hypothesis_id")
            and hasattr(obj, "generate_signals")
            and not name.startswith("_")
        ):
            return obj()
    return None
