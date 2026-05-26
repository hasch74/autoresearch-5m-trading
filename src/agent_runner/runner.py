"""Research loop runner.

Implements the AutoResearch keep-or-revert loop:

  1. Load registered hypotheses from strategies/hypotheses/
  2. For each hypothesis with status < validated_candidate:
     a. Run backtest over train window
     b. Score result (composite_score)
     c. Check gates
     d. If passes: update manifest status → validated_candidate
     e. If fails:  update manifest status → quarantined (but keep the record)
  3. Write a research report to reports/

The agent may call run_once() after creating a new hypothesis file.
This runner is in the agent's allow_write list for reports/ only.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtester.engine import run_backtest
from src.evaluator.scoring import check_gates, score
from src.hypothesis_engine.registry import load_all
from src.types import HypothesisStatus

logger = logging.getLogger(__name__)

_REPORTS_DIR = Path("reports")
_FEATURES_DIR = Path("data/features")


class ResearchRunner:
    """Executes one research cycle: load → backtest → gate → keep/revert.

    Parameters
    ----------
    features_dir:
        Directory containing precomputed <symbol>_5m.parquet feature files.
    reports_dir:
        Directory to write per-run JSON reports.
    hypotheses_dir:
        Directory to scan for hypothesis .py files.
    """

    def __init__(
        self,
        features_dir: Path = _FEATURES_DIR,
        reports_dir: Path = _REPORTS_DIR,
        hypotheses_dir: Path = Path("strategies/hypotheses"),
    ) -> None:
        self.features_dir = features_dir
        self.reports_dir = reports_dir
        self.hypotheses_dir = hypotheses_dir

    def run_once(self) -> dict[str, Any]:
        """Execute one research cycle.

        Returns a summary dict describing outcomes for all evaluated hypotheses.
        """
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        registry = load_all(self.hypotheses_dir)
        if not registry:
            logger.warning("No hypotheses found in %s", self.hypotheses_dir)
            return {"hypotheses_evaluated": 0}

        # Load all available feature files
        feature_frames = self._load_feature_frames()
        if not feature_frames:
            logger.warning("No feature data found in %s; run data_ingest first", self.features_dir)
            return {"hypotheses_evaluated": 0, "warning": "no_feature_data"}

        combined_df = pd.concat(feature_frames.values(), ignore_index=True).sort_values("event_time")

        results = {}
        for hyp_id, hyp in registry.items():
            logger.info("Evaluating %s ...", hyp_id)
            try:
                raw_result = run_backtest(combined_df, hyp)
                scored_result = score(raw_result)
                gate = check_gates(scored_result)

                outcome = {
                    "hypothesis_id": hyp_id,
                    "n_trades": scored_result.total_trades,
                    "net_pnl": scored_result.net_pnl,
                    "composite_score": scored_result.composite_score,
                    "gate_passed": gate.passed,
                    "failed_gates": gate.failed_gates,
                    "recommended_status": (
                        HypothesisStatus.VALIDATED_CANDIDATE.value if gate.passed
                        else HypothesisStatus.QUARANTINED.value
                    ),
                }
                results[hyp_id] = outcome
                logger.info(
                    "%s: score=%.3f gates=%s trades=%d",
                    hyp_id, scored_result.composite_score,
                    "PASS" if gate.passed else "FAIL",
                    scored_result.total_trades,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Error evaluating %s: %s", hyp_id, exc)
                results[hyp_id] = {"hypothesis_id": hyp_id, "error": str(exc)}

        report = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "hypotheses_evaluated": len(results),
            "results": results,
        }
        self._save_report(report)
        return report

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load_feature_frames(self) -> dict[str, pd.DataFrame]:
        frames = {}
        for path in sorted(self.features_dir.glob("*_5m.parquet")):
            symbol = path.name.replace("_5m.parquet", "")
            try:
                frames[symbol] = pd.read_parquet(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load features for %s: %s", symbol, exc)
        return frames

    def _save_report(self, report: dict) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.reports_dir / f"research_run_{ts}.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Report saved to %s", path)
