"""Evaluator: compute composite score and check evaluation gates.

Gates come from research_policy.yaml / configs/risk.yaml.
This module is in the agent's DENY_WRITE list — the agent must not modify it.

Composite score formula (equal-weighted average of normalized sub-scores):
  1. net_pnl_score      = clamp(net_pnl / net_pnl_target, 0, 1)
  2. profit_factor_score = clamp((profit_factor - 1) / 1, 0, 1)   [pf=2 → 1.0]
  3. win_rate_score      = clamp(win_rate / 0.6, 0, 1)
  4. drawdown_score      = clamp(1 - max_drawdown / drawdown_cap, 0, 1)
  5. activity_score      = clamp(trades_per_day / trades_per_day_target, 0, 1)
  6. slippage_score      = clamp(1 - slippage_sensitivity / 2.0, 0, 1)

composite_score ∈ [0, 1]; threshold for promotion defined in risk.yaml.
"""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

from src.types import EvalResult, HypothesisStatus


class GateResult(NamedTuple):
    passed: bool
    failed_gates: list[str]
    composite_score: float


# Default thresholds — must match configs/risk.yaml promotion_thresholds
_DEFAULTS = {
    "min_trades": 30,
    "min_win_rate": 0.40,
    "min_profit_factor": 1.10,
    "max_drawdown_pct": 0.10,    # 10% of starting equity
    "starting_equity": 10_000.0,
    "min_net_pnl": 0.0,
    "max_slippage_sensitivity": 1.0,
    "min_trades_per_day": 0.5,
    "net_pnl_target": 100.0,     # benchmark net PnL for full score
    "trades_per_day_target": 3.0,
    "drawdown_cap": 200.0,       # max expected drawdown for normalisation
}


def score(result: EvalResult, thresholds: dict | None = None) -> EvalResult:
    """Compute composite_score and return updated EvalResult."""
    t = {**_DEFAULTS, **(thresholds or {})}
    cs = _composite(result, t)
    return replace(result, composite_score=round(cs, 4))


def check_gates(result: EvalResult, thresholds: dict | None = None) -> GateResult:
    """Check all evaluation gates. Returns GateResult with pass/fail detail."""
    t = {**_DEFAULTS, **(thresholds or {})}
    failed: list[str] = []

    if result.total_trades < t["min_trades"]:
        failed.append(f"min_trades: {result.total_trades} < {t['min_trades']}")
    if result.win_rate < t["min_win_rate"]:
        failed.append(f"min_win_rate: {result.win_rate:.3f} < {t['min_win_rate']}")
    if result.profit_factor < t["min_profit_factor"]:
        failed.append(f"min_profit_factor: {result.profit_factor:.3f} < {t['min_profit_factor']}")
    if result.max_drawdown > t["max_drawdown_pct"] * t["starting_equity"]:
        failed.append(f"max_drawdown: {result.max_drawdown:.3f} > limit")
    if result.net_pnl < t["min_net_pnl"]:
        failed.append(f"min_net_pnl: {result.net_pnl:.3f} < {t['min_net_pnl']}")
    if result.slippage_sensitivity > t["max_slippage_sensitivity"]:
        failed.append(f"slippage_sensitivity: {result.slippage_sensitivity:.3f} > {t['max_slippage_sensitivity']}")
    if result.trades_per_day < t["min_trades_per_day"]:
        failed.append(f"min_trades_per_day: {result.trades_per_day:.3f} < {t['min_trades_per_day']}")

    cs = _composite(result, t)
    return GateResult(passed=len(failed) == 0, failed_gates=failed, composite_score=round(cs, 4))


def promote_status(result: EvalResult, gate: GateResult) -> HypothesisStatus:
    """Suggest next lifecycle status based on gate results."""
    if not gate.passed:
        return HypothesisStatus.QUARANTINED
    if gate.composite_score >= 0.7:
        return HypothesisStatus.VALIDATED_CANDIDATE
    return HypothesisStatus.BACKTEST_CANDIDATE


# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _composite(r: EvalResult, t: dict) -> float:
    s1 = _clamp(r.net_pnl / t["net_pnl_target"])
    s2 = _clamp((r.profit_factor - 1.0) / 1.0)
    s3 = _clamp(r.win_rate / 0.6)
    s4 = _clamp(1.0 - r.max_drawdown / t["drawdown_cap"])
    s5 = _clamp(r.trades_per_day / t["trades_per_day_target"])
    s6 = _clamp(1.0 - r.slippage_sensitivity / 2.0)
    return (s1 + s2 + s3 + s4 + s5 + s6) / 6.0
