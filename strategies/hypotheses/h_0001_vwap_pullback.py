"""h_0001 — VWAP Pullback with Elevated Relative Volume.

Edge hypothesis:
  After a gapped-up open, price pulls back to VWAP with high relative volume.
  When the bar closes at or just above VWAP, enter long.
  Exit after max_hold_bars or on stop/take-profit.

This file may be modified by the research agent.
The Hypothesis interface contract (strategies/hypotheses/interfaces.py) must be respected.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from src.types import Bar, Direction, Signal


class VwapPullbackRvol:
    """VWAP Pullback with elevated relative volume — version 0.1."""

    name: str = "vwap_pullback_rvol"
    version: str = "0.1"
    hypothesis_id: str = "h_0001"

    # --- parameters (agent may tune these) ---
    rvol_min: float = 1.5
    gap_pct_min: float = 0.003
    max_spread_atr_fraction: float = 0.20
    stop_atr_mult: float = 1.5
    take_profit_atr_mult: float = 3.0
    max_hold_bars: int = 6

    def required_features(self) -> Sequence[str]:
        return ["vwap", "rvol_20", "atr_14", "gap_pct", "ret_open_to_now"]

    def generate_signals(
        self,
        bars: Sequence[Bar],
        features: dict,
    ) -> Sequence[Signal]:
        """Return signals for the latest bar.

        Parameters
        ----------
        bars:
            Sequence of Bar objects ending with the most recent bar.
            At least 1 bar required.
        features:
            Mapping of feature name -> float value for the latest bar.
            Must contain all keys from required_features().

        Returns
        -------
        Sequence of Signal objects (empty if no entry condition is met).
        """
        if not bars:
            return []

        bar = bars[-1]

        vwap: float | None = features.get("vwap")
        rvol: float | None = features.get("rvol_20")
        atr: float | None = features.get("atr_14")
        gap_pct: float | None = features.get("gap_pct")

        # Require all features present
        if any(v is None for v in (vwap, rvol, atr, gap_pct)):
            return []

        # Gate 1: stock must have gapped up
        if gap_pct < self.gap_pct_min:
            return []

        # Gate 2: price must be at or just above VWAP (pullback touched VWAP)
        if bar.close < vwap:
            return []
        if bar.close > vwap + 0.10 * atr:  # too far above VWAP — not a pullback
            return []

        # Gate 3: elevated relative volume
        if rvol < self.rvol_min:
            return []

        # Gate 4: spread / ATR check (proxy: ATR must be > 0 and reasonable)
        if atr <= 0:
            return []

        signal = Signal(
            hypothesis_id=self.hypothesis_id,
            symbol=bar.symbol,
            bar_time=bar.event_time,
            direction=Direction.LONG,
            confidence=min(1.0, (rvol - self.rvol_min) / 3.0 + 0.5),
            stop_distance_atr=self.stop_atr_mult,
            take_profit_distance_atr=self.take_profit_atr_mult,
            max_hold_bars=self.max_hold_bars,
        )
        return [signal]
