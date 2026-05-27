"""h_0005 — Selective VWAP reversion variant.

Variant hypothesis:
  Trade only cleaner mean-reversion setups where price is below VWAP,
  but avoid panic volume and very early/late-session noise.

This file may be modified by the research agent.
The Hypothesis interface contract (strategies/hypotheses/interfaces.py) must be respected.
"""

from __future__ import annotations

from typing import Sequence

from src.types import Bar, Direction, Signal


class VwapReversionSelective:
    """Lower-frequency VWAP mean-reversion variant."""

    name: str = "vwap_reversion_selective"
    version: str = "0.1"
    hypothesis_id: str = "h_0005"

    entry_atr_band: float = 1.10
    min_rvol: float = 0.85
    max_rvol: float = 1.20
    max_abs_volume_zscore: float = 1.10
    min_minutes_since_open: int = 30
    min_minutes_to_close: int = 35
    min_ret_open_to_now: float = -0.015
    stop_atr_mult: float = 0.9
    take_profit_atr_mult: float = 1.0
    max_hold_bars: int = 4

    def required_features(self) -> Sequence[str]:
        return [
            "vwap",
            "atr_14",
            "rvol_20",
            "volume_zscore_20",
            "minutes_since_open",
            "minutes_to_close",
            "ret_open_to_now",
        ]

    def generate_signals(
        self,
        bars: Sequence[Bar],
        features: dict,
    ) -> Sequence[Signal]:
        if not bars:
            return []

        bar = bars[-1]
        vwap: float | None = features.get("vwap")
        atr: float | None = features.get("atr_14")
        rvol: float | None = features.get("rvol_20")
        volume_zscore: float | None = features.get("volume_zscore_20")
        minutes_since_open: float | None = features.get("minutes_since_open")
        minutes_to_close: float | None = features.get("minutes_to_close")
        ret_open_to_now: float | None = features.get("ret_open_to_now")

        if any(
            v is None
            for v in (
                vwap,
                atr,
                rvol,
                volume_zscore,
                minutes_since_open,
                minutes_to_close,
                ret_open_to_now,
            )
        ):
            return []
        if atr <= 0:
            return []
        if minutes_since_open < self.min_minutes_since_open:
            return []
        if minutes_to_close < self.min_minutes_to_close:
            return []
        if not (self.min_rvol <= rvol <= self.max_rvol):
            return []
        if abs(volume_zscore) > self.max_abs_volume_zscore:
            return []
        if ret_open_to_now < self.min_ret_open_to_now:
            return []

        deviation = (vwap - bar.close) / atr
        if deviation < self.entry_atr_band:
            return []

        confidence = min(1.0, 0.40 + deviation / 2.5)
        return [
            Signal(
                hypothesis_id=self.hypothesis_id,
                symbol=bar.symbol,
                bar_time=bar.event_time,
                direction=Direction.LONG,
                confidence=confidence,
                stop_distance_atr=self.stop_atr_mult,
                take_profit_distance_atr=self.take_profit_atr_mult,
                max_hold_bars=self.max_hold_bars,
            )
        ]
