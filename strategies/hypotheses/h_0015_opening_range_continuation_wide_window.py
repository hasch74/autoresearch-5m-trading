"""h_0015 - Opening-range continuation with wider morning window.

Single-factor variant based on h_0008:
  - Keep all RVOL/context/trigger thresholds unchanged.
  - Only widen allowed time window to test opportunity scarcity.
"""

from __future__ import annotations

from typing import Sequence

from src.types import Bar, Direction, Signal


class OpeningRangeContinuationWideWindow:
    """Opening-range continuation with a wider intraday execution window."""

    name: str = "opening_range_continuation_wide_window"
    version: str = "0.1"
    hypothesis_id: str = "h_0015"

    min_minutes_since_open: int = 10
    max_minutes_since_open: int = 150
    min_rvol: float = 1.15
    min_breakout_buffer_atr: float = 0.06
    max_extension_above_or_atr: float = 0.55
    min_session_gap_pct: float = 0.0
    min_ret_open_to_now: float = 0.002
    min_atr_pct: float = 0.002
    max_atr_pct: float = 0.015
    stop_atr_mult: float = 0.85
    take_profit_atr_mult: float = 1.55
    max_hold_bars: int = 8

    _FUNNEL_KEYS: tuple[str, ...] = (
        "bars_total",
        "opening_range_available",
        "in_time_window",
        "rvol_pass",
        "atr_volatility_pass",
        "gap_filter_pass",
        "ret_open_to_now_pass",
        "breakout_up",
        "extension_pass",
        "final_signals",
    )

    def __init__(self) -> None:
        self.reset_signal_funnel()

    def reset_signal_funnel(self) -> None:
        self._signal_funnel = {k: 0 for k in self._FUNNEL_KEYS}

    def get_signal_funnel(self) -> dict[str, int]:
        return dict(self._signal_funnel)

    def _bump_funnel(self, key: str) -> None:
        self._signal_funnel[key] = self._signal_funnel.get(key, 0) + 1

    def required_features(self) -> Sequence[str]:
        return [
            "or_high",
            "atr_14",
            "atr_pct",
            "rvol_20",
            "minutes_since_open",
            "session_gap_pct",
            "ret_open_to_now",
        ]

    def generate_signals(self, bars: Sequence[Bar], features: dict) -> Sequence[Signal]:
        self._bump_funnel("bars_total")
        if len(bars) < 2:
            return []

        bar = bars[-1]
        prev_bar = bars[-2]

        or_high: float | None = features.get("or_high")
        atr: float | None = features.get("atr_14")
        atr_pct: float | None = features.get("atr_pct")
        rvol: float | None = features.get("rvol_20")
        minutes_since_open: float | None = features.get("minutes_since_open")
        session_gap_pct: float | None = features.get("session_gap_pct")
        ret_open_to_now: float | None = features.get("ret_open_to_now")

        if or_high is not None:
            self._bump_funnel("opening_range_available")

        if any(v is None for v in (or_high, atr, atr_pct, rvol, minutes_since_open, session_gap_pct, ret_open_to_now)):
            return []
        if atr <= 0:
            return []

        if not (self.min_minutes_since_open <= minutes_since_open <= self.max_minutes_since_open):
            return []
        self._bump_funnel("in_time_window")
        if rvol < self.min_rvol:
            return []
        self._bump_funnel("rvol_pass")
        if not (self.min_atr_pct <= atr_pct <= self.max_atr_pct):
            return []
        self._bump_funnel("atr_volatility_pass")

        if session_gap_pct < self.min_session_gap_pct:
            return []
        self._bump_funnel("gap_filter_pass")
        if ret_open_to_now < self.min_ret_open_to_now:
            return []
        self._bump_funnel("ret_open_to_now_pass")

        if prev_bar.close >= or_high:
            return []

        breakout_buffer = self.min_breakout_buffer_atr * atr
        if bar.close <= or_high + breakout_buffer:
            return []
        self._bump_funnel("breakout_up")

        extension_atr = (bar.close - or_high) / atr
        if extension_atr > self.max_extension_above_or_atr:
            return []
        self._bump_funnel("extension_pass")

        confidence = min(1.0, 0.45 + 0.12 * extension_atr + 0.08 * max(0.0, rvol - self.min_rvol))
        self._bump_funnel("final_signals")
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
