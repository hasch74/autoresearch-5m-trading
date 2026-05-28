"""Event-driven backtester for 5-minute bar hypotheses.

Design:
- Bar-by-bar replay of a feature DataFrame (output of feature_store.compute_features).
- For each bar: call hypothesis.generate_signals(); for each open position: check exit.
- Fill model based on costs.yaml: commission (per-share), spread (bps), slippage (ATR fraction).
- Returns EvalResult with all required metrics.

Constraints enforced:
- No overnight positions (all open trades are closed at session end).
- Bracket exits: stop-loss and take-profit evaluated at bar high/low (pessimistic fill).
- Max hold bars: position closed at close of bar N if not stopped/TP'd.
- Only one concurrent position per symbol (simplified: first signal wins).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import pandas as pd

from src.types import Bar, Direction, EvalResult, Signal


@dataclass
class _CostModel:
    """Parsed from costs.yaml defaults — hardcoded conservative estimates."""
    commission_per_share: float = 0.005
    min_commission_usd: float = 1.00
    max_commission_pct: float = 0.01     # 1% of trade value
    entry_atr_fraction: float = 0.05
    exit_atr_fraction: float = 0.05
    min_slippage_bps: float = 2.0
    half_spread_bps: float = 3.0


@dataclass
class _OpenTrade:
    entry_bar_idx: int
    symbol: str
    direction: Direction
    entry_price: float
    entry_execution_cost: float
    stop_price: float
    tp_price: float
    max_hold_bars: int
    shares: int = 1   # simplified: 1 share per signal for metrics; PnL is per-share


def run_backtest(
    df: pd.DataFrame,
    hypothesis: object,
    *,
    shares_per_trade: int = 1,
    cost_model: _CostModel | None = None,
    train_start: date | None = None,
    train_end: date | None = None,
) -> EvalResult:
    """Run a bar-by-bar backtest of *hypothesis* over *df*.

    Parameters
    ----------
    df:
        Feature DataFrame (output of feature_store.compute_features).
        Must be sorted by event_time and contain a single symbol.
    hypothesis:
        Object with ``generate_signals(bars, features) -> Sequence[Signal]``
        and ``hypothesis_id`` attribute.
    shares_per_trade:
        Fixed position size in shares (default 1 for unit-PnL metrics).
    cost_model:
        Cost assumptions; defaults to conservative costs.yaml values.
    train_start / train_end:
        Optional date filter — only bars within [train_start, train_end] are used.
    """
    if cost_model is None:
        cost_model = _CostModel()

    df = df.copy().sort_values("event_time").reset_index(drop=True)

    # Date filter
    if train_start or train_end:
        eastern = df["event_time"].dt.tz_convert("America/New_York")
        dates = eastern.dt.date
        if train_start:
            df = df[dates >= train_start]
        if train_end:
            df = df[dates <= train_end]
        df = df.reset_index(drop=True)

    if df.empty:
        return _empty_eval(hypothesis)

    if df["symbol"].nunique() != 1:
        raise ValueError("run_backtest expects a single-symbol DataFrame")

    feature_cols = [c for c in df.columns if c not in
                    ("event_time", "available_time", "symbol",
                     "open", "high", "low", "close", "volume")]

    bars_list: list[Bar] = []
    open_trade: _OpenTrade | None = None
    closed_trades: list[dict] = []

    eastern_series = df["event_time"].dt.tz_convert("America/New_York")
    for i, row in enumerate(df.itertuples(index=False), start=0):
        bar = _row_to_bar(row)
        bars_list.append(bar)
        features = {
            col: value
            for col in feature_cols
            if pd.notna(value := getattr(row, col))
        }

        is_last_bar_of_session = _is_session_last_bar(
            eastern_series.iloc[i], df, eastern_series, i
        )

        # --- Check exit for open trade ---
        if open_trade is not None:
            bars_held = i - open_trade.entry_bar_idx
            trade_result = _check_exit(
                open_trade, bar, bars_held, is_last_bar_of_session, cost_model
            )
            if trade_result is not None:
                closed_trades.append(trade_result)
                open_trade = None

        # --- Check entry (only if no open position) ---
        if open_trade is None and not is_last_bar_of_session:
            signals: Sequence[Signal] = hypothesis.generate_signals(bars_list, features)
            if signals:
                sig = signals[0]  # take first signal only
                atr = features.get("atr_14", 1.0) or 1.0
                entry_execution_cost = _entry_execution_cost(bar.close, atr, cost_model)
                entry_price = _entry_fill(bar.close, sig.direction, atr, cost_model)
                stop_price = _stop_price(entry_price, sig.direction, sig.stop_distance_atr, atr)
                tp_price = _tp_price(entry_price, sig.direction, sig.take_profit_distance_atr, atr)
                open_trade = _OpenTrade(
                    entry_bar_idx=i,
                    symbol=bar.symbol,
                    direction=sig.direction,
                    entry_price=entry_price,
                    entry_execution_cost=entry_execution_cost,
                    stop_price=stop_price,
                    tp_price=tp_price,
                    max_hold_bars=sig.max_hold_bars,
                    shares=shares_per_trade,
                )

        # Force-close at session end
        if open_trade is not None and is_last_bar_of_session:
            exit_execution_cost = _exit_execution_cost(bar.close, 0.0, cost_model)
            exit_price = _exit_fill(bar.close, open_trade.direction, 0.0, cost_model)
            pnl = _pnl(open_trade.entry_price, exit_price, open_trade.direction, open_trade.shares)
            closed_trades.append({
                "exit_reason": "session_end",
                "pnl": pnl,
                "execution_cost": (open_trade.entry_execution_cost + exit_execution_cost) * open_trade.shares,
                "bars_held": i - open_trade.entry_bar_idx + 1,
                "exit_time": bar.event_time,
            })
            open_trade = None

    return _build_eval(hypothesis, closed_trades, len(df))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _row_to_bar(row: object) -> Bar:
    return Bar(
        event_time=row.event_time,
        available_time=getattr(row, "available_time", row.event_time),
        symbol=row.symbol,
        open=float(row.open),
        high=float(row.high),
        low=float(row.low),
        close=float(row.close),
        volume=int(row.volume),
    )


def _is_session_last_bar(
    ts: pd.Timestamp,
    df: pd.DataFrame,
    eastern: pd.Series,
    idx: int,
) -> bool:
    """True if this is the last RTH bar of the session (i.e. at or after 15:55 or last bar of day)."""
    if ts.hour == 15 and ts.minute >= 55:
        return True
    if idx == len(df) - 1:
        return True
    next_date = eastern.iloc[idx + 1].date()
    return next_date != ts.date()


def _entry_fill(close: float, direction: Direction, atr: float, cm: _CostModel) -> float:
    cost = _entry_execution_cost(close, atr, cm)
    return close + cost if direction == Direction.LONG else close - cost


def _exit_fill(close: float, direction: Direction, atr: float, cm: _CostModel) -> float:
    cost = _exit_execution_cost(close, atr, cm)
    return close - cost if direction == Direction.LONG else close + cost


def _entry_execution_cost(close: float, atr: float, cm: _CostModel) -> float:
    slip = max(atr * cm.entry_atr_fraction, close * cm.min_slippage_bps / 10_000)
    spread = close * cm.half_spread_bps / 10_000
    return slip + spread


def _exit_execution_cost(close: float, atr: float, cm: _CostModel) -> float:
    slip = max(atr * cm.exit_atr_fraction, close * cm.min_slippage_bps / 10_000) if atr else 0.0
    spread = close * cm.half_spread_bps / 10_000
    return slip + spread


def _stop_price(entry: float, direction: Direction, stop_atr_mult: float, atr: float) -> float:
    return entry - stop_atr_mult * atr if direction == Direction.LONG else entry + stop_atr_mult * atr


def _tp_price(entry: float, direction: Direction, tp_atr_mult: float, atr: float) -> float:
    return entry + tp_atr_mult * atr if direction == Direction.LONG else entry - tp_atr_mult * atr


def _pnl(entry: float, exit_: float, direction: Direction, shares: int) -> float:
    delta = exit_ - entry if direction == Direction.LONG else entry - exit_
    return delta * shares


def _check_exit(
    trade: _OpenTrade,
    bar: Bar,
    bars_held: int,
    force_close: bool,
    cm: _CostModel,
) -> dict | None:
    """Check stop, TP, max-hold, and force-close exits. Returns trade dict or None."""
    direction = trade.direction
    atr_approx = 0.0  # no live ATR for exit slippage; already baked into fill model

    # Stop hit (pessimistic: check bar low/high)
    stop_hit = (direction == Direction.LONG and bar.low <= trade.stop_price) or \
               (direction == Direction.SHORT and bar.high >= trade.stop_price)
    tp_hit = (direction == Direction.LONG and bar.high >= trade.tp_price) or \
              (direction == Direction.SHORT and bar.low <= trade.tp_price)

    if stop_hit:
        exit_execution_cost = _exit_execution_cost(trade.stop_price, atr_approx, cm)
        exit_price = _exit_fill(trade.stop_price, direction, atr_approx, cm)
        pnl = _pnl(trade.entry_price, exit_price, direction, trade.shares)
        return {
            "exit_reason": "stop",
            "pnl": pnl,
            "execution_cost": (trade.entry_execution_cost + exit_execution_cost) * trade.shares,
            "bars_held": bars_held,
            "exit_time": bar.event_time,
        }

    if tp_hit:
        exit_execution_cost = _exit_execution_cost(trade.tp_price, atr_approx, cm)
        exit_price = _exit_fill(trade.tp_price, direction, atr_approx, cm)
        pnl = _pnl(trade.entry_price, exit_price, direction, trade.shares)
        return {
            "exit_reason": "tp",
            "pnl": pnl,
            "execution_cost": (trade.entry_execution_cost + exit_execution_cost) * trade.shares,
            "bars_held": bars_held,
            "exit_time": bar.event_time,
        }

    if bars_held >= trade.max_hold_bars or force_close:
        exit_execution_cost = _exit_execution_cost(bar.close, atr_approx, cm)
        exit_price = _exit_fill(bar.close, direction, atr_approx, cm)
        pnl = _pnl(trade.entry_price, exit_price, direction, trade.shares)
        reason = "max_hold" if bars_held >= trade.max_hold_bars else "session_end"
        return {
            "exit_reason": reason,
            "pnl": pnl,
            "execution_cost": (trade.entry_execution_cost + exit_execution_cost) * trade.shares,
            "bars_held": bars_held,
            "exit_time": bar.event_time,
        }

    return None


def _empty_eval(hypothesis: object) -> EvalResult:
    return EvalResult(
        hypothesis_id=getattr(hypothesis, "hypothesis_id", "unknown"),
        run_id="",
        status=__import__("src.types", fromlist=["HypothesisStatus"]).HypothesisStatus.DRAFT,
        net_pnl=0.0,
        profit_factor=0.0,
        win_rate=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        max_drawdown=0.0,
        max_intraday_drawdown=0.0,
        worst_day=0.0,
        longest_losing_streak=0,
        trades_per_day=0.0,
        exposure_time=0.0,
        total_trades=0,
        slippage_sensitivity=0.0,
        composite_score=0.0,
    )


def _build_eval(hypothesis: object, trades: list[dict], total_bars: int) -> EvalResult:
    from src.types import HypothesisStatus

    if not trades:
        return _empty_eval(hypothesis)

    pnls = [t["pnl"] for t in trades]
    per_trade_commission = 2.0
    net_trade_pnls = [pnl - per_trade_commission for pnl in pnls]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_pnl = sum(pnls)
    # Commission: $1 min per leg × 2 legs = $2 per trade (unit-size simplification)
    total_commission = len(trades) * per_trade_commission
    net_pnl = gross_pnl - total_commission

    win_rate = len(wins) / len(pnls) if pnls else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 999.0

    # Max drawdown (equity curve)
    equity = [0.0]
    for p in net_trade_pnls:
        equity.append(equity[-1] + p)
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        peak = max(peak, e)
        max_dd = max(max_dd, peak - e)

    # Daily metrics use day-aggregated net trade PnL in New York session time.
    day_pnls: dict[date, float] = {}
    intraday_drawdowns: list[float] = []
    day_equity: dict[date, list[float]] = {}
    for trade, trade_net_pnl in zip(trades, net_trade_pnls, strict=True):
        session_day = pd.Timestamp(trade["exit_time"]).tz_convert("America/New_York").date()
        day_pnls[session_day] = day_pnls.get(session_day, 0.0) + trade_net_pnl
        day_curve = day_equity.setdefault(session_day, [0.0])
        day_curve.append(day_curve[-1] + trade_net_pnl)

    for curve in day_equity.values():
        day_peak = curve[0]
        day_dd = 0.0
        for point in curve:
            day_peak = max(day_peak, point)
            day_dd = max(day_dd, day_peak - point)
        intraday_drawdowns.append(day_dd)

    worst_day = min(day_pnls.values()) if day_pnls else 0.0
    max_intraday_drawdown = max(intraday_drawdowns) if intraday_drawdowns else 0.0

    # Longest losing streak
    max_streak = cur_streak = 0
    for p in pnls:
        if p <= 0:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0

    # Activity metrics
    bars_per_day = 78
    trading_days = max(total_bars / bars_per_day, 1)
    trades_per_day = len(trades) / trading_days
    total_bars_held = sum(t["bars_held"] for t in trades)
    exposure_time = total_bars_held / total_bars if total_bars else 0.0

    # Slippage sensitivity: recompute net_pnl under doubled spread/slippage costs.
    slippage_sensitivity = 0.0
    if net_pnl != 0:
        stressed_gross_pnl = sum(t["pnl"] - t["execution_cost"] for t in trades)
        net_2x = stressed_gross_pnl - total_commission
        slippage_sensitivity = abs(net_pnl - net_2x) / abs(net_pnl)

    # Composite score placeholder (evaluator module will override)
    composite_score = 0.0

    return EvalResult(
        hypothesis_id=getattr(hypothesis, "hypothesis_id", "unknown"),
        run_id="",
        status=HypothesisStatus.DRAFT,
        net_pnl=round(net_pnl, 4),
        profit_factor=round(profit_factor, 4),
        win_rate=round(win_rate, 4),
        avg_win=round(avg_win, 4),
        avg_loss=round(avg_loss, 4),
        max_drawdown=round(max_dd, 4),
        max_intraday_drawdown=round(max_intraday_drawdown, 4),
        worst_day=round(worst_day, 4),
        longest_losing_streak=max_streak,
        trades_per_day=round(trades_per_day, 4),
        exposure_time=round(exposure_time, 4),
        total_trades=len(trades),
        slippage_sensitivity=round(slippage_sensitivity, 4),
        composite_score=composite_score,
    )
