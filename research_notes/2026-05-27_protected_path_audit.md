# Protected Path Audit — 2026-05-27

Context:
- Latest full run completed with 299 walk-forward folds for `h_0001` and `h_0002`.
- Both hypotheses failed economically, so remaining uncertainty is mostly metric quality and evaluation interpretation, not data availability.
- This note records weaknesses in protected modules without modifying them.

## Confirmed weaknesses

1. `slippage_sensitivity` in `src/backtester/engine.py` is not currently measuring slippage.
It doubles `total_commission` only and leaves spread/slippage fill assumptions unchanged. The resulting metric is therefore a commission sensitivity proxy, even though the policy and evaluator treat it as slippage robustness.

2. The drawdown gate in `src/evaluator/scoring.py` scales the limit by `abs(net_pnl)`.
The current check is:
`result.max_drawdown > t["max_drawdown_pct"] * abs(result.net_pnl or 1.0)`
This makes the drawdown threshold depend on realized strategy outcome instead of account equity or a fixed capital base. Losing strategies can therefore face distorted gate behavior.

3. `worst_day` in `src/backtester/engine.py` is currently the minimum trade PnL, not the worst aggregated trading day.
That understates multi-trade day losses and makes the metric inconsistent with the research policy wording.

4. `max_intraday_drawdown` in `src/backtester/engine.py` is currently copied from total `max_drawdown`.
This is a placeholder rather than a distinct intraday metric and should not be interpreted as session-specific drawdown evidence.

## Adjacent non-protected finding fixed in this session

`src/papertrader/simulator.py` previously continued iterating through hypotheses after opening a position for a symbol, which allowed later hypotheses to overwrite the first one. This has been fixed and covered by a regression test in `tests/test_papertrader.py`.

## Safe next steps

1. Keep using new baseline hypotheses to compare idea quality while protected metrics stay unchanged.
2. Treat `slippage_sensitivity`, `worst_day`, and `max_intraday_drawdown` as provisional when interpreting reports.
3. If a human wants metric corrections, the first protected-path changes should target:
   - true 2x spread/slippage recomputation in `src/backtester/engine.py`
   - day-level PnL aggregation for `worst_day`
   - capital-based drawdown gating in `src/evaluator/scoring.py`