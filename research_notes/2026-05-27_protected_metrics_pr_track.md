# Protected Metrics PR Track (Draft)

Purpose:
Define a separate, review-focused change track for protected paths (`src/backtester`, `src/evaluator`) without mixing strategy work.

## Scope (in)

1. Replace pseudo-slippage sensitivity with explicit scenario stress:
   - base costs
   - 2x slippage
   - 2x spread
   - 2x slippage + 2x spread
2. Add realistic stop/target exit slippage handling (instead of fixed `atr_approx = 0.0`).
3. Compute `worst_day` from day-aggregated PnL (calendar/session grouped), not per-trade min.
4. Compute true `max_intraday_drawdown` from intraday equity curves by session.
5. Revisit drawdown gate denominator logic in evaluator to avoid dependence on `abs(net_pnl)`.

## Scope (out)

1. No new hypothesis logic.
2. No risk-limit loosening.
3. No broker/execution integration changes.
4. No cost/slippage assumption reductions.

## Deliverables

1. Code changes in protected modules only.
2. Dedicated tests for each corrected metric path.
3. One reproducible report diff showing before/after metric impact on the same dataset.
4. Explicit reviewer checklist tied to each metric.

## Suggested PR slicing

1. PR-A: slippage scenarios + stop/target fill stress + tests.
2. PR-B: worst_day and max_intraday_drawdown correctness + tests.
3. PR-C: evaluator drawdown gate denominator correction + tests + migration note.

## Acceptance criteria

1. CI green including protected-path guard.
2. No changes to strategy hypothesis files in this track.
3. Report provenance unchanged except expected metric outputs.
4. Metric computations traceable and documented in test assertions.
