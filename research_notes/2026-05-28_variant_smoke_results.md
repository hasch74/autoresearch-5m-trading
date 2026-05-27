# Variant Smoke Results (2026-05-28)

Context:
- New variant hypotheses added: `h_0005`, `h_0006`.
- Unit tests pass in `tests/test_hypotheses.py`.
- Quick integration run executed on `SPY` only to get fast signal before expensive full multi-symbol run.

## SPY quick run summary

| Hypothesis | Trades | Net PnL | Win rate | Profit factor | WF folds | Positive fold ratio | Median OOS net PnL |
|---|---:|---:|---:|---:|---:|---:|---:|
| h_0005 | 1319 | -3309.1760 | 0.1873 | 0.1914 | 13 | 0.0000 | -202.5552 |
| h_0006 | 33 | -83.0554 | 0.0606 | 0.0740 | 13 | 0.0000 | -2.6132 |

## Interpretation

1. Both variants remain non-viable on SPY in current form.
2. `h_0006` is much lower frequency than `h_0005`, but still fails economically.
3. Full multi-symbol evaluation is still required for final ranking, but there is no immediate positive signal from this smoke pass.
