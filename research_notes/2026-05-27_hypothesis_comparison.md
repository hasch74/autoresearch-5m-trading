# Hypothesis Comparison Snapshot (2026-05-27)

Reference reports:
- Baseline older run: `reports/research_run_20260526_203411.json`
- Prior full multi-hypothesis run: `reports/research_run_20260527_172751.json`
- Latest clean reproducible run: `reports/research_run_20260527_220714.json`

## Executive summary

- No hypothesis passed gates in either run.
- The newer run removes `insufficient_folds` and provides real walk-forward coverage (`299` folds).
- `h_0003` and `h_0004` act as useful control baselines, but both are economically poor in current form.

## Side-by-side metrics

| Hypothesis | Older run trades | Older net PnL | Newer run trades | Newer net PnL | Newer WF folds | Newer WF positive ratio | Gate passed (newer) |
|---|---:|---:|---:|---:|---:|---:|---|
| h_0001 | 42 | -95.8362 | 488 | -1128.6750 | 299 | 0.0368 | false |
| h_0002 | 38 | -59.5894 | 513 | -839.7106 | 299 | 0.0334 | false |
| h_0003 | n/a | n/a | 63935 | -146458.5655 | 299 | 0.0000 | false |
| h_0004 | n/a | n/a | 2377 | -5437.8194 | 299 | 0.0100 | false |

## Interpretation

1. Data/window readiness improved: walk-forward now executes meaningfully.
2. Strategy quality remains insufficient: all four hypotheses fail profitability and/or robustness gates.
3. Baselines are informative control points but not promotion candidates.

## Refresh outcome after timestamp fix

- Clean run provenance now confirms reproducibility:
	- `commit`: `76c6c96a9890125f9a02f2ed63503085308e1721`
	- `is_dirty`: `false`
- Key hypothesis metrics and gate outcomes remain unchanged versus prior full runs.
- Conclusion remains stable: data and walk-forward infrastructure is functioning; strategy quality is still insufficient.
