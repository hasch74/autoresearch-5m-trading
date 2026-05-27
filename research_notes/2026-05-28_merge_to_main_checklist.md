# Merge To Main Checklist (2026-05-28)

Purpose:
Provide a maintainer-facing checklist for moving the best current research state from
`copilot/simplify-instructions-for-coding` to `main` without ambiguity.

## Candidate branch head

- Branch: `copilot/simplify-instructions-for-coding`
- Current tip at review time: `5a4a9c6`

## Minimum merge payload

1. Baseline hypotheses and tests:
   - `h_0003`, `h_0004`, `h_0005`, `h_0006`
   - updated `tests/test_hypotheses.py`
2. Reproducible report reference:
   - `reports/research_run_20260527_220714.json`
3. Research notes:
   - comparison snapshot
   - protected-path audit
   - protected metrics PR track
   - variant smoke results

## Pre-merge checks

1. Confirm branch is up to date with `main` and rebased/merged cleanly.
2. Confirm CI is green on merge candidate commit.
3. Confirm report provenance is clean:
   - `provenance.git.is_dirty == false`
   - `provenance.git.commit` equals merge candidate commit.
4. Confirm all hypotheses remain non-promoted (`quarantined` outcomes expected currently).

## Suggested PR description skeleton

1. Scope:
   - branch parity update from copilot branch to main
2. Included artifacts:
   - list exact files under `strategies/hypotheses/`, `tests/`, `research_notes/`, `reports/`
3. Validation:
   - test command outputs
   - report provenance snippet
4. Risks:
   - no trading-ready strategy yet
   - protected-metrics corrections still pending in follow-up PR track

## Post-merge immediate actions

1. Open dedicated protected-metrics PR-A / PR-B / PR-C as defined in
   `research_notes/2026-05-27_protected_metrics_pr_track.md`.
2. Re-run one clean reference report on `main` after merge and archive it under `reports/`.
