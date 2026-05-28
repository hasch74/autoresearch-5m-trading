# autoresearch-5m-trading

Autonomous 5-minute trading research system with a **hard safety cage**.

## Goal

Maximize **risk-adjusted net PnL** (after costs and slippage) under strict drawdown and ruin constraints.

## Scope (MVP)

- 5-minute bars
- Liquid US universe (small initial set)
- Regular trading hours only
- No overnight positions
- Research and paper-trading first, broker live execution later

## Core Principle

The agent can propose and test strategy hypotheses, but must not change the judging logic:

- agent-editable: strategy hypothesis files and research reports
- protected: evaluator, backtester, risk rules, broker integration, and cost assumptions

See `/research_policy.yaml` for the enforceable research policy and `/.github/copilot-instructions.md` for implementation behavior.

## Current MVP Scaffold

- `data/{raw,normalized,features,paper_trades}`
- `src/{data_ingest,feature_store,backtester,papertrader,evaluator,risk,broker_ibkr,hypothesis_engine,agent_runner}`
- `strategies/hypotheses`
- `configs`
- `reports`
- `logs`
- `research_notes`

## Research Runner CLI

Run one research cycle:

```powershell
python -m src.agent_runner.runner
```

### Targeted runs

Run only specific hypotheses:

```powershell
python -m src.agent_runner.runner --hypotheses h_0006
```

Exclude expensive/irrelevant hypotheses:

```powershell
python -m src.agent_runner.runner --exclude-hypotheses h_0003 h_0005
```

Run on selected symbols only:

```powershell
python -m src.agent_runner.runner --symbols SPY QQQ
```

Combine filters:

```powershell
python -m src.agent_runner.runner --workers 8 --hypotheses h_0006 --symbols SPY QQQ
```

### Fast smoke mode

Use fast mode for quick iteration:

```powershell
python -m src.agent_runner.runner --fast
```

`--fast` applies defaults when not explicitly provided:

- workers: `4`
- max-days: `120`
- symbols: `SPY`, `QQQ`

Explicit flags always override fast defaults:

```powershell
python -m src.agent_runner.runner --fast --workers 8 --symbols NVDA --max-days 30
```

### Window control

Restrict each symbol to the trailing N calendar days by `event_time`:

```powershell
python -m src.agent_runner.runner --max-days 120
```

### Notes

- Hypothesis filtering order is deterministic: include first, then exclude.
- Reports include execution provenance for `fast_mode`, `max_days`, hypotheses filters, and symbol filters.