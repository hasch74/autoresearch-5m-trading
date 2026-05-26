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