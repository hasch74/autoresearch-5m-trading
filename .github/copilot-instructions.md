# Copilot Coding Instructions

## 1) Think before coding

- State assumptions explicitly.
- If multiple interpretations exist, list them.
- Prefer the simpler solution and push back on unnecessary complexity.
- If something is unclear, stop and ask.

## 2) Simplicity first

- Implement only what was requested.
- Avoid speculative abstractions and configurability.
- Keep changes minimal and direct.

## 3) Surgical changes

- Touch only files required by the task.
- Do not refactor unrelated code.
- Match existing style.
- Remove only dead code introduced by your own changes.

## 4) Goal-driven execution

- Define verifiable success criteria for each task.
- Use small steps with validation after each step.
- Prefer tests that reproduce bugs and prove fixes.

## Trading research guardrails

- Agent may edit strategy hypotheses and research outputs only.
- Agent must not alter risk logic, evaluator logic, cost/slippage assumptions, or broker execution code.
- Strategy promotion requires passing all evaluation gates from `research_policy.yaml`.
