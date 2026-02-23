# Docstring Guidelines (Phase 1)

This project uses **NumPy-style docstrings** for developer-facing code.

## Scope and priorities

- Primary audience for docstrings: developers/maintainers.
- Primary audience for in-app explanatory UI copy: players.
- Use dual terminology based on context:
  - gameplay: **game / round**
  - RL/training: **episode / step**

When both are valid, write both once, e.g.:

`round index (episode step)`

## Required sections (for public functions/classes)

- One-line summary in imperative tone.
- Optional short context paragraph if behavior is non-obvious.
- `Parameters`
- `Returns`
- `Raises` (when applicable)
- `Notes` (when tradeoffs/assumptions matter)

## Style rules

- Keep docstrings behavior-oriented, not implementation-narrative.
- Document perspective of rewards explicitly (player vs agent).
- State action encoding explicitly (`0=rock, 1=paper, 2=scissors`).
- Prefer ASCII; avoid symbolic shorthand unless already standard.
- Keep private helper docstrings short unless helper is tricky.

## Dual terminology conventions

- `RoundObservation.step`:
  - round index for gameplay, or
  - step index for RL episode
- `session_index` in storage:
  - gameplay reset counter within same game id
- `reward_delta`:
  - always from the perspective of the actor represented by that transition
