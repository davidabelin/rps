# Strategy Glossary (Developer)

This glossary links user-facing strategy names to implementation modules and
summarizes behavior in plain language.

Action encoding used throughout:

- `0 = rock`
- `1 = paper`
- `2 = scissors`

## Constant and simple policies

| Strategy name | Module | Summary |
|---|---|---|
| `rock` / `paper` / `scissors` | `rps_agents/heuristic/constants.py` | Always play one fixed action. |
| `nash_equilibrium` | `rps_agents/heuristic/constants.py` | Uniform random mixed strategy baseline. |
| `hit_last_own_action` | `rps_agents/heuristic/constants.py` | Cycles own action each round (anti-copy tendency). |

## Reactive/statistical policies

| Strategy name | Module | Summary |
|---|---|---|
| `copy_opponent` | `rps_agents/heuristic/basic.py` | Mirrors opponent's previous action. |
| `reactionary` | `rps_agents/heuristic/basic.py` | Keeps action after wins; counters opponent after tie/loss. |
| `counter_reactionary` | `rps_agents/heuristic/basic.py` | Meta-policy intended to exploit reactionary behavior. |
| `statistical` | `rps_agents/heuristic/basic.py` | Counters the opponent's most frequent historical move. |

## Sequence-modeling policies

| Strategy name | Module | Summary |
|---|---|---|
| `markov` | `rps_agents/heuristic/markov.py` | Markov-style sequence predictor over mixed action context. |
| `memory_patterns` | `rps_agents/heuristic/memory_patterns.py` | Memorizes repeated mixed patterns and predicts continuation. |
| `opponent_transition_matrix` | `rps_agents/heuristic/opponent_transition.py` | Learns opponent-to-opponent transition probabilities. |
| `decision_tree` | `rps_agents/heuristic/decision_tree.py` | Online decision tree over rolling handcrafted features. |

## Ensemble/meta policies

| Strategy name | Module | Summary |
|---|---|---|
| `multi_armed_bandit` | `rps_agents/heuristic/multi_armed_bandit.py` | Thompson-sampling chooser over many predictor arms. |
| `polling_agent` | `rps_agents/heuristic/ensemble.py` | Weighted voting across voter agents. |
| `rotating_ensemble` | `rps_agents/heuristic/ensemble.py` | Randomly rotates active voter on prime intervals. |

## Registry and factories

- Agent registration metadata is defined in `rps_agents/heuristic/__init__.py`.
- Build by name using `build_heuristic_agent(name)`.
- Enumerate available specs using `list_agent_specs()`.
