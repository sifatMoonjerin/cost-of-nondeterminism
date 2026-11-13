# 3. Million-NFA runs

**Question.** How close do random NFAs get to the `2^n` ceiling when we look at a million of them, and how does that depend on n?

**Code.** `src/experiments/blowup_experiment_1m.py` · **Results.** three folders in `data/`

```
cd src
python -m experiments.blowup_experiment_1m                                                     # 5-12 states, seed 2
python -m experiments.blowup_experiment_1m --min-states 12 --max-states 12 --seed 3 --out-dir ../data/experiment_1m_12_states
python -m experiments.blowup_experiment_1m --min-states 6  --max-states 6  --seed 4 --out-dir ../data/experiment_1m_6_states
```

| Folder | n | NFAs | Runtime (8 cores) |
|---|---|---|---|
| `experiment_1m` | 5–12 | 1,000,000 | 3 min |
| `experiment_1m_12_states` | 12 | 1,000,000 | 7.5 min |
| `experiment_1m_6_states` | 6 | 1,000,000 | 1 min |

Each is split evenly over the six classes, **distinct by full sha256 key** (duplicates across worker shards are dropped and replaced), blow-up family excluded. Work is split into 5,000-NFA shards run in parallel.

**Outputs** (per folder): `blowup.duckdb` (table `nfa_results`, one row per NFA, ~70 MB), `summary_by_class.csv`, `summary_by_nfa_size.csv` (or `summary_by_transitions.csv` when n is a single value), `figures/larger_dfa_violin.png`, `figures/ceiling_approach.png`. With one n, the scatter's x-axis is the number of transitions instead of n.

**Findings.**

| Run | Best DFA | Share of ceiling | NFAs at the full ceiling |
|---|---|---|---|
| 5–12 states | 32 of 32 (n = 5) | 100 % | 3 |
| 6 states | 64 of 64 | 100 % | 1 |
| 12 states | 1,142 of 4,096 | 28 % | 0 |

The ceiling is reachable by chance only for NFAs of about 5–6 states. At 12 states, DFA size peaks at roughly 24–35 transitions and falls off on both sides: too little non-determinism leaves nothing to track, too much makes the subsets collapse.

**Extra charts.** `python -m experiments.blowup_experiment_1m --scatter-only --out-dir <folder>` redraws, from the existing database, `figures/ceiling_scatter.png` (the left panel alone, legend beside the axes) and `figures/ceiling_hexbin.png` (all one million NFAs binned into hexagons coloured by count, to avoid overplotting).

**Matched baselines for the structured runs.** [Experiment 6](06-structured-nfas.md) needs one million unrestricted NFAs *without ε edges* at 5, 6 and 7 states. Each size is generated as 2,000,000 NFAs, whose non-ε half (sparse, mixed, dense) is the baseline:

```
python -m experiments.blowup_experiment_1m --count 2000000 --seed 3 --min-states 6 --max-states 6 --out-dir ../data/experiment_unrestricted_6_states_2m
```
(likewise for 5 and 7; about 75 s each, folders `experiment_unrestricted_<n>_states_2m`).

**Notes.** These runs predate the sorted-alphabet fix (commit `e3d4593`), so their seeds do not regenerate the same NFAs byte for byte; sizes are unaffected. The database stores keys, features and sizes, not the NFAs.
