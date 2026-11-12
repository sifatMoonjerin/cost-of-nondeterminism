# Documentation

Project: how much larger is the DFA than the NFA it came from, and which NFAs blow up? Code is in `src/`, results in `data/`.

| Read | For |
|---|---|
| [terms.md](terms.md) | Definitions: sparse / dense / mixed as ratios, sizes, ceiling, features, "extreme" |
| [feature-search.md](feature-search.md) | How the traits behind large DFAs were found |
| [trait-tests.md](trait-tests.md) | How those traits were tested (replication and one-change-at-a-time tests) |
| [experiments/](experiments/) | One short page per experiment, in the order below |

## Experiments

1. [First sweep (3k)](experiments/01-first-sweep.md)
2. [Balanced 10k run](experiments/02-balanced-10k.md) (the trait analysis runs on this)
3. [Million-NFA runs](experiments/03-million-runs.md)
4. [Ceiling hunt, 6 states](experiments/04-ceiling-hunt.md)
5. [Visualizations](experiments/05-visualizations.md)

## Code layout

```
src/
  core/            automata.py      NFA, DFA, generator, subset construction, Moore minimization
  experiments/     blowup_experiment.py        first sweep, DuckDB views, base plots
                   blowup_experiment_10k.py    balanced 10k run, violin and ceiling plots
                   blowup_experiment_1m.py     million-NFA runs (parallel)
                   trait_confirmation.py       feature search, replication, interventions
                   feature_search_extras.py    p-values, random forest, robustness of the feature search
                   ceiling_hunt.py             find NFAs that reach the ceiling
  visualization/   visualize_dfa.py, visualize_nfa.py, render_running_example.py, render_ceiling_hits.py
  nfa-generator.py                             earlier stub, unused
nfa-generator.ipynb                            original notebook (generator, subset construction, Moore)
data/generated/    the hand-made sparse / dense / ε / blow-up NFAs (JSON)
```

Run every script **from `src/`** as a module, e.g. `cd src && python -m experiments.ceiling_hunt`. Python packages: duckdb, pandas, numpy, matplotlib, seaborn (plus scikit-learn and scipy for `feature_search_extras.py`). Plots use the Georgia font if present. Graphviz is needed for `visualization/`.
