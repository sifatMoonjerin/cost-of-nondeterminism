# Measuring the Cost of Nondeterminism

How much larger is the DFA than the NFA it came from, and which NFAs blow up? This repository holds the code for generating random NFAs, converting them with subset construction, minimizing the result (Moore), and measuring the sizes, together with the experiments behind the paper `docs/final-report-edited.tex`.

## Where to look

| | |
|---|---|
| `docs/README.md` | Index of the documentation: terms, how the traits were found and tested, one page per experiment |
| `docs/paper-figure-map.md` | The script and data behind each figure and table of the paper |
| `docs/final-report-edited.tex`, `docs/figures/` | The paper source and its figures (the template class and `refs.bib` are not included) |
| `src/` | Code; run every script from `src/` as a module, for example `cd src && python -m experiments.ceiling_hunt` |
| `nfa-generator.ipynb` | The original notebook: generator, subset construction, Moore minimization, first experiment |
| `data/generated/` | The hand-made sparse / dense / ε / blow-up NFAs of the first experiment (JSON) |

Result folders (`data/experiment_*`) are not committed; the commands in `docs/experiments/` regenerate them.

## Reproducing the results

Python 3 with duckdb, pandas, numpy, matplotlib and seaborn (plus scikit-learn and scipy for the trait analysis); Graphviz for the drawings. The random generator is seeded and gives the same NFAs in every run (`docs/terms.md`). Every run's conversion is checked for correctness (`docs/experiments/07-validation.md`).
