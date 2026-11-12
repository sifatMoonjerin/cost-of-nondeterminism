# 1. First sweep (3k)

**Question.** How large are DFAs built from random NFAs, compared with the NFA, and how much does minimization remove?

**Code.** `src/experiments/blowup_experiment.py` · **Results.** `data/experiment_3k/`

```
cd src
python -m experiments.blowup_experiment --count 3000 --include-json --out-dir ../data/experiment_3k
```
(Settings inferred from the saved data: 3,000 random NFAs plus the JSON NFAs.)

| | |
|---|---|
| NFAs | 3,000 random (5–12 states, seed 0; each draws its own density and ε setting) + 207 from `data/generated/*.json` (50 each of sparse, dense, sparse-ε, dense-ε, plus 7 blow-up) = 3,207 |
| Distinct | by NFA id (sha256 of the canonical form) |
| Classes | sparse, mixed, dense, each with and without ε, plus blow-up ([terms.md](../terms.md)) |
| Measures | subset DFA states, minimized DFA states, size ratio, transition counts, structural features |

**Outputs.** `blowup.duckdb` (table `nfa_results` and views such as `larger_rate_by_type`, `size_outcome_by_type`, `features_larger_vs_not`), `nfas.json`, and four figures: `relative_size`, `larger_rates`, `size_outcome_by_type`, `subset_size_outcome_by_type`.

**Findings.** Sparse NFAs end up with a larger minimized DFA most often (~86 %); dense NFAs rarely (~17 %). The raw subset DFA is larger for most NFAs, but minimization removes most of that for dense NFAs.

**Notes.**
- The code's default output folder is `data/experiment`; the existing results were moved to `data/experiment_3k`.
- The 7 blow-up rows in this folder were produced before the generator fix and list n instead of n + 1 states, so their size ratios are slightly too high. DFA sizes are correct.
