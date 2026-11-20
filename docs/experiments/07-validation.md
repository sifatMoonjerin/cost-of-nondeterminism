# 7. Validation and rerun

**Question.** Does the NFA -> subset DFA -> minimized DFA pipeline preserve every NFA's language, and do the results survive the generator fix (commit `e3d4593`, sorted alphabet) that made NFA generation reproducible?

**Code.** `src/core/validate.py` (the checks), `src/experiments/validate_runs.py` (runs them), and the inline hook in `src/experiments/blowup_experiment_1m.py` · **Results.** `data/validation/*.json` and `validation.json` in each million-NFA folder

```
cd src
python -m experiments.validate_runs stored                    # 10k run, 3k sweep, data/generated, ceiling hits
python -m experiments.validate_runs structured --states 6     # regenerate from stored seeds, then check (also 5, 7)
python -m experiments.validate_runs baseline --states 6       # same for the unrestricted 2M run
python -m experiments.blowup_experiment_1m                    # million run; checks run inline (--random-share 0.01)
```

## The checks (`core/validate.py`)

They do not call the construction code: the NFA is simulated with its own ε-closure and the exact checks explore pairs of states directly. Each raises `ValidationFailure` with the NFA's label, the string and all three automata's answers; a run stops at the first failure.

| Check | What it does |
|---|---|
| NFA vs subset DFA (exact) | explores every (NFA state set, DFA state) pair reachable together; accepting status must always agree. Proves equivalence. |
| Subset DFA vs minimized DFA (exact) | same exploration over pairs of DFA states |
| Random strings | 500 random strings of length 0-20 per NFA; the NFA, subset DFA and minimized DFA must all accept or all reject each |

Deliberately broken automata are caught by all three checks.

## What was checked (all runs: 0 failures)

| Run | NFAs checked exactly | Also random strings |
|---|---|---|
| Balanced 10k (10,000 + 7 blow-up) | 10,007 | all |
| First sweep, 3,200 + 7 (`experiment_3k`) | 3,207 | all |
| `data/generated/*.json` (first experiment) | 257 | all |
| Ceiling hits | 10 | all |
| Million runs, 5-12 / 12 / 6 states (inline) | 1,000,079 / 1,000,000 / 1,000,255 | about 1 % (9,941 / 9,812 / 10,061) |
| Structured, 5 / 6 / 7 states | 1,000,000 each | 10,143 each |
| Unrestricted baselines, 5 / 6 / 7 states | 2,000,000 each (all classes) | 20,155 each |

About 12 million NFAs in all. The ceiling hunt's 2.5 million scanned NFAs are not each checked, only its ten hits. The structured and baseline NFAs are regenerated from the seeds stored in their databases (inside the worker processes) and each regenerated NFA's key is compared with the stored key (0 mismatches). The inline counts for the million runs include the few duplicates dropped after generation.

## Rerun after the generator fix

The balanced, three million and 6-state runs were rerun with the fixed generator (old folders kept locally as `<name>_prefix`), then the trait tests, feature-search extras, charts and the ceiling hunt.

- **10k run:** all 10,000 random NFAs have the same seed, id, sizes and features as before. The trait CSVs and the extras are byte-identical.
- **Million runs:** every shared seed has identical sizes. Between old and new, 42 (5-12 states) and 162 (6 states) seeds differ, and none at 12 states: which copy of an NFA that two parallel shards both produced is kept depends on arrival order. The size distribution moves by 10 NFAs of a million at 5-12 states, and the 6-state run by 0.001 percentage points (33.577 % to 33.576 % larger). The old stored keys differ from the new ones for 25-50 % of NFAs, which are the 0/1-swapped copies the bug produced.
- **Ceiling hunt:** the rerun finds the same ten hits (same canonical keys, same order). It scanned 2,497,372 NFAs against 2,517,323 before (3.60 against 3.58 hits per million), again because of arrival order.
- **Reproducibility:** 11,000 NFAs (5,000 from the generator, 6,000 structured) give the same keys under `PYTHONHASHSEED` 1, 2 and 3. With the pre-fix line the keys change with the hash seed, so the test does detect the problem.

## Other notes

- Sparse-epsilon: 84.523 % of the 10k NFAs have a larger subset DFA. It rounds to 85 %. A bar label once read 84 % because the DuckDB view rounded to 84.5 before the plot rounded again (Python formats 84.5 as "84"); the views now keep three decimals.
- DuckDB: the databases written by DuckDB 2.0.0 (structured, baselines, 3k) cannot be opened by 1.5.x. The reruns were written with 1.5.5.
