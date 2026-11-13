# 4. Ceiling hunt (6 states)

**Question.** What do the NFAs that reach the full ceiling look like?

**Code.** `src/experiments/ceiling_hunt.py` · **Results.** `data/experiment_ceiling_hunt_6_states/`

```
cd src
python -m experiments.ceiling_hunt                 # until 10 hits
```

**Method.**
1. Start from the earlier 6-state million run, which had one NFA at the ceiling (the *head start*).
2. Generate distinct random 6-state NFAs (same parameter distribution as the sweeps) in parallel shards.
3. For each, build the subset DFA; only if it has all 64 states, minimize it. A **hit** has 64 states after minimization too.
4. Stop at 10 hits. Duplicates are never counted: an NFA and its 0/1-swapped copy share a **canonical key**. The earlier million cannot be excluded (only its keys were stored), so distinctness holds within this run and against the head start.

**Outputs.** `hits_nfas.json` (the 10 NFAs), `hits.csv`, `summary.json`, and `ceiling_nfas.pdf` (a page per NFA followed by a page for its 64-state DFA; made by `visualization/render_ceiling_hits.py`).

**Regular expressions.** `python -m experiments.ceiling_hits_regex` converts the 10 NFAs to regular expressions with `src/core/regex.py` (state elimination; works on any DFA or NFA, including ε edges) and writes `hits_regex.md`. Each expression is checked against its NFA on every string up to length 12, which is a test, not a proof of equivalence. The 64-state DFAs are not converted: state elimination makes expressions far too large to read at that size.

**Findings.** 2,517,323 new NFAs scanned in 68 s gave 9 new hits (about 3.6 per million). All 10 hits are **sparse, no ε edges, 12–14 symbol edges, 1–3 non-deterministic pairs, 1–3 accepting states**. Every hit's DFA contains all 64 subsets, including the empty set (dead state), and minimization merges none. They also share one more thing that was only noticed afterwards: **exactly one (state, symbol) slot has no transition** and the other 11 are defined (at q0 in 5 of the 10). [Experiment 6](06-structured-nfas.md) tests that shape.
