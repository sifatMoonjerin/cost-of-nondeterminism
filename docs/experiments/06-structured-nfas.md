# 6. Structured NFAs (5, 6 and 7 states)

**Question.** The 10 NFAs from the [ceiling hunt](04-ceiling-hunt.md) share a shape. If we generate *only* NFAs of that shape, how much more often do they get large DFAs, and do they reach the `2^n` ceiling?

**Code.** `src/experiments/structured_nfa_experiment.py` · **Results.** `data/experiment_structured_<n>_states/`

```
cd src
python -m experiments.structured_nfa_experiment --states 6          # also 5 and 7; --count 20000 for a quick check
python -m experiments.structured_nfa_experiment --states 6 --figures-only    # redraw figures and the baseline comparison
python -m experiments.structured_summary_table                      # data/structured_experiments_table.png
```

**The generator** (`generate_structured_nfa`). n states over {0, 1}, start q0, **no ε edges**:
1. 1–3 random accepting states.
2. One of the 2n (state, symbol) slots, chosen uniformly, is left **empty** (no preference for q0).
3. Every other slot gets one uniformly random destination (any state, itself included).
4. 1–3 of those defined slots, chosen at random, get a second, different destination. That gives 12–14 edges at n = 6 and 1–3 non-deterministic pairs, 73–91 % of the defined slots deterministic.
5. Redraw unless every state is reachable from q0. Each NFA comes from its own seed, so it regenerates exactly.

Each run is **1,000,000 distinct NFAs** (an NFA and its 0/1-swapped copy count as one), converted with subset construction and minimized, on all cores (1–2 min per size).

**Baseline.** One million unrestricted NFAs *without ε edges* at the same n: the sparse, mixed and dense classes of the 2,000,000-NFA runs described in [experiment 3](03-million-runs.md). (The structured NFAs have no ε edges, so the ε classes are left out of the comparison.)

**Outputs** (per folder). `blowup.duckdb` (table `nfa_results`, one row per NFA, plus `undefined_state` and `undefined_symbol`), `hits_nfas.json` (the NFAs at the ceiling), `summary.json`, `size_distribution.csv`, and `figures/`: `size_threshold_bars.png` (grouped bars, labelled "1 in N"), `size_histogram.png` (share of NFAs at each DFA size, log scale), `ceiling_hexbin.png`. The 3-row-per-size table is `data/structured_experiments_table.png`.

**Findings** (structured vs unrestricted, 1,000,000 each).

| n (ceiling) | Median DFA | At least ½ the ceiling | At the ceiling |
|---|---|---|---|
| 5 (32) | 9 vs 5 | 11.4 % vs 1.72 % (7×) | 608 vs 34 (18×) |
| 6 (64) | 13 vs 6 | 3.77 % vs 0.215 % (18×) | 116 vs 3 (39×) |
| 7 (128) | 20 vs 7 | 1.14 % vs 0.0255 % (45×) | 30 vs 0 |

- The shape raises the median DFA about 2–3× and the chance of a very large DFA by an order of magnitude or more. The gap widens with n.
- Reaching the full ceiling still gets harder as n grows even with this shape (608, 116, 30 per million).
- The ceiling counts of the unrestricted baseline (34, 3, 0) are small, so the ceiling ratios are rough; the threshold columns are the better evidence.
- Putting the empty slot at q0 gives no consistent advantage (n = 6: 14 of 143,353 NFAs with q0 empty reach the ceiling vs 102 of 856,647 otherwise).

**Limits.** The three ingredients (one empty slot, accepting states, non-deterministic pairs) are fixed together, so this does not say which part matters. Only 5–7 states and the alphabet {0, 1}. The first comparison used smaller baselines (the mixed 5–12 state run); the old summaries are kept as `summary_small_baseline.json` and `size_distribution_small_baseline.csv`.
