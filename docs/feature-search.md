# Feature search: which NFA traits go with large DFAs?

Code: `src/experiments/trait_confirmation.py` (steps 1–2), data: `data/experiment_10k/`, output: `data/experiment_10k/trait_confirmation/`. Terms are defined in [terms.md](terms.md).

**Question.** Among NFAs that look alike, which structural traits do the ones with an unusually large minimized DFA share?

## Method

1. **Data.** The balanced 10k run: 10,000 random NFAs, evenly split across the six classes, n = 5–12 (`experiments/blowup_experiment_10k.py`). The 7 hand-built blow-up NFAs are excluded.

2. **Candidate traits (24).**
   - Stored per NFA at generation time (`get_structural_features`): accepting states, symbol edges, ε edges, average and maximum out-degree, non-deterministic pairs, average branching, symbol coverage.
   - Graph traits added in `get_graph_features`: BFS depth from q0, nearest and farthest accept distance, number of SCCs, largest-SCC fraction, share of states with a non-deterministic symbol, share of missing (state, symbol) slots, start-state loops and degree, accepting-state out-degree, in-degree statistics.

3. **Outcome: "extreme".** An NFA is extreme if its minimized DFA is in the **top 5 %** of its group, where a group is *same class and same n*. This gives 520 extreme NFAs of 10,000, about 5 % in every class, so no class is over-represented by construction.

4. **Which traits differ? (`feature_z_differences`)** Each trait is **z-scored inside its group** (subtract the group mean, divide by the group standard deviation). Then compare the mean z of extreme NFAs with the mean z of the rest. A negative number means extreme NFAs have *less* of that trait than their peers.

5. **How do extremes depend on the trait? (`trait_rates`)** Bin each trait and report the **share of NFAs in each bin that are extreme**. The baseline is 5.2 %; a bin at 8 % makes an extreme DFA about 1.5× as likely. Binning matters because several traits are **not monotonic** (inverted U).

6. **Profile.** The bin results were combined into a profile (one accepting state, accept distance ≥ 2, 0.25–1.0 non-deterministic pairs per state). Because the thresholds were chosen after looking at this data, they are tested on fresh NFAs rather than trusted. See [trait-tests.md](trait-tests.md).

## Results (10k run)

Largest z differences, extreme minus rest (all in the **negative** direction):

| Trait | z difference | Extreme mean | Rest mean |
|---|---|---|---|
| average branching | −0.70 | 1.65 | 1.86 |
| average out-degree | −0.57 | 3.02 | 3.41 |
| non-det. pairs per state | −0.55 | 0.88 | 1.03 |
| symbol edges | −0.52 | 25.0 | 28.0 |
| accepting states | −0.40 | 1.69 | 2.01 |
| ε edges | −0.35 | 0.76 | 1.03 |

Share of NFAs that are extreme, by trait value (baseline 5.2 %):

| Trait | Values → % extreme |
|---|---|
| non-det. pairs per state | ≤0.25: 3.8 · 0.25–0.5: **8.4** · 0.5–0.75: **7.9** · 0.75–1.0: 6.6 · 1.0–1.5: 4.9 · >1.5: **1.3** |
| average branching | ≤1.2: 5.4 · 1.2–1.4: **8.6** · 1.4–1.6: 8.0 · 2.0–2.5: 2.4 · >2.5: **0.4** |
| accepting states | 1: **8.0** · 2: 4.4 · 3: 3.2 |
| accept distance | 0: 4.5 · 1: 4.3 · 2: 6.5 · **3: 9.0** · 4: 7.3 · 5+: 3.5 |
| ε edges (ε classes) | **1: 10.0** · 2: 4.7 · 3: 1.5 |
| single SCC | yes: 5.7 · no: 2.1 |

**Reading it.** Extreme DFAs come from NFAs with *some but not much* non-determinism, one accepting state a few steps from the start, few ε edges and one connected component. Too little non-determinism leaves nothing to track; too much makes the subsets collapse into the same few sets, which minimization merges.

## Extra checks (`src/experiments/feature_search_extras.py`)

Run `cd src && python -m experiments.feature_search_extras` (needs scikit-learn and scipy). Output: `data/experiment_10k/feature_search_extras/`.

1. **Significance.** A Mann-Whitney U test per trait on the within-group z-scores, extreme vs rest. For the top-5 % definition, **19 of 24 traits have p < 0.001** (smallest 2 × 10⁻⁵⁴, for average branching). The five that do not are `max_accept_depth` (p = 0.81), `symbol_coverage` and `missing_slot_frac` (0.099 each, the same information), `start_nondet` (0.027) and `in_degree_1_frac` (0.006). Those traits do not separate extreme NFAs from the rest.
2. **How predictive are the traits?** A random forest (300 trees) predicts "extreme" from the within-group z-scored traits plus two extra ratios (symbol edges per state, share of slots with exactly one destination). **5-fold cross-validated AUC = 0.82**, where 0.5 is chance and 1.0 is perfect: the traits carry real signal but do not determine which NFAs are extreme. The traits the forest relies on most (AUC lost when shuffled): ε edges 0.020, accepting states 0.017, deterministic-slot share 0.014, average branching 0.010.
3. **Does the definition of "extreme" matter?** Repeating the comparison with the top **1 %** (122 NFAs) and with the raw **subset** DFA (528 NFAs) gives the same leading traits: average branching, non-deterministic pairs, out- and in-degree. Effects are larger for the top 1 % (branching z = −0.98 vs −0.70). For the subset DFA, ε edges and missing (state, symbol) transitions rank higher, since minimization has not yet merged the ε-induced duplicates.

## Limits

- Association, not cause. [trait-tests.md](trait-tests.md) addresses cause for four of the traits.
- Only NFAs with 5–12 states over {0, 1}, and only the generator's distribution.
- The ε rows compare within the ε classes only, since other classes have no ε edges.
