# Trait tests: do the traits actually cause larger DFAs?

Code: `src/experiments/trait_confirmation.py` (`replicate`, `run_interventions`). Output: `data/experiment_10k/trait_confirmation/{replication,replication_by_class,interventions}.csv`. Background: [feature-search.md](feature-search.md), terms: [terms.md](terms.md).

The feature search found associations on one dataset. Two further tests check that they are not an accident of that dataset and that they are not just correlation.

## Test A: replication on fresh NFAs

1. **Fresh sample.** 12,000 new random NFAs from `generate_balanced_nfas(seed=7)`: same generator, balanced over the six classes, never seen during the search.
2. **Extreme threshold.** For each (class, n), the 95th percentile of minimized DFA size **in the 10k run**. A fresh NFA is extreme if its DFA is strictly larger than the threshold for its group.
3. **Profile membership** (`matches_profile`): exactly one accepting state, accept distance ≥ 2, and 0.25–1.0 non-deterministic pairs per state.
4. **Compare** the extreme rate of profile NFAs with the rest, with a 95 % interval from 2,000 bootstrap resamples of each group.

| Group | NFAs | % extreme |
|---|---|---|
| matches profile | 955 | **9.11** |
| everything else | 11,045 | 4.73 |
| difference | | +4.38 points (95 % CI +2.59 to +6.24) |

The first run of the same idea gave 9.2 %. By class, the profile lifts mixed (≈13–14 %) and dense (12.5 %) but not sparse (6.6 % vs 5.1 %) or sparse-ε (5.6 % vs 5.5 %); sparse NFAs already look like the profile. Dense-ε had only 8 matching NFAs, which is too few to read.

## Test B: change one trait at a time (interventions)

**Design.** Take the first 4,000 of the fresh NFAs. For each, apply **one** structural change, rebuild the DFA, and compare with the same NFA's DFA before the change. Because every comparison is *paired on the same NFA*, differences between NFAs cannot explain the result. For each change we record:
- the **size ratio** after / before (mean and median),
- the share whose DFA **grew** and **shrank**,
- the **extreme rate** before and after, using the Test A thresholds.

**Changes** (`INTERVENTION_SEED = 11` drives the random choices):

| Change | What it does | Applies to |
|---|---|---|
| keep one accepting state | keep only the accepting state farthest from q0 | NFAs with ≥ 2 accepting states |
| move accepting state deep | accepting set becomes the single deepest reachable state | all |
| delete ε edges | remove every ε transition | NFAs with ε edges |
| cap branching at 2 | any (state, symbol) with > 2 destinations keeps a random 2 | NFAs that have such a pair |
| **control:** add accepting states | add random accepting states until there are 3 | NFAs with < 3 |
| **control:** add random edges | add one random symbol edge from every state | all |
| combined | delete ε edges, then move the accepting state deep | all |

The two controls change the trait the *opposite* way, so they should shrink the DFA if the traits matter.

**Results**

| Change | NFAs | Mean × | Median × | % grew | % shrank | Extreme before → after |
|---|---|---|---|---|---|---|
| keep one accepting state | 2,690 | 1.76 | 1.20 | 57.7 | 10.7 | 4.1 → 7.1 |
| move accepting state deep | 4,000 | 1.70 | 1.16 | 56.4 | 18.1 | 5.0 → 8.0 |
| cap branching at 2 | 3,082 | 1.67 | 1.10 | 52.8 | 10.5 | 4.4 → **13.9** |
| delete ε edges | 2,051 | 2.16 | 1.25 | 57.6 | 11.9 | 5.5 → 4.7 * |
| combined | 4,000 | 2.41 | 1.40 | 66.2 | 16.8 | 5.0 → 7.4 * |
| control: add accepting states | 2,669 | 0.85 | 1.00 | 8.5 | **50.0** | 5.6 → 3.4 |
| control: add random edges | 4,000 | 0.91 | 0.83 | 16.1 | **53.4** | 5.0 → 2.2 |

\* Without ε edges an NFA behaves like a member of the non-ε class, so its "after" extreme status is judged against that class's thresholds. The DFA gets bigger mostly because the NFA moved to a bigger-DFA class, not because it moved up within its class.

## What the tests show

- **Direction:** changes toward the profile grew the DFA in 53–66 % of NFAs and shrank it in 10–18 %. The opposite changes shrank it in about half and grew it in 9–16 %.
- **Size:** the *typical* effect is modest (median 1.1–1.4×). Means are higher because a few NFAs jump a lot. Reaching the real extremes needs several traits at once.
- **ε edges:** the within-class pattern in the feature search is the cleaner evidence; the deletion test is confounded as described above.
- **Strongly connected component** was not tested (cannot be changed in isolation).

## Limits

- Each change alters the language the NFA accepts, so this shows that *structure drives size*, not that one language gets a smaller or larger DFA.
- One random seed for the changes and one fresh sample.
- n = 5–12, alphabet {0, 1}.
