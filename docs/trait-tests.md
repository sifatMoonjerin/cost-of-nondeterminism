# Trait tests: do the traits actually cause larger DFAs?

Code: `src/experiments/trait_confirmation.py` (`replicate`, `run_interventions`, `run_replicated_interventions`; `--interventions-only` reruns just Test B). Output: `data/experiment_10k/trait_confirmation/{replication,replication_by_class,interventions,interventions_by_replication}.csv` and `figures/interventions.png`. Background: [feature-search.md](feature-search.md), terms: [terms.md](terms.md).

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

**Design.** Take a fresh sample of 4,000 random NFAs. For each, apply **one** structural change, rebuild the DFA, and compare with the same NFA's DFA before the change. Because every comparison is *paired on the same NFA*, differences between NFAs cannot explain the result. For each change we record:
- the **size ratio** after / before (mean and median),
- the share whose DFA **grew** and **shrank**,
- the **extreme rate** before and after, using the Test A thresholds.

**Replication.** The whole test is repeated **10 times**, each on an independent fresh balanced sample (`FRESH_SEED + r`, r = 0…9) with its own random choices (`INTERVENTION_SEED + r`, r = 0…9). The table gives the mean over the 10 replications; the 95 % interval is a t-interval across replications, so it covers both which NFAs were drawn and which edges or accepting states each change picked. Ties when choosing the "deepest" state are broken by state number, so a run gives the same numbers under any `PYTHONHASHSEED`.

**Changes:**

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

**Results** (mean of 10 replications; `trait_confirmation/interventions.csv`, per-replication rows in `interventions_by_replication.csv`, chart `figures/interventions.png`)

| Change | NFAs per replication | Mean × | Median × | % grew (95 % interval) | % shrank | Extreme before → after |
|---|---|---|---|---|---|---|
| keep one accepting state | ≈ 2,670 | 1.71 | 1.19 | 57.0 (56.3–57.7) | 10.3 | 3.8 → 7.7 |
| move accepting state deep | 4,000 | 1.68 | 1.16 | 56.3 (55.8–56.9) | 17.2 | 5.2 → 8.4 |
| cap branching at 2 | ≈ 2,970 | 1.68 | 1.12 | 53.7 (53.0–54.5) | 10.2 | 4.8 → **14.0** |
| delete ε edges | 2,000 | 2.15 | 1.23 | 57.4 (56.8–58.0) | 11.9 | 5.1 → 4.9 * |
| combined | 4,000 | 2.35 | 1.38 | 66.3 (65.6–67.0) | 16.2 | 5.2 → 7.9 * |
| control: add accepting states | ≈ 2,660 | 0.86 | 1.00 | 8.6 (8.1–9.1) | **48.1** | 6.3 → 3.3 |
| control: add random edges | 4,000 | 0.96 | 0.90 | 17.3 (17.0–17.5) | **51.6** | 5.2 → 2.1 |

\* Without ε edges an NFA behaves like a member of the non-ε class, so its "after" extreme status is judged against that class's thresholds. The DFA gets bigger mostly because the NFA moved to a bigger-DFA class, not because it moved up within its class.

*First pass.* The test was first run once on the first 4,000 of the 12,000 Test A NFAs (`interventions_first_pass.csv`). Its numbers agree with the replicated ones within a point or two (for example 52.8 % grew for capping branching, against 53.7 % here), but three changes depended on unpinned tie-breaking and differed slightly between reruns, which is why the test was replicated with fixed tie-breaking.

## What the tests show

- **Direction:** changes toward the profile grew the DFA in 54–66 % of NFAs and shrank it in 10–17 %. The opposite changes shrank it in about half (48–52 %) and grew it in 9–17 %. The intervals across replications are narrow (about ±1 point), so the direction is not an accident of one sample.
- **Size:** the *typical* effect is modest (median 1.1–1.4×, 1.12–1.38× in the table). Means are higher because a few NFAs jump a lot. Reaching the real extremes needs several traits at once.
- **ε edges:** the within-class pattern in the feature search is the cleaner evidence; the deletion test is confounded as described above.
- **Strongly connected component** was not tested (cannot be changed in isolation).

## Limits

- Each change alters the language the NFA accepts, so this shows that *structure drives size*, not that one language gets a smaller or larger DFA.
- The replications share one NFA generator, one set of extreme thresholds (from the 10k run) and one alphabet; the intervals do not cover those choices.
- n = 5–12, alphabet {0, 1}.
