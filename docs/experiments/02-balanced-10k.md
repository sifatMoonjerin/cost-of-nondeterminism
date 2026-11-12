# 2. Balanced 10k run

**Question.** The same question as run 1, but with equal numbers of every class so classes can be compared fairly. This is the dataset the trait analysis uses.

**Code.** `src/experiments/blowup_experiment_10k.py` (reuses run 1's pipeline) · **Results.** `data/experiment_10k/`

```
cd src
python -m experiments.blowup_experiment_10k            # --count 10000 --seed 1
```

| | |
|---|---|
| NFAs | 10,000 random, 5–12 states, **1,667 / 1,666 per class** (6 classes) + the 7 blow-up NFAs |
| Distinct | all ids unique; seed 1, a different sample from run 1 |
| Generation | a draw is kept only while its class still has room (`generate_balanced_nfas`) |

**Outputs.** `blowup.duckdb`, `nfas.json`, and figures: the four from run 1 plus `larger_dfa_violin.png` (how much larger the DFA is, per class, minimized and raw) and `ceiling_approach.png` (DFA size against the `2^n` ceiling, with how far toward the ceiling the larger DFAs get). `trait_confirmation/` holds the CSVs from [feature-search](../feature-search.md) and [trait-tests](../trait-tests.md).

**Findings.** Share with a larger minimized DFA: sparse 82 %, sparse-ε 56 %, mixed 51 %, mixed-ε 25 %, dense 18 %, dense-ε 6 %. Median size ratio among the larger ones: 2.3×, 1.8×, 2.0×, 1.6×, 1.5×, 1.3×. The best random NFA reaches 69 % of its ceiling; 41 of 10,000 get more than three quarters of the way there in the exponent.

**Note.** Rerun after the blow-up state-count fix: the 10,000 random rows are identical to the first run, only the blow-up rows changed.
