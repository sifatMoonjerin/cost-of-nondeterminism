# Terms

Every term used in the experiments, tied to the code that defines it. `n` is the number of NFA states, the alphabet is Σ = {0, 1} (so |Σ| = 2), and q0 is always the start state.

## 1. How the random NFAs are generated

Code: `generate_nfa`, `generate_random_transitions`, `initialize_nfa` in `src/core/automata.py` (same code as `nfa-generator.ipynb`).

1. Create states `0 … n-1`; start state is 0; pick **1–3 random accepting states**.
2. For **each state**, draw `d` transitions, where `d` is uniform in `[min_transitions_per_state, max_transitions_per_state]`. Each draw is a random **(symbol, target)** pair, target uniform over all `n` states.
3. If `allow_epsilon`, add **1–3 ε edges** between two *different* random states.
4. Keep the NFA only if **every state is reachable from q0** (ε edges count); otherwise redraw (up to 1000 attempts).

Draws go into a set, so a repeated (symbol, target) for the same state collapses to one edge. The number of edges is therefore **at most** `n·d`, usually slightly less.

## 2. Sparse, dense and mixed (the density classes)

The class depends only on the generator parameters (`classify_sweep_nfa` in `src/experiments/blowup_experiment.py`):

| Class | Rule on (min, max) transitions per state | Notebook / `data/generated` |
|---|---|---|
| **sparse** | `max ≤ 3` | 1–3 per state |
| **dense** | `min ≥ 4` | 4–7 per state |
| **mixed** | everything else (range straddles 3 and 4) | not generated there |

The same rule with ε edges added gives **sparse-ε, mixed-ε, dense-ε**. That makes six random classes plus the hand-built **blow-up** family.

### Density as a ratio

An NFA over Σ has **S = n · |Σ| · n = 2n²** possible symbol edges `(state, symbol, target)`.

| Ratio | Definition | Meaning |
|---|---|---|
| **draws per state `d`** | the generator parameter | 1–3 sparse, ≥ 4 dense |
| **draws per (state, symbol) slot** | `d / |Σ| = d / 2` | sparse ≤ 1.5, dense ≥ 2.0 |
| **edge density** | `symbol edges / (2n²)` | share of all possible edges present |
| **symbol coverage** | `(state, symbol) pairs with any edge / (n·|Σ|)` | share of slots that are not dead ends |
| **average branching** | `symbol edges / (state, symbol) pairs with any edge` | destinations per defined slot |

So the **sparse/dense cut at 3 vs 4 transitions per state is a cut at 1.5 vs 2 draws per (state, symbol) slot**.

Measured averages in the balanced 10k run (`data/experiment_10k`, n = 5–12, mean n ≈ 8.4):

| Class | Symbol edges | Edge density | Symbol coverage | Avg branching | Non-det. pairs |
|---|---|---|---|---|---|
| sparse | 16.9 | 13 % | 0.73 | 1.37 | 4.3 |
| mixed | 29.1 | 21 % | 0.89 | 1.90 | 9.5 |
| dense | 37.4 | 28 % | 0.96 | 2.28 | 12.6 |

The ε variants have the same densities, plus about 2 ε edges on average (1–3).

In the sweep, each NFA draws its own parameters: `min ∈ {1,…,4}`, `max = min + {0,…,4}`, ε with probability 0.5, `n` uniform in the chosen range (`generate_experiment_nfas`).

## 3. Sizes and outcomes

| Term | Definition |
|---|---|
| **Subset DFA** | DFA from subset construction (`nfa_to_dfa`), before minimizing. Includes the empty set (dead state) if reachable. |
| **Minimized DFA** | Moore partition refinement of the subset DFA (`minimize_dfa`). This is the size used for most analysis (`dfa_states`). |
| **Size ratio** | `minimized DFA states / n`. Raw version: `subset_states / n`. |
| **Larger / same / smaller** | minimized DFA states `>` / `=` / `<` n. |
| **Ceiling** | `2^n`, the most states a DFA built from an n-state NFA can have. |
| **Share of ceiling** | `DFA states / 2^n` (1.0 = reaches the ceiling). |
| **Exponent fraction** | `log2(DFA states) / n` (1.0 = ceiling, 0.5 = halfway, i.e. `2^(n/2)` states). |
| **Reaches the ceiling** | minimized DFA has exactly `2^n` states. |

## 4. Structural features of an NFA

Computed in `get_structural_features` (`blowup_experiment.py`) and `get_graph_features` (`trait_confirmation.py`).

| Feature | Definition |
|---|---|
| **Non-deterministic pair** | a (state, symbol) with **more than one** destination. ε edges are not counted. |
| **Non-det. pairs per state** | `non-deterministic pairs / n` |
| **Accepting states** | number of accepting states (1–3 in generated NFAs) |
| **Accept distance** | shortest path (BFS, ε edges included) from q0 to the nearest accepting state; 0 = q0 accepts |
| **ε edges** | number of ε transitions |
| **Strongly connected component (SCC)** | maximal set of states that can all reach each other. "Single SCC" means the whole NFA is one such set. |
| **Out-degree / in-degree** | edges leaving / entering a state (ε included) |

## 5. Groups, "extreme", "profile"

| Term | Definition |
|---|---|
| **Group** | NFAs with the same **class and same n**. Comparisons are always made inside a group so that "dense NFAs are small" and "big NFAs are bigger" cannot explain a result. |
| **Extreme** | minimized DFA size is in the **top 5 %** of its group. |
| **Profile** | one accepting state, accept distance ≥ 2, and 0.25–1.0 non-deterministic pairs per state. Defined from the 10k analysis, then tested on fresh NFAs. |
| **Blow-up family** | hand-built NFA "the n-th symbol from the end is 1": states q0…qn (**n + 1 states**), DFA has `2^n` states (half of its `2^(n+1)` ceiling). |

## 6. Reproducibility and "distinct"

| Term | Definition |
|---|---|
| **Seed** | `generate_nfa(seed=…)` seeds a private `random.Random`. Since the sorted-alphabet fix (commit `e3d4593`) a seed always gives the same NFA. Earlier runs could give the 0/1-swapped copy; DFA sizes are unaffected. |
| **NFA key** | sha256 of a canonical sorted form (states, alphabet, start, accepting, transitions). Two NFAs are *distinct* when their keys differ. |
| **Canonical key** | the smaller of an NFA's key and its 0/1-swapped copy's key; used by `ceiling_hunt.py` so a mirror image counts as a duplicate. |
