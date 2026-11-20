# The paper: where each figure and table comes from

Source: `final-report-edited.tex` (this folder), figures in `figures/`. The template class (`ieeeconf.cls`) and `refs.bib` are not in the repository.
Run scripts from `src/`. "Data" is under `data/`, which is not committed: regenerate it with the commands in [experiments/](experiments/).

| Paper | File in `figures/` | Made by | Data |
|---|---|---|---|
| Fig. 1 running example NFA | `running_example_nfa.png` | NFA drawing of the blow-up NFA for n = 3 (`visualization/visualize_nfa.py` draws NFAs in this style) | |
| Fig. 2 subset construction | `subset_table.png` | `visualization/render_running_example.py` (n = 3 gives the 8-state DFA) | `data/visualizations/running_example/n03_8_states.png` |
| Fig. 3 blow-up family | `blowup_family.png` | `experiments/blowup_family_chart.py` | `data/experiment_10k` |
| Fig. 4 vs subset / minimized DFA | `vs_subset.png`, `vs_minimized.png` | `experiments/blowup_experiment_10k.py` | `data/experiment_10k/figures/subset_size_outcome_by_type.png`, `size_outcome_by_type.png` |
| Fig. 5 left, violin | `violin.png` | `experiments/blowup_experiment_10k.py --scatter-only` | `figures/larger_dfa_violin_minimized.png` |
| Fig. 5 right, ceiling scatter | `ceiling_scatter.png` | same | `figures/ceiling_scatter.png` |
| Fig. 6 one-at-a-time changes | `trait_tests.png` | `experiments/trait_confirmation.py --interventions-only` | `figures/interventions.png` |
| Fig. 7 hexbin, 6 states | `hexbin_6_states.png` | `experiments/blowup_experiment_1m.py --scatter-only --out-dir ../data/experiment_1m_6_states` | `data/experiment_1m_6_states/figures/ceiling_hexbin.png` |
| Fig. 8 ceiling NFA | `ceiling_nfa.png` | `visualization/render_ceiling_hits.py` | `data/visualizations/ceiling_nfa/` |
| Table I runs | | by hand | [experiments/](experiments/) |
| Table II first experiment | | `nfa-generator.ipynb`; averages recomputed from `data/generated/*.json` | |
| Table III traits | | `experiments/trait_confirmation.py` | `data/experiment_10k/trait_confirmation/trait_rates.csv` |
| Table IV structured vs unrestricted | | `experiments/structured_summary_table.py` (PNG version `figures/structured_table.png`) | `data/experiment_structured_<n>_states/`, `experiment_unrestricted_<n>_states_2m/` |

Figures in `figures/` not used in the paper: `ceiling_dfa.png`, `extreme_nfa.png`, `structured_distribution.png`, `structured_frequency.png`.

Every number quoted in the text was checked against these data (see [experiments/07-validation.md](experiments/07-validation.md) for the rerun and the correctness checks).
