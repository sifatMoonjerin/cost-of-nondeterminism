# 5. Visualizations

**Code.** `src/visualization/` · **Output.** `data/visualizations/`, plus PDFs inside experiment folders.

| File | Use |
|---|---|
| `visualize_dfa.py` | `render_dfa(dfa, "out.png", names="letters", layout="neato")`. Letter names, NFA-state sets under each name, dead states dashed, layouts `dot` / `neato` / `fdp` / `sfdp` / `circo` / `twopi` (use `sfdp` beyond ~100 states). Command line: `python -m visualization.visualize_dfa ../data/generated/blowup_nfas.json --index 0 --minimize`. |
| `visualize_nfa.py` | `render_nfa(nfa, "out.png")`. Red edges = one of several destinations for the same (state, symbol); dashed blue = ε. |
| `render_running_example.py` | DFAs of the blow-up NFA for n = 1…10 into `data/visualizations/running_example/` (PNG up to n = 6, SVG beyond). |
| `render_ceiling_hits.py` | The ceiling-hunt PDF (experiment 4). |

Every state is drawn at the same size (scaled to the longest label); accepting states are slightly smaller inside so the double ring brings them to the same outer size. Needs the Graphviz binaries (`brew install graphviz`).
