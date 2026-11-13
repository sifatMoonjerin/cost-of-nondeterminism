"""One table (PNG) comparing the structured-NFA experiments for 5, 6 and 7 states with the unrestricted (no ε) generator.

Each size has three rows: structured, unrestricted, and structured ÷ unrestricted. Both sides are 1,000,000 NFAs per size.
Reads summary.json and size_distribution.csv from data/experiment_structured_<n>_states/ and writes
data/structured_experiments_table.png.

    cd src && python -m experiments.structured_summary_table
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.blowup_experiment_10k import use_georgia
from experiments.structured_nfa_experiment import DATA_DIR

STATES = (5, 6, 7)
HEADER_COLOR = "#2F3E4E"
STRUCTURED_COLOR = "#FBE3E1"
UNRESTRICTED_COLOR = "#F1F1F1"
RATIO_COLOR = "#FFF4CC"
LABEL_COLOR = "#E3EAF2"
COLUMNS = ["NFA states", "Generator", "NFAs", "Median DFA\nstates", "DFA ≥ ¼ of\nthe ceiling", "DFA ≥ ½ of\nthe ceiling",
           "DFA ≥ ¾ of\nthe ceiling", "DFA at the\nceiling"]


def share(percent):
    return f"{percent:.3g}%" if percent >= 0.01 else f"{percent:.2g}%"


def times(structured, baseline, decimals=0):
    return f"{structured / baseline:.{decimals}f}×" if baseline else "–"


def ceiling_cell(hits, total):
    return f"{hits:,}"


def build_rows():
    rows, kinds = [], []
    for n in STATES:
        summary = json.loads((DATA_DIR / f"experiment_structured_{n}_states" / "summary.json").read_text())
        sizes = pd.read_csv(DATA_DIR / f"experiment_structured_{n}_states" / "size_distribution.csv")
        structured, baseline = sizes.iloc[0], sizes.iloc[1]
        ceiling = summary["ceiling"]
        cuts = [ceiling // 4, ceiling // 2, 3 * ceiling // 4]

        label = f"{n} states"
        for who, row in (("structured", structured), ("unrestricted", baseline)):
            hits = int(row["at_ceiling"])
            rows.append([label if who == "unrestricted" else "", who, f"{row['nfas']:,.0f}", f"{row['median_dfa_states']:.0f}"]
                        + [share(row[f"pct_at_least_{cut}"]) for cut in cuts] + [ceiling_cell(hits, int(row["nfas"]))])
            kinds.append("structured" if who == "structured" else "unrestricted")
        rows.append(["", "ratio", "", times(structured["median_dfa_states"], baseline["median_dfa_states"], 1)]
                    + [times(structured[f"pct_at_least_{cut}"], baseline[f"pct_at_least_{cut}"]) for cut in cuts]
                    + [times(summary["at_ceiling"], summary["baseline_at_ceiling"]) if summary["baseline_at_ceiling"] else f"{summary['at_ceiling']} vs 0"])
        kinds.append("ratio")
    return rows, kinds


def main():
    use_georgia()
    rows, kinds = build_rows()
    widths = [0.15, 0.12, 0.10, 0.10, 0.12, 0.12, 0.12, 0.13]

    fig, ax = plt.subplots(figsize=(13, 8.6))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=COLUMNS, colWidths=widths, cellLoc="center", loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if row == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(0.085)
            continue
        kind = kinds[row - 1]
        cell.set_height(0.105 if kind != "ratio" else 0.07)
        cell.set_facecolor({"structured": STRUCTURED_COLOR, "unrestricted": UNRESTRICTED_COLOR, "ratio": RATIO_COLOR}[kind])
        if kind == "ratio":
            cell.set_text_props(fontweight="bold")
        if col == 0:
            # One label per block, in the middle row; the three cells share a colour
            cell.set_facecolor(LABEL_COLOR)
            cell.set_text_props(fontweight="bold", va="center")

    ax.set_title("Structured vs unrestricted NFAs",
                 fontsize=14, fontweight="bold", pad=6)
    fig.tight_layout()
    path = DATA_DIR / "structured_experiments_table.png"
    fig.savefig(path, dpi=200)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
