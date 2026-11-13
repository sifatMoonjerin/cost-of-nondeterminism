"""Bar chart of the blow-up family: NFA states (n + 1) next to the DFA states (2^n) for n = 4 … 10.

Reads the hand-built blow-up rows of the balanced 10k run and writes data/experiment_10k/figures/blowup_family.png.

    cd src && python -m experiments.blowup_family_chart
"""
import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np

from experiments.blowup_experiment_10k import EXPERIMENT_10K_DIR

NFA_COLOR = "#F4A942"
DFA_COLOR = "#4285F4"


def load_family(db_path):
    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.sql(
        "SELECT original_states, dfa_states FROM nfa_results WHERE nfa_type = 'blow-up' ORDER BY original_states"
    ).fetchall()
    con.close()
    return rows


def plot_family(rows, path):
    ns = [nfa_states - 1 for nfa_states, _ in rows]
    nfa_counts = [nfa_states for nfa_states, _ in rows]
    dfa_counts = [dfa_states for _, dfa_states in rows]
    x = np.arange(len(rows))
    width = 0.38

    plt.rcParams["font.family"] = "Georgia"
    fig, ax = plt.subplots(figsize=(8, 5.5))
    nfa_bars = ax.bar(x - width / 2, nfa_counts, width, color=NFA_COLOR, label="NFA states (n + 1)")
    dfa_bars = ax.bar(x + width / 2, dfa_counts, width, color=DFA_COLOR, label="DFA states (2ⁿ, identical before and after minimization)")
    ax.bar_label(nfa_bars, fontsize=9, padding=2)
    ax.bar_label(dfa_bars, fontsize=9, padding=2)

    ax.set_xticks(x, [str(n) for n in ns])
    ax.set_xlabel("n (the NFA has n + 1 states)")
    ax.set_ylim(0, max(dfa_counts) * 1.08)
    ax.set_title("NFA states and DFA states vs. n", fontsize=12)
    ax.grid(axis="y", color="#D9D9D9")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Blow-up family: NFA states vs DFA states")
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_10K_DIR)
    args = parser.parse_args()

    rows = load_family(args.experiment_dir / "blowup.duckdb")
    assert all(dfa == 2 ** (nfa - 1) for nfa, dfa in rows), "blow-up DFA sizes should be 2^n"
    path = args.experiment_dir / "figures" / "blowup_family.png"
    plot_family(rows, path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
