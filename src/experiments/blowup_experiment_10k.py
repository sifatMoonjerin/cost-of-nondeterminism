"""Second run of the NFA to DFA blow-up experiment: 10,000 random NFAs split evenly across the six non-blow-up classes.

Reuses the pipeline in blowup_experiment.py (features, DuckDB views, original plots) and writes to its own folder,
so the first run in data/experiment/ is left untouched.
"""
import argparse
import random
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager

from core.automata import generate_nfa, get_nfa_id
from experiments.blowup_experiment import (
    ROOT_DIR,
    DATABASE_NAME,
    build_dataframe,
    classify_sweep_nfa,
    load_json_nfas,
    plot_larger_rates,
    plot_relative_size,
    plot_size_outcome_by_type,
    save_nfas_json,
    save_to_duckdb,
)

EXPERIMENT_10K_DIR = ROOT_DIR / "data" / "experiment_10k"

# The six non-blow-up classes, sparsest to densest. The blow-up family is added separately.
RANDOM_CLASSES = ["sparse", "sparse-epsilon", "mixed", "mixed-epsilon", "dense", "dense-epsilon"]

CLASS_COLORS = {
    "sparse": "#D9534F", "sparse-epsilon": "#E58A87",
    "mixed": "#F0AD4E", "mixed-epsilon": "#F5CB8A",
    "dense": "#4285F4", "dense-epsilon": "#8FB4F8",
}


# ---------------------------------------------------------------- Generation

def class_quotas(total, classes):
    # Splits total as evenly as possible; the first (total % len) classes get one extra NFA
    base, extra = divmod(total, len(classes))
    return {name: base + (1 if i < extra else 0) for i, name in enumerate(classes)}


def generate_balanced_nfas(total, seed=1, min_states=5, max_states=12):
    # Same parameter distribution as generate_experiment_nfas, but a draw is only kept while its class still has room,
    # so every class ends up with its quota. Duplicate NFAs (same structural id) are rejected.
    quotas = class_quotas(total, RANDOM_CLASSES)
    counts = {name: 0 for name in RANDOM_CLASSES}
    param_rng = random.Random(seed)
    nfa_seed = seed * 1_000_000
    entries = []
    seen_ids = set()
    num_failed = 0
    num_duplicates = 0

    while len(entries) < total:
        params = {
            "seed": nfa_seed,
            "num_states": param_rng.randint(min_states, max_states),
            "min_transitions_per_state": param_rng.randint(1, 4),
            "allow_epsilon": param_rng.random() < 0.5,
        }
        params["max_transitions_per_state"] = params["min_transitions_per_state"] + param_rng.randint(0, 4)
        nfa_seed += 1

        nfa_type = classify_sweep_nfa(params)

        if counts[nfa_type] >= quotas[nfa_type]:
            continue

        try:
            nfa = generate_nfa(**params)
        except RuntimeError:
            num_failed += 1
            continue

        nfa_id = get_nfa_id(nfa)

        if nfa_id in seen_ids:
            num_duplicates += 1
            continue

        seen_ids.add(nfa_id)
        counts[nfa_type] += 1
        entries.append((nfa, dict(params, source="sweep", nfa_type=nfa_type)))

    return entries, num_failed, num_duplicates


# ---------------------------------------------------------------- Extra plots

def use_georgia():
    for name in ["Georgia.ttf", "Georgia Bold.ttf"]:
        path = Path("/System/Library/Fonts/Supplemental") / name
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams["font.family"] = ["Georgia", "DejaVu Sans"]


def style_axis(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_larger_violin(con, path):
    # How much larger is the DFA, for the NFAs where it is larger? Log-scaled ratio, minimized vs subset-construction DFA.
    df = con.sql("SELECT nfa_type, original_states, subset_states, dfa_states FROM random_nfas").df()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    ticks = [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24]
    top = np.log2(df["subset_states"].div(df["original_states"]).max()) + 0.3

    for ax, column, title in [
        (axes[0], "dfa_states", "Minimized DFA"),
        (axes[1], "subset_states", "Raw subset-construction DFA"),
    ]:
        ratio = df[column] / df["original_states"]
        totals = df.groupby("nfa_type").size()
        larger = df[ratio > 1].assign(log_ratio=np.log2(ratio[ratio > 1]))
        medians = larger.assign(ratio=2 ** larger["log_ratio"]).groupby("nfa_type")["ratio"].median()

        sns.violinplot(data=larger, x="nfa_type", y="log_ratio", order=RANDOM_CLASSES, hue="nfa_type",
                       palette=CLASS_COLORS, legend=False, inner=None, cut=0, density_norm="width",
                       linewidth=1, saturation=1, ax=ax)
        sns.stripplot(data=larger, x="nfa_type", y="log_ratio", order=RANDOM_CLASSES, color="black",
                      size=1.2, alpha=0.2, jitter=0.18, ax=ax)

        labels = []
        for i, name in enumerate(RANDOM_CLASSES):
            count = int((larger["nfa_type"] == name).sum())
            ax.hlines(np.log2(medians[name]), i - 0.3, i + 0.3, color="black", linewidth=2, zorder=5)
            ax.text(i + 0.34, np.log2(medians[name]), f"{medians[name]:.1f}×", va="center", fontsize=10, fontweight="bold")
            labels.append(f"{name}\n{100 * count / totals[name]:.0f}% larger\n(n={count} of {totals[name]})")

        ax.set_xticks(range(len(RANDOM_CLASSES)), labels, fontsize=9)
        ax.set_yticks(np.log2(ticks), [f"{t:g}×" for t in ticks])
        ax.set_ylim(-0.05, top)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("")
        ax.set_title(title, fontsize=12)
        style_axis(ax)

    axes[0].set_ylabel("DFA states ÷ NFA states (log scale)")
    fig.suptitle("How much larger? Size of the DFA relative to its NFA, for the NFAs whose DFA is larger (black bar = median)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_ceiling_approach(con, path):
    # Minimized DFA size against the 2^n ceiling, and how far toward it the "larger" cases get
    df = con.sql("SELECT nfa_type, original_states AS n, dfa_states AS m FROM random_nfas").df()
    df["larger"] = df["m"] > df["n"]
    # Share of the ceiling in the exponent: 1.0 means 2^n states, 0.5 means 2^(n/2)
    df["fraction"] = np.log2(df["m"]) / df["n"]
    jitter = np.random.default_rng(0).uniform(-0.28, 0.28, len(df))
    log_states = np.log2(df["m"])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15.5, 6.2), gridspec_kw={"width_ratios": [1.2, 1]})

    ax_a.scatter((df["n"] + jitter)[~df["larger"]], log_states[~df["larger"]], s=8, color="#9E9E9E", alpha=0.35,
                 label="DFA same size or smaller", linewidths=0)
    ax_a.scatter((df["n"] + jitter)[df["larger"]], log_states[df["larger"]], s=10, color="#D9534F", alpha=0.5,
                 label="DFA larger than NFA", linewidths=0)
    n_range = np.linspace(df["n"].min() - 0.4, df["n"].max() + 0.4, 100)
    ax_a.plot(n_range, n_range, color="black", linewidth=1.6, linestyle="--", label="ceiling: 2$^n$ states")
    ax_a.plot(n_range, n_range / 2, color="#444444", linewidth=1.2, linestyle=":", label="halfway (exponent): 2$^{n/2}$")
    ax_a.plot(n_range, np.log2(n_range), color="black", linewidth=1.2, label="NFA size: n states")
    ax_a.set_xlim(n_range[0], n_range[-1])
    ax_a.set_ylim(0, df["n"].max() + 1.2)
    ticks = list(range(0, int(df["n"].max()) + 1, 2))
    ax_a.set_yticks(ticks, [f"{2 ** t:,}" for t in ticks])
    ax_a.set_xlabel("NFA states (n)")
    ax_a.set_ylabel("Minimized DFA states (log scale)")
    ax_a.set_title("A. Minimized DFA size against the 2$^n$ ceiling", fontsize=12)
    ax_a.legend(loc="upper left", frameon=False, fontsize=9)
    style_axis(ax_a, "both")

    larger = df[df["larger"]]
    sns.violinplot(data=larger, x="nfa_type", y="fraction", order=RANDOM_CLASSES, hue="nfa_type", palette=CLASS_COLORS,
                   legend=False, inner=None, cut=0, density_norm="width", linewidth=1, saturation=1, ax=ax_b)
    sns.stripplot(data=larger, x="nfa_type", y="fraction", order=RANDOM_CLASSES, color="black", size=1.4,
                  alpha=0.2, jitter=0.18, ax=ax_b)
    for i, name in enumerate(RANDOM_CLASSES):
        group = larger[larger["nfa_type"] == name]
        ax_b.hlines(group["fraction"].median(), i - 0.3, i + 0.3, color="black", linewidth=2, zorder=5)
        ax_b.text(i, 1.035, f"{100 * (group['fraction'] >= 0.5).mean():.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax_b.text(-0.55, 1.07, "share at or above halfway:", ha="left", fontsize=9, style="italic")
    ax_b.set_xticks(range(len(RANDOM_CLASSES)),
                    [f"{name}\n(n={int((larger['nfa_type'] == name).sum())})" for name in RANDOM_CLASSES], fontsize=9)
    ax_b.axhline(1, color="black", linestyle="--", linewidth=1.6)
    ax_b.axhline(0.5, color="#444444", linestyle=":", linewidth=1.2)
    ax_b.text(5.5, 0.985, "ceiling (2$^n$)", va="top", ha="right", fontsize=9)
    ax_b.text(5.5, 0.51, "halfway (2$^{n/2}$)", va="bottom", ha="right", fontsize=9)
    ax_b.set_ylim(0, 1.14)
    ax_b.set_xlabel("")
    ax_b.set_ylabel("log$_2$(DFA states) ÷ n   (1.0 = the 2$^n$ ceiling)")
    ax_b.set_title("B. How close do the 'larger' DFAs get?", fontsize=12)
    style_axis(ax_b)

    fig.suptitle(f"Do larger DFAs approach the 2$^n$ ceiling?  ({len(df):,} random NFAs, {df['n'].min()}–{df['n'].max()} states; blow-up family excluded)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return {
        "larger": len(larger),
        "median_fraction": larger["fraction"].median(),
        "pct_larger_at_or_above_halfway": 100 * (larger["fraction"] >= 0.5).mean(),
        "num_at_or_above_75": int((df["fraction"] >= 0.75).sum()),
        "max_fraction": df["fraction"].max(),
    }


# ---------------------------------------------------------------- Main

def main():
    parser = argparse.ArgumentParser(description="Balanced 10k-NFA blow-up experiment")
    parser.add_argument("--count", type=int, default=10_000, help="random NFAs, split evenly across the six classes")
    parser.add_argument("--seed", type=int, default=1, help="differs from the first run (0) so the NFAs are a fresh sample")
    parser.add_argument("--min-states", type=int, default=5)
    parser.add_argument("--max-states", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=EXPERIMENT_10K_DIR)
    args = parser.parse_args()

    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    entries, num_failed, num_duplicates = generate_balanced_nfas(args.count, args.seed, args.min_states, args.max_states)
    print(f"Generated {len(entries)} unique NFAs ({num_failed} draws had no fully reachable NFA, {num_duplicates} duplicates rejected)")

    # The hand-built blow-up family is kept so the by-type charts still show it
    blow_up = [(nfa, params) for nfa, params in load_json_nfas() if params["nfa_type"] == "blow-up"]
    entries += blow_up
    print(f"Added {len(blow_up)} blow-up NFAs from data/generated")

    ids = [get_nfa_id(nfa) for nfa, _ in entries]
    assert len(ids) == len(set(ids)), "duplicate NFAs in the dataset"

    save_nfas_json(entries, args.out_dir / "nfas.json")

    df = build_dataframe(entries)
    con = save_to_duckdb(df, args.out_dir / DATABASE_NAME)

    use_georgia()
    plot_relative_size(con, figures_dir / "relative_size.png")
    plot_larger_rates(con, figures_dir / "larger_rates.png")
    plot_size_outcome_by_type(
        con, "size_outcome_by_type", "minimized DFA",
        "Random NFA's Compared to Minimized DFA", figures_dir / "size_outcome_by_type.png",
    )
    plot_size_outcome_by_type(
        con, "subset_size_outcome_by_type", "subset-construction DFA",
        "Random NFA's Compared to Subset Construction DFA", figures_dir / "subset_size_outcome_by_type.png",
    )
    plot_larger_violin(con, figures_dir / "larger_dfa_violin.png")
    ceiling = plot_ceiling_approach(con, figures_dir / "ceiling_approach.png")

    for title, view in [
        ("Share of NFAs whose minimized DFA is larger, by NFA type", "larger_rate_by_type"),
        ("... by average out-degree", "larger_rate_by_out_degree"),
        ("... by NFA size", "larger_rate_by_states"),
        ("... by epsilon use", "larger_rate_by_epsilon"),
        ("Minimized DFA: smaller / same size / larger, by NFA type", "size_outcome_by_type"),
        ("Subset-construction DFA: smaller / same size / larger, by NFA type", "subset_size_outcome_by_type"),
        ("Average features: larger vs. not larger (states)", "features_larger_vs_not"),
        ("Largest relative size", "largest_relative_size"),
    ]:
        print()
        print(title + ":")
        print(con.sql(f"SELECT * FROM {view}").df().to_string(index=False))

    print()
    print("Distance to the 2^n ceiling (random NFAs):", {key: round(float(value), 3) for key, value in ceiling.items()})

    con.close()
    print()
    print(f"Wrote {args.out_dir / DATABASE_NAME} and figures to {figures_dir}")


if __name__ == "__main__":
    main()
