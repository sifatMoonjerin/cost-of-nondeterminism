"""Million-NFA run of the NFA to DFA blow-up experiment, in its own folder (data/experiment_1m/).

1,000,000 distinct random NFAs, split evenly across the six non-blow-up classes (166,667 or 166,666 each), each converted
with subset construction and minimized. Two charts: how much larger the DFA is (violins) and how close it gets to the
2^n ceiling (scatter plus violins). Uses all CPU cores. The first two runs are left untouched.

    python -m experiments.blowup_experiment_1m                  # the full run
    python -m experiments.blowup_experiment_1m --count 6000     # quick check
"""
import argparse
import hashlib
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from core.automata import generate_nfa, minimize_dfa, nfa_to_dfa
from experiments.blowup_experiment import DATABASE_NAME, ROOT_DIR, classify_sweep_nfa, get_structural_features
from experiments.blowup_experiment_10k import CLASS_COLORS, RANDOM_CLASSES, class_quotas, style_axis, use_georgia

EXPERIMENT_1M_DIR = ROOT_DIR / "data" / "experiment_1m"
SHARD_SIZE = 5_000
SEEDS_PER_SHARD = 10_000_000      # each shard draws NFA seeds from its own block, so shards never repeat an NFA
VIOLIN_SAMPLE = 30_000            # points per class used to draw each violin's shape (medians use every point)
SCATTER_SAMPLE = 40_000           # points drawn in the scatter (the envelope uses every point)


# ---------------------------------------------------------------- Generation (runs in worker processes)

def get_dedup_key(nfa):
    # Full sha256 of a canonical form: the 10-hex-digit NFA id is too short to rule out a collision among a million NFAs
    canonical = json.dumps([
        sorted(nfa.states),
        sorted(nfa.alphabet),
        nfa.start_state,
        sorted(nfa.accept_states),
        sorted(
            (from_state, symbol, to_state)
            for from_state, symbol_dict in nfa.transitions.items()
            for symbol, destinations in symbol_dict.items()
            for to_state in destinations
        ),
    ])

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def run_shard(task):
    # One shard: `count` distinct NFAs of one class, with the same parameter distribution as the earlier sweeps
    nfa_type, shard_id, count, seed, min_states, max_states = task
    param_rng = random.Random(seed * 1_000_003 + shard_id)
    nfa_seed = (shard_id + 1) * SEEDS_PER_SHARD
    seen_keys = set()
    rows = []
    num_failed = 0

    while len(rows) < count:
        params = {
            "seed": nfa_seed,
            "num_states": param_rng.randint(min_states, max_states),
            "min_transitions_per_state": param_rng.randint(1, 4),
            "allow_epsilon": param_rng.random() < 0.5,
        }
        params["max_transitions_per_state"] = params["min_transitions_per_state"] + param_rng.randint(0, 4)
        nfa_seed += 1

        if classify_sweep_nfa(params) != nfa_type:
            continue

        try:
            nfa = generate_nfa(**params)
        except RuntimeError:
            num_failed += 1
            continue

        key = get_dedup_key(nfa)

        if key in seen_keys:
            continue

        seen_keys.add(key)
        subset_dfa = nfa_to_dfa(nfa)
        min_dfa = minimize_dfa(subset_dfa)
        features = get_structural_features(nfa)

        rows.append({
            "nfa_key": key,
            "nfa_type": nfa_type,
            "seed": params["seed"],
            "original_states": len(nfa.states),
            "min_transitions_per_state": params["min_transitions_per_state"],
            "max_transitions_per_state": params["max_transitions_per_state"],
            **features,
            "subset_states": len(subset_dfa.states),
            "dfa_states": len(min_dfa.states),
        })

    return rows, num_failed


def generate_distinct_nfas(total, seed, min_states, max_states, workers):
    # Shards run in parallel; duplicates across shards are removed and topped up until every class has its quota
    quotas = class_quotas(total, RANDOM_CLASSES)
    kept = {name: [] for name in RANDOM_CLASSES}
    seen_keys = set()
    num_failed = 0
    num_duplicates = 0
    next_shard = {name: 0 for name in RANDOM_CLASSES}
    start = time.time()

    def make_tasks():
        tasks = []
        for index, name in enumerate(RANDOM_CLASSES):
            missing = quotas[name] - len(kept[name])
            while missing > 0:
                size = min(SHARD_SIZE, missing)
                tasks.append((name, next_shard[name] + index * 100_000, size, seed, min_states, max_states))
                next_shard[name] += 1
                missing -= size
        return tasks

    with Pool(workers) as pool:
        while True:
            tasks = make_tasks()

            if not tasks:
                break

            for rows, failed in pool.imap_unordered(run_shard, tasks):
                num_failed += failed

                for row in rows:
                    if row["nfa_key"] in seen_keys:
                        num_duplicates += 1
                    elif len(kept[row["nfa_type"]]) < quotas[row["nfa_type"]]:
                        seen_keys.add(row["nfa_key"])
                        kept[row["nfa_type"]].append(row)

                done = sum(len(rows_) for rows_ in kept.values())
                print(f"  {done:>9,} / {total:,} NFAs  ({time.time() - start:.0f}s)", flush=True)

    records = [row for name in RANDOM_CLASSES for row in kept[name]]
    return pd.DataFrame(records), num_failed, num_duplicates


# ---------------------------------------------------------------- Summaries

def add_size_columns(df):
    df["state_ratio"] = df["dfa_states"] / df["original_states"]
    df["subset_ratio"] = df["subset_states"] / df["original_states"]
    df["larger_in_states"] = df["dfa_states"] > df["original_states"]
    df["subset_larger"] = df["subset_states"] > df["original_states"]
    df["total_transitions"] = df["num_symbol_transitions"] + df["num_epsilon_transitions"]
    df["ceiling"] = 2.0 ** df["original_states"]
    df["share_of_ceiling"] = df["dfa_states"] / df["ceiling"]
    # Share of the ceiling in the exponent: 1.0 is 2^n states, 0.5 is 2^(n/2)
    df["exponent_fraction"] = np.log2(df["dfa_states"]) / df["original_states"]
    return df


def summarize(df, size_column="original_states"):
    by_class = df.groupby("nfa_type").apply(
        lambda group: pd.Series({
            "nfas": len(group),
            "pct_subset_larger": 100 * group["subset_larger"].mean(),
            "pct_minimized_larger": 100 * group["larger_in_states"].mean(),
            "median_ratio_when_larger": group.loc[group["larger_in_states"], "state_ratio"].median(),
            "p99_ratio": group["state_ratio"].quantile(0.99),
            "max_ratio": group["state_ratio"].max(),
            "median_exponent_fraction_when_larger": group.loc[group["larger_in_states"], "exponent_fraction"].median(),
            "pct_larger_at_or_above_halfway": 100 * (group.loc[group["larger_in_states"], "exponent_fraction"] >= 0.5).mean(),
            "max_exponent_fraction": group["exponent_fraction"].max(),
            "max_share_of_ceiling": group["share_of_ceiling"].max(),
        }),
        include_groups=False,
    ).reindex(RANDOM_CLASSES)

    by_size = df.groupby(size_column).agg(
        nfas=("dfa_states", "size"),
        max_dfa_states=("dfa_states", "max"),
        ceiling=("ceiling", "first"),
        max_share_of_ceiling=("share_of_ceiling", "max"),
        p99_9_exponent_fraction=("exponent_fraction", lambda values: values.quantile(0.999)),
        max_exponent_fraction=("exponent_fraction", "max"),
    )

    return by_class.round(3), by_size.round(4)


# ---------------------------------------------------------------- Charts

def sample_per_class(df, size, seed=0):
    parts = [group.sample(min(size, len(group)), random_state=seed) for _, group in df.groupby("nfa_type")]
    return pd.concat(parts)


def draw_violins(ax, larger, value_column, full_larger, transform, order=RANDOM_CLASSES):
    # Violin shapes come from a sample per class; medians and counts come from every NFA
    sample = sample_per_class(larger, VIOLIN_SAMPLE)
    sample = sample.assign(plot_value=transform(sample[value_column]))
    sns.violinplot(data=sample, x="nfa_type", y="plot_value", order=order, hue="nfa_type", palette=CLASS_COLORS,
                   legend=False, inner=None, cut=0, density_norm="width", linewidth=1, saturation=1, ax=ax)
    medians = full_larger.groupby("nfa_type")[value_column].median()

    for position, name in enumerate(order):
        ax.hlines(transform(medians[name]), position - 0.3, position + 0.3, color="black", linewidth=2, zorder=5)

    return medians


def plot_larger_violin(df, path):
    # How much larger is the DFA, for the NFAs where it is larger? Minimized and raw subset-construction DFA
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    ticks = [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
    # A few extreme NFAs would stretch the axis, so it stops at the 99.99th percentile and each class's maximum is written on top
    top = np.log2(df.groupby("nfa_type")["subset_ratio"].quantile(0.9999).max()) + 0.6

    for ax, ratio_column, larger_column, title in [
        (axes[0], "state_ratio", "larger_in_states", "Minimized DFA"),
        (axes[1], "subset_ratio", "subset_larger", "Raw subset-construction DFA"),
    ]:
        larger = df[df[larger_column]]
        medians = draw_violins(ax, larger, ratio_column, larger, np.log2)
        totals = df.groupby("nfa_type").size()
        counts = larger.groupby("nfa_type").size()
        labels = []

        maximums = df.groupby("nfa_type")[ratio_column].max()

        for position, name in enumerate(RANDOM_CLASSES):
            ax.text(position + 0.34, np.log2(medians[name]), f"{medians[name]:.1f}×", va="center", fontsize=10, fontweight="bold")
            ax.text(position, top - 0.12, f"max {maximums[name]:.0f}×", ha="center", va="top", fontsize=8.5, style="italic", color="#555555")
            labels.append(f"{name}\n{100 * counts[name] / totals[name]:.0f}% larger\n{counts[name]:,} of\n{totals[name]:,}")

        ax.set_xticks(range(len(RANDOM_CLASSES)), labels, fontsize=9)
        ax.set_yticks(np.log2(ticks), [f"{tick:g}×" for tick in ticks])
        ax.set_ylim(-0.05, top)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("")
        ax.set_title(title, fontsize=12)
        style_axis(ax)

    axes[0].set_ylabel("DFA states ÷ NFA states (log scale)")
    fig.suptitle(f"How much larger? DFA size relative to its NFA, for the NFAs whose DFA is larger "
                 f"({len(df):,} random NFAs; black bar = median)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_ceiling_approach(df, path):
    # Left: a sample of NFAs against the 2^n ceiling, with the highest point reached at each NFA size.
    # Right: how far toward the ceiling (in the exponent) the larger DFAs get, per class.
    sample = df.sample(min(SCATTER_SAMPLE, len(df)), random_state=0)
    jitter = np.random.default_rng(0).uniform(-0.28, 0.28, len(sample))
    log_states = np.log2(sample["dfa_states"])
    larger = sample["larger_in_states"].to_numpy()
    n_max = int(df["original_states"].max())
    # With one NFA size there is nothing to spread along n, so the x-axis becomes the number of transitions
    single_size = df["original_states"].nunique() == 1
    x_column = "total_transitions" if single_size else "original_states"

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15.5, 6.2), gridspec_kw={"width_ratios": [1.2, 1]})

    ax_a.scatter((sample[x_column] + jitter)[~larger], log_states[~larger], s=4, color="#9E9E9E", alpha=0.25,
                 label="DFA same size or smaller", linewidths=0)
    ax_a.scatter((sample[x_column] + jitter)[larger], log_states[larger], s=5, color="#D9534F", alpha=0.3,
                 label="DFA larger than NFA", linewidths=0)

    grouped = df.groupby(x_column)["dfa_states"]
    counts_per_x = grouped.size()
    keep = counts_per_x[counts_per_x >= 200].index
    ax_a.plot(keep, np.log2(grouped.max()[keep]), "^", color="black", markersize=9, label=f"highest of all {len(df):,}", zorder=6)
    ax_a.plot(keep, np.log2(grouped.quantile(0.999)[keep]), "D", color="#1F3A6E", markersize=6, label="99.9th percentile", zorder=6)

    x_range = np.linspace(df[x_column].min() - 0.8, df[x_column].max() + 0.8, 100)
    if single_size:
        n = n_max
        ax_a.axhline(n, color="black", linewidth=1.6, linestyle="--", label=f"ceiling: 2$^{{{n}}}$ = {2 ** n:,} states")
        ax_a.axhline(n / 2, color="#444444", linewidth=1.2, linestyle=":", label=f"halfway (exponent): 2$^{{{n // 2}}}$ = {2 ** (n // 2)} states")
        ax_a.axhline(np.log2(n), color="black", linewidth=1.2, label=f"NFA size: {n} states")
        ax_a.set_xlabel("NFA transitions (symbol and ε edges)")
    else:
        ax_a.plot(x_range, x_range, color="black", linewidth=1.6, linestyle="--", label="ceiling: 2$^n$ states")
        ax_a.plot(x_range, x_range / 2, color="#444444", linewidth=1.2, linestyle=":", label="halfway (exponent): 2$^{n/2}$")
        ax_a.plot(x_range, np.log2(x_range), color="black", linewidth=1.2, label="NFA size: n states")
        ax_a.set_xlabel("NFA states (n)")

    ax_a.set_xlim(x_range[0], x_range[-1])
    ax_a.set_ylim(0, n_max + 1.2)
    ticks = list(range(0, n_max + 1, 2))
    ax_a.set_yticks(ticks, [f"{2 ** tick:,}" for tick in ticks])
    ax_a.set_ylabel("Minimized DFA states (log scale)")
    ax_a.set_title(f"A. Minimized DFA size against the 2$^n$ ceiling ({len(sample):,}-NFA sample)", fontsize=12)
    ax_a.legend(loc="upper left", frameon=False, fontsize=8.5)
    style_axis(ax_a, "both")

    larger_all = df[df["larger_in_states"]]
    medians = draw_violins(ax_b, larger_all, "exponent_fraction", larger_all, lambda values: values)

    for position, name in enumerate(RANDOM_CLASSES):
        group = larger_all[larger_all["nfa_type"] == name]
        ax_b.text(position, 1.035, f"{100 * (group['exponent_fraction'] >= 0.5).mean():.0f}%", ha="center", fontsize=10, fontweight="bold")

    ax_b.text(-0.55, 1.07, "share at or above halfway:", ha="left", fontsize=9, style="italic")
    counts = larger_all.groupby("nfa_type").size()
    ax_b.set_xticks(range(len(RANDOM_CLASSES)), [f"{name}\n{counts[name]:,} NFAs" for name in RANDOM_CLASSES], fontsize=9)
    ax_b.axhline(1, color="black", linestyle="--", linewidth=1.6)
    ax_b.axhline(0.5, color="#444444", linestyle=":", linewidth=1.2)
    ax_b.text(5.5, 0.985, "ceiling (2$^n$)", va="top", ha="right", fontsize=9)
    ax_b.text(5.5, 0.51, "halfway (2$^{n/2}$)", va="bottom", ha="right", fontsize=9)
    ax_b.set_ylim(0, 1.14)
    ax_b.set_xlabel("")
    ax_b.set_ylabel("log$_2$(DFA states) ÷ n   (1.0 = the 2$^n$ ceiling)")
    ax_b.set_title("B. How close do the 'larger' DFAs get?", fontsize=12)
    style_axis(ax_b)

    size_text = f"{n_max}-state" if single_size else f"{df['original_states'].min()}–{n_max} states"
    fig.suptitle(f"Do larger DFAs approach the 2$^n$ ceiling?  ({len(df):,} random NFAs, "
                 f"{size_text}; blow-up family excluded)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Main

def main():
    parser = argparse.ArgumentParser(description="Million-NFA blow-up experiment")
    parser.add_argument("--count", type=int, default=1_000_000, help="distinct random NFAs, split evenly over the six classes")
    parser.add_argument("--seed", type=int, default=2, help="differs from the first two runs (0 and 1)")
    parser.add_argument("--min-states", type=int, default=5)
    parser.add_argument("--max-states", type=int, default=12)
    parser.add_argument("--workers", type=int, default=None, help="default: one per CPU core")
    parser.add_argument("--out-dir", type=Path, default=EXPERIMENT_1M_DIR)
    args = parser.parse_args()

    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers or __import__("os").cpu_count()

    print(f"Generating {args.count:,} distinct NFAs on {workers} workers ...", flush=True)
    start = time.time()
    df, num_failed, num_duplicates = generate_distinct_nfas(args.count, args.seed, args.min_states, args.max_states, workers)
    print(f"Done in {time.time() - start:.0f}s: {len(df):,} distinct NFAs "
          f"({num_failed:,} draws with no reachable NFA, {num_duplicates:,} duplicates dropped)", flush=True)

    assert df["nfa_key"].is_unique and len(df) == args.count
    df = add_size_columns(df)

    database = args.out_dir / DATABASE_NAME
    database.unlink(missing_ok=True)
    con = duckdb.connect(str(database))
    con.register("results_df", df)
    con.execute("CREATE TABLE nfa_results AS SELECT * FROM results_df")
    con.close()

    single_size = df["original_states"].nunique() == 1
    size_column = "total_transitions" if single_size else "original_states"
    by_class, by_size = summarize(df, size_column)
    by_class.to_csv(args.out_dir / "summary_by_class.csv")
    by_size.to_csv(args.out_dir / ("summary_by_transitions.csv" if single_size else "summary_by_nfa_size.csv"))
    print("\nBy class:")
    print(by_class.to_string())
    print(f"\nBy {size_column} (the ceiling column is 2^n):")
    print(by_size.to_string())

    use_georgia()
    plot_larger_violin(df, figures_dir / "larger_dfa_violin.png")
    plot_ceiling_approach(df, figures_dir / "ceiling_approach.png")

    closest = df.nlargest(10, "exponent_fraction")[["nfa_key", "nfa_type", "original_states", "dfa_states", "share_of_ceiling", "exponent_fraction"]]
    print("\nClosest to the ceiling (by exponent fraction):")
    print(closest.round(3).to_string(index=False))
    print(f"\nWrote {database} and figures to {figures_dir}")


if __name__ == "__main__":
    main()
