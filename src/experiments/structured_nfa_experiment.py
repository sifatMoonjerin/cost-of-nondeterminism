"""Do NFAs shaped like the ceiling hits reach the 2^n ceiling more often than the unrestricted generator?

The 10 distinct 6-state NFAs that reached the ceiling (experiment_ceiling_hunt_6_states) share a shape: exactly one
(state, symbol) slot has no transition, the other 11 have a destination, 1-3 accepting states, and 1-3 slots with two
destinations (73-91 % of the defined slots deterministic). This experiment generates only NFAs of that shape, with the
undefined slot picked uniformly at random (no preference for the start state), and records how often they reach 64 states.

Generator (`generate_structured_nfa`), for n states (default 6; `--states 5` or `--states 7` for other sizes) over {0, 1}, start q0, no ε edges:
  1. 1-3 random accepting states.
  2. One of the 2n slots (state, symbol), uniformly at random, is left empty.
  3. Every other slot gets one uniformly random destination (any of the n states, itself included).
  4. 1-3 of those defined slots, chosen at random, get a second, different destination.
  5. Redraw unless every state is reachable from q0.

1,000,000 distinct NFAs (an NFA and its 0/1-swapped copy count as one), all converted and minimized, on all CPU cores.

    python -m experiments.structured_nfa_experiment                 # the full run
    python -m experiments.structured_nfa_experiment --count 20000   # quick check
    python -m experiments.structured_nfa_experiment --states 7      # other NFA sizes

Output: data/experiment_structured_<n>_states/ (blowup.duckdb, summary.json, size_distribution.csv, hits_nfas.json, figures/).
"""
import argparse
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

from core.automata import NFA, get_reachable_states, initialize_nfa, minimize_dfa, nfa_to_dfa, nfa_to_dict
from experiments.blowup_experiment import DATABASE_NAME, ROOT_DIR, get_structural_features
from experiments.blowup_experiment_1m import plot_ceiling_hexbin
from experiments.blowup_experiment_10k import use_georgia, style_axis
from experiments.ceiling_hunt import get_canonical_key, key_to_int

DEFAULT_STATES = 6
DATA_DIR = ROOT_DIR / "data"
SHARD_SIZE = 20_000
SEEDS_PER_SHARD = 1_000_000_000


def generate_structured_nfa(seed, num_states=DEFAULT_STATES, max_attempts=1000):
    # Returns (nfa, undefined_state, undefined_symbol); the same seed always gives the same NFA
    rng = random.Random(seed)

    for _ in range(max_attempts):
        nfa = initialize_nfa(num_states, rng)
        slots = [(state, symbol) for state in range(num_states) for symbol in ("0", "1")]
        empty = rng.choice(slots)
        defined = [slot for slot in slots if slot != empty]

        for state, symbol in defined:
            nfa.add_transition(state, symbol, rng.randrange(num_states))

        for state, symbol in rng.sample(defined, rng.randint(1, 3)):
            first = next(iter(nfa.get_transitions(state, symbol)))
            nfa.add_transition(state, symbol, rng.choice([target for target in range(num_states) if target != first]))

        if len(get_reachable_states(nfa)) == num_states:
            return nfa, empty[0], empty[1]

    raise RuntimeError("no fully reachable NFA after the maximum number of attempts")


# ---------------------------------------------------------------- Worker

def run_shard(task):
    # One shard of `size` NFAs from a seed block of its own; returns a row per distinct NFA and the NFAs that reach the ceiling
    shard_id, size, num_states = task
    ceiling = 2 ** num_states
    nfa_seed = (shard_id + 1) * SEEDS_PER_SHARD
    seen = set()
    rows = []
    hits = []
    num_failed = 0

    while len(rows) < size:
        seed = nfa_seed
        nfa_seed += 1

        try:
            nfa, empty_state, empty_symbol = generate_structured_nfa(seed, num_states)
        except RuntimeError:
            num_failed += 1
            continue

        key = get_canonical_key(nfa)

        if key in seen:
            continue

        seen.add(key)
        subset_dfa = nfa_to_dfa(nfa)
        min_dfa = minimize_dfa(subset_dfa)
        row = {
            "nfa_key": key,
            "seed": seed,
            "original_states": num_states,
            "undefined_state": empty_state,
            "undefined_symbol": empty_symbol,
            **get_structural_features(nfa),
            "subset_states": len(subset_dfa.states),
            "dfa_states": len(min_dfa.states),
        }
        rows.append(row)

        if row["dfa_states"] == ceiling:
            hits.append({**row, "nfa": nfa_to_dict(nfa)})

    return rows, hits, num_failed


def generate_distinct(total, workers, num_states):
    # Shards run in parallel; duplicates across shards are dropped and topped up until `total` distinct NFAs exist
    rows, hits = [], []
    seen = set()
    duplicates = failed = 0
    next_shard = 0
    start = time.time()

    with Pool(workers) as pool:
        while len(rows) < total:
            missing = total - len(rows)
            tasks = []
            while missing > 0:
                tasks.append((next_shard, min(SHARD_SIZE, missing), num_states))
                next_shard += 1
                missing -= SHARD_SIZE

            for shard_rows, shard_hits, num_failed in pool.imap_unordered(run_shard, tasks):
                failed += num_failed
                kept = set()

                for row in shard_rows:
                    key = key_to_int(row["nfa_key"])
                    if key in seen or len(rows) >= total:
                        duplicates += 1
                    else:
                        seen.add(key)
                        kept.add(row["nfa_key"])
                        rows.append(row)

                hits += [hit for hit in shard_hits if hit["nfa_key"] in kept]
                print(f"  {len(rows):>9,} / {total:,} NFAs, {len(hits)} at the ceiling  ({time.time() - start:.0f}s)", flush=True)

    return pd.DataFrame(rows), hits, duplicates, failed


# ---------------------------------------------------------------- Summary and chart

def size_distribution(df, label, ceiling):
    # Share of NFAs at or above each DFA-size threshold
    sizes = df["dfa_states"]
    row = {"run": label, "nfas": len(df), "median_dfa_states": float(sizes.median())}
    for threshold in sorted({ceiling // 4, ceiling // 2, 3 * ceiling // 4, ceiling - 1, ceiling}):
        row[f"pct_at_least_{threshold}"] = 100 * (sizes >= threshold).mean()
    row["at_ceiling"] = int((sizes == ceiling).sum())
    return row


def with_total_transitions(df):
    # The hexbin plot puts the number of NFA transitions on the x axis
    return df.assign(total_transitions=df["num_symbol_transitions"] + df["num_epsilon_transitions"])


def plot_size_histogram(structured, baseline, path, num_states):
    # Share of NFAs at each minimized DFA size, structured generator against the unrestricted generator
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ceiling = 2 ** num_states
    sizes = np.arange(1, ceiling + 1)
    for df, label, color in [(baseline, "unrestricted generator, no ε edges", "#9E9E9E"), (structured, "one undefined slot, 1-3 non-det. pairs", "#D9534F")]:
        share = df["dfa_states"].value_counts(normalize=True).reindex(sizes, fill_value=0) * 100
        ax.plot(sizes, share.values, marker="o", markersize=3, linewidth=1.2, color=color, label=label)
    ax.set_yscale("log")
    ax.axvline(ceiling, color="black", linestyle="--", linewidth=1.4)
    ax.text(ceiling - 0.8, ax.get_ylim()[1] * 0.6, f"ceiling: {ceiling}", ha="right", fontsize=9)
    ax.set_xlabel("Minimized DFA states")
    ax.set_ylabel("Share of NFAs (%, log scale)")
    ax.set_title(f"Minimized DFA sizes of {num_states}-state NFAs")
    ax.legend(frameon=False)
    style_axis(ax, "both")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def one_in(share_percent):
    return "none" if share_percent == 0 else f"1 in {round(100 / share_percent):,}"


def plot_threshold_bars(structured, baseline, path, num_states):
    # How likely is an NFA to get at least this big a DFA? Grouped bars, labelled "1 in N" so the gap reads at a glance
    ceiling = 2 ** num_states
    thresholds = [(ceiling * k // 4, f"at least {ceiling * k // 4} states") for k in (1, 2, 3)] + [(ceiling, f"{ceiling} states (the ceiling)")]
    runs = [(baseline, "unrestricted generator, no ε edges", "#9E9E9E"), (structured, "one undefined slot, 1-3 non-det. pairs", "#D9534F")]
    x = np.arange(len(thresholds))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for offset, (df, label, color) in zip((-width / 2, width / 2), runs):
        shares = [100 * (df["dfa_states"] >= threshold).mean() for threshold, _ in thresholds]
        bars = ax.bar(x + offset, shares, width, color=color, label=label)
        for bar, share in zip(bars, shares):
            ax.text(bar.get_x() + bar.get_width() / 2, share * 1.15, one_in(share), ha="center", va="bottom", fontsize=10)

    ax.set_yscale("log")
    ax.set_ylim(0.00003, 300)
    ax.set_xticks(x, [name for _, name in thresholds])
    ax.set_ylabel("Share of NFAs (%)")
    ax.set_title(f"How often does a {num_states}-state NFA get a large minimized DFA?")
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def load_baseline(num_states):
    # Unrestricted generator with the same number of states: a 2,000,000-NFA run (blowup_experiment_1m --min-states n --max-states n),
    # of which the classes without ε edges (sparse, mixed, dense, about 1,000,000 NFAs) are the baseline, since the structured NFAs have none
    path = DATA_DIR / f"experiment_unrestricted_{num_states}_states_2m" / DATABASE_NAME
    label = f"unrestricted, no ε (experiment_unrestricted_{num_states}_states_2m)"
    con = duckdb.connect(str(path), read_only=True)
    df = con.sql("SELECT dfa_states, nfa_type FROM nfa_results WHERE nfa_type NOT LIKE '%epsilon'").df()
    con.close()
    return df, label


def write_summary(df, baseline, baseline_label, out_dir, num_states, generation):
    # Size table and summary.json; `generation` holds the facts of the run that made df (duplicates, seconds, ...)
    ceiling = 2 ** num_states
    table = pd.DataFrame([size_distribution(df, "structured", ceiling), size_distribution(baseline, baseline_label, ceiling)])
    table.round(4).to_csv(out_dir / "size_distribution.csv", index=False)

    at_ceiling = int((df["dfa_states"] == ceiling).sum())
    baseline_hits = int((baseline["dfa_states"] == ceiling).sum())
    summary = {
        "num_states": num_states, "ceiling": ceiling, "distinct_nfas": len(df), **generation,
        "epsilon_edges_in_structured_nfas": int(df["num_epsilon_transitions"].sum()),
        "at_ceiling": at_ceiling, "ceiling_hits_per_million": 1e6 * at_ceiling / len(df),
        "baseline": baseline_label, "baseline_nfas": len(baseline), "baseline_at_ceiling": baseline_hits,
        "baseline_hits_per_million": 1e6 * baseline_hits / len(baseline),
    }
    with open(out_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=2)
    return table, summary


def draw_figures(df, baseline, figures_dir, num_states):
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_size_histogram(df, baseline, figures_dir / "size_histogram.png", num_states)
    plot_threshold_bars(df, baseline, figures_dir / "size_threshold_bars.png", num_states)
    plot_ceiling_hexbin(with_total_transitions(df), figures_dir / "ceiling_hexbin.png")


def main():
    parser = argparse.ArgumentParser(description="Structured NFA experiment (one undefined slot, 1-3 non-deterministic pairs, no ε edges)")
    parser.add_argument("--states", type=int, default=DEFAULT_STATES, help="number of NFA states")
    parser.add_argument("--count", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: data/experiment_structured_<states>_states")
    parser.add_argument("--figures-only", action="store_true",
                        help="redraw the figures and recompute the comparison with the baseline from the existing database")
    args = parser.parse_args()

    num_states = args.states
    ceiling = 2 ** num_states
    out_dir = args.out_dir or DATA_DIR / f"experiment_structured_{num_states}_states"
    baseline, baseline_label = load_baseline(num_states)
    use_georgia()

    if args.figures_only:
        df = duckdb.connect(str(out_dir / DATABASE_NAME), read_only=True).sql("SELECT * FROM nfa_results").df()
        previous = json.loads((out_dir / "summary.json").read_text())
        generation = {key: previous[key] for key in ("duplicates_dropped", "draws_with_no_reachable_nfa", "seconds")}
        table, summary = write_summary(df, baseline, baseline_label, out_dir, num_states, generation)
        draw_figures(df, baseline, out_dir / "figures", num_states)
        print(json.dumps(summary, indent=2))
        print(table.round(3).T.to_string(header=False))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers or __import__("os").cpu_count()

    print(f"Generating {args.count:,} distinct structured {num_states}-state NFAs on {workers} workers ...", flush=True)
    start = time.time()
    df, hits, duplicates, failed = generate_distinct(args.count, workers, num_states)
    seconds = round(time.time() - start)
    assert df["nfa_key"].is_unique
    assert df["num_epsilon_transitions"].sum() == 0, "structured NFAs must have no ε edges"

    database = out_dir / DATABASE_NAME
    database.unlink(missing_ok=True)
    con = duckdb.connect(str(database))
    con.register("results_df", df)
    con.execute("CREATE TABLE nfa_results AS SELECT * FROM results_df")
    con.close()

    with open(out_dir / "hits_nfas.json", "w") as file:
        json.dump([{"id": hit["nfa_key"], **hit["nfa"]} for hit in hits], file, indent=4)

    by_position = df.assign(at_ceiling=df["dfa_states"] == ceiling, undefined_at_start=df["undefined_state"] == 0).groupby("undefined_at_start").agg(
        nfas=("dfa_states", "size"), at_ceiling=("at_ceiling", "sum"), mean_dfa_states=("dfa_states", "mean"))
    table, summary = write_summary(df, baseline, baseline_label, out_dir, num_states, {
        "duplicates_dropped": duplicates, "draws_with_no_reachable_nfa": failed, "seconds": seconds})
    draw_figures(df, baseline, out_dir / "figures", num_states)

    print(json.dumps(summary, indent=2))
    print(table.round(3).T.to_string(header=False))
    print("\nBy position of the undefined slot:")
    print(by_position.round(3).to_string())
    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
