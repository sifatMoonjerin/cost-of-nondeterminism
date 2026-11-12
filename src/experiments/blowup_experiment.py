import argparse
import json
import random
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from core.automata import (
    EPSILON,
    generate_nfa,
    get_nfa_id,
    minimize_dfa,
    nfa_from_dict,
    nfa_to_dfa,
    nfa_to_dict,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
GENERATED_DIR = ROOT_DIR / "data" / "generated"
EXPERIMENT_DIR = ROOT_DIR / "data" / "experiment"
DATABASE_NAME = "blowup.duckdb"

# Per-NFA structural columns compared between NFAs whose DFA is larger and those whose isn't
FEATURE_COLUMNS = [
    "original_states",
    "num_accept_states",
    "num_symbol_transitions",
    "num_epsilon_transitions",
    "avg_out_degree",
    "max_out_degree",
    "nondeterministic_pairs",
    "avg_branching",
    "symbol_coverage",
]

OUT_DEGREE_BINS = [0, 2, 3, 4, 5, float("inf")]
OUT_DEGREE_LABELS = ["1-2", "2-3", "3-4", "4-5", "5+"]

LARGER_COLOR = "#D9534F"
SMALLER_COLOR = "#9E9E9E"
EQUAL_COLOR = "#F0AD4E"
TRANSITION_COLOR = "#4285F4"

# Row order for the by-type charts: sparsest to densest
TYPE_ORDER = ["blow-up", "sparse", "sparse-epsilon", "mixed", "mixed-epsilon", "dense", "dense-epsilon"]


# ---------------------------------------------------------------- Generation

def classify_sweep_nfa(params):
    # Same cut-offs as the notebook: sparse is 1-3 transitions per state, dense is 4+.
    # Ranges that straddle both are "mixed". The -epsilon suffix marks NFAs with ε edges.
    if params["max_transitions_per_state"] <= 3:
        density = "sparse"
    elif params["min_transitions_per_state"] >= 4:
        density = "dense"
    else:
        density = "mixed"

    return density + "-epsilon" if params["allow_epsilon"] else density


def generate_experiment_nfas(count, seed=0, min_states=5, max_states=12):
    # Each NFA draws its own generator parameters, so the dataset covers many shapes
    param_rng = random.Random(seed)
    entries = []
    seen_ids = set()
    nfa_seed = seed * 1_000_000
    num_failed = 0

    while len(entries) < count:
        params = {
            "seed": nfa_seed,
            "num_states": param_rng.randint(min_states, max_states),
            "min_transitions_per_state": param_rng.randint(1, 4),
            "allow_epsilon": param_rng.random() < 0.5,
        }
        params["max_transitions_per_state"] = params["min_transitions_per_state"] + param_rng.randint(0, 4)
        nfa_seed += 1

        try:
            nfa = generate_nfa(**params)
        except RuntimeError:
            num_failed += 1
            continue

        nfa_id = get_nfa_id(nfa)

        if nfa_id not in seen_ids:
            seen_ids.add(nfa_id)
            entries.append((nfa, dict(params, source="sweep", nfa_type=classify_sweep_nfa(params))))

    return entries, num_failed


def load_json_nfas(directory=GENERATED_DIR):
    entries = []

    # epsilon_nfas.json is the old single epsilon class, replaced by the sparse / dense split
    for name, nfa_type in [
        ("sparse", "sparse"),
        ("dense", "dense"),
        ("sparse_epsilon", "sparse-epsilon"),
        ("dense_epsilon", "dense-epsilon"),
        ("blowup", "blow-up"),
    ]:
        path = directory / f"{name}_nfas.json"

        with open(path) as file:
            for data in json.load(file):
                entries.append((nfa_from_dict(data), {"source": path.stem, "nfa_type": nfa_type}))

    return entries


def save_nfas_json(entries, path):
    data = []

    for nfa, _ in entries:
        record = nfa_to_dict(nfa)
        record["id"] = get_nfa_id(nfa)
        data.append(record)

    with open(path, "w") as file:
        json.dump(data, file, indent=4)


# ---------------------------------------------------------------- DataFrame

def get_structural_features(nfa):
    num_states = len(nfa.states)
    out_degree = {state: 0 for state in nfa.states}
    symbol_transitions = 0
    epsilon_transitions = 0
    defined_pairs = 0
    nondeterministic_pairs = 0

    for state, symbol_dict in nfa.transitions.items():
        for symbol, destinations in symbol_dict.items():
            out_degree[state] += len(destinations)

            if symbol == EPSILON:
                epsilon_transitions += len(destinations)
            else:
                symbol_transitions += len(destinations)
                defined_pairs += 1
                if len(destinations) > 1:
                    nondeterministic_pairs += 1

    return {
        "num_accept_states": len(nfa.accept_states),
        "num_symbol_transitions": symbol_transitions,
        "num_epsilon_transitions": epsilon_transitions,
        "avg_out_degree": (symbol_transitions + epsilon_transitions) / num_states,
        "max_out_degree": max(out_degree.values()),
        # (state, symbol) pairs with more than one destination: the source of non-determinism
        "nondeterministic_pairs": nondeterministic_pairs,
        "avg_branching": symbol_transitions / defined_pairs if defined_pairs else 0,
        # fraction of the n * |alphabet| (state, symbol) slots that have any transition
        "symbol_coverage": defined_pairs / (num_states * len(nfa.alphabet)),
    }


def build_record(nfa, params):
    features = get_structural_features(nfa)
    nfa_states = len(nfa.states)
    nfa_transitions = features["num_symbol_transitions"] + features["num_epsilon_transitions"]

    subset_dfa = nfa_to_dfa(nfa)
    min_dfa = minimize_dfa(subset_dfa)
    dfa_states = len(min_dfa.states)
    # The minimized DFA is complete (dead state included), so this is always states * |alphabet|
    dfa_transitions = sum(len(symbol_dict) for symbol_dict in min_dfa.transitions.values())

    state_info = {
        "states": sorted(nfa.states),
        "start_state": nfa.start_state,
        "accept_states": sorted(nfa.accept_states),
        **features,
    }

    return {
        "nfa_id": get_nfa_id(nfa),
        "source": params["source"],
        "nfa_type": params["nfa_type"],
        "seed": params.get("seed"),
        "min_transitions_per_state": params.get("min_transitions_per_state"),
        "max_transitions_per_state": params.get("max_transitions_per_state"),
        "state_info": state_info,
        "original_states": nfa_states,
        **features,
        "has_epsilon": features["num_epsilon_transitions"] > 0,
        "subset_states": len(subset_dfa.states),
        "dfa_states": dfa_states,
        "nfa_transitions": nfa_transitions,
        "dfa_transitions": dfa_transitions,
        # Relative size of the minimized DFA: 2.0 means twice as large, 0.5 means half the size
        "state_ratio": dfa_states / nfa_states,
        "transition_ratio": dfa_transitions / nfa_transitions,
        "larger_in_states": dfa_states > nfa_states,
        "larger_in_transitions": dfa_transitions > nfa_transitions,
    }


def build_dataframe(entries):
    df = pd.DataFrame([build_record(nfa, params) for nfa, params in entries])
    df["out_degree_bin"] = pd.cut(df["avg_out_degree"], bins=OUT_DEGREE_BINS, labels=OUT_DEGREE_LABELS).astype(str)
    return df


# ---------------------------------------------------------------- DuckDB

LARGER_RATE_COLUMNS = """
    count(*) AS nfas,
    round(100 * avg(larger_in_states::INT), 1) AS pct_larger_states,
    round(100 * avg(larger_in_transitions::INT), 1) AS pct_larger_transitions,
    round(median(state_ratio), 2) AS median_state_ratio
"""


def save_to_duckdb(df, path):
    path.unlink(missing_ok=True)
    con = duckdb.connect(str(path))

    # state_info is a dict, so it is stored as a JSON column
    results = df.copy()
    results["state_info"] = results["state_info"].apply(json.dumps)
    con.register("results_df", results)
    con.execute("CREATE TABLE nfa_results AS SELECT * REPLACE (state_info::JSON AS state_info) FROM results_df")

    # The queries are views, so they stay current if rows are added to nfa_results.
    # The hand-built blow-up family is left out of everything except the by-type view.
    con.execute("CREATE VIEW random_nfas AS SELECT * FROM nfa_results WHERE nfa_type <> 'blow-up'")

    con.execute("""
        CREATE VIEW largest_relative_size AS
        SELECT nfa_id, nfa_type, original_states, dfa_states, round(state_ratio, 2) AS state_ratio,
               nfa_transitions, dfa_transitions, round(transition_ratio, 2) AS transition_ratio,
               round(avg_out_degree, 2) AS avg_out_degree, has_epsilon
        FROM random_nfas ORDER BY state_ratio DESC LIMIT 20
    """)

    for view, source, column, order_by in [
        ("larger_rate_by_type", "nfa_results", "nfa_type", "pct_larger_states DESC"),
        ("larger_rate_by_out_degree", "random_nfas", "out_degree_bin", "min(avg_out_degree)"),
        ("larger_rate_by_states", "random_nfas", "original_states", "original_states"),
        ("larger_rate_by_epsilon", "random_nfas", "has_epsilon", "has_epsilon"),
    ]:
        con.execute(f"""
            CREATE VIEW {view} AS
            SELECT {column}, {LARGER_RATE_COLUMNS}
            FROM {source} GROUP BY {column} ORDER BY {order_by}
        """)

    # Smaller / same size / larger than the NFA, for the minimized DFA and for the raw subset-construction DFA
    for view, size_column in [
        ("size_outcome_by_type", "dfa_states"),
        ("subset_size_outcome_by_type", "subset_states"),
    ]:
        con.execute(f"""
            CREATE VIEW {view} AS
            SELECT nfa_type, count(*) AS nfas,
                   round(100 * avg(({size_column} < original_states)::INT), 1) AS pct_smaller,
                   round(100 * avg(({size_column} = original_states)::INT), 1) AS pct_equal,
                   round(100 * avg(({size_column} > original_states)::INT), 1) AS pct_larger
            FROM nfa_results GROUP BY nfa_type ORDER BY pct_larger DESC
        """)

    feature_averages = ",\n".join(f"round(avg({column}), 2) AS {column}" for column in FEATURE_COLUMNS)
    con.execute(f"""
        CREATE VIEW features_larger_vs_not AS
        SELECT larger_in_states, count(*) AS nfas, {feature_averages}
        FROM random_nfas GROUP BY larger_in_states ORDER BY larger_in_states
    """)

    return con


# ---------------------------------------------------------------- Plots

def plot_relative_size(con, path):
    df = con.sql("SELECT state_ratio, larger_in_states FROM random_nfas").df()
    pct_larger = 100 * df["larger_in_states"].mean()

    # Bins are evenly spaced on a log scale, so 1/2x and 2x are the same distance from 1x.
    # Edges sit just above each power, so a ratio of exactly 1 lands left of the 1x line.
    bins = [2 ** (k / 2) * 1.0001 for k in range(-9, 9)]
    ticks = [1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16]
    tick_labels = ["1/8×", "1/4×", "1/2×", "1×", "2×", "4×", "8×", "16×"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        [df.loc[~df["larger_in_states"], "state_ratio"], df.loc[df["larger_in_states"], "state_ratio"]],
        bins=bins, stacked=True, color=[SMALLER_COLOR, LARGER_COLOR],
        label=["DFA same size or smaller", "DFA larger"],
    )
    ax.axvline(1, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks(ticks, tick_labels)
    ax.minorticks_off()
    ax.set_xlabel("Minimized DFA states ÷ NFA states")
    ax.set_ylabel("Number of NFAs")
    ax.set_title(f"Minimized DFA size relative to the NFA ({pct_larger:.0f}% of NFAs end up larger)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_larger_rates(con, path):
    panels = [
        ("By NFA type", "larger_rate_by_type", "nfa_type"),
        ("By average out-degree", "larger_rate_by_out_degree", "out_degree_bin"),
        ("By NFA size (states)", "larger_rate_by_states", "original_states"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, (title, view, column) in zip(axes, panels):
        rates = con.sql(f"SELECT * FROM {view}").df()
        positions = range(len(rates))
        width = 0.4

        ax.bar([p - width / 2 for p in positions], rates["pct_larger_states"], width, color=LARGER_COLOR, label="Larger in states")
        ax.bar([p + width / 2 for p in positions], rates["pct_larger_transitions"], width, color=TRANSITION_COLOR, label="Larger in transitions")
        ax.set_xticks(list(positions), rates[column].astype(str), rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 100)

    axes[0].set_ylabel("% of NFAs whose minimized DFA is larger")
    axes[0].legend(loc="upper right")
    fig.suptitle("Which NFAs end up with a larger minimized DFA?")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_size_outcome_by_type(con, view, dfa_name, title, path):
    outcomes = con.sql(f"SELECT * FROM {view}").df()
    # Fixed type order, so the minimized and subset charts line up row for row
    outcomes = outcomes.set_index("nfa_type").reindex([t for t in TYPE_ORDER if t in set(outcomes["nfa_type"])]).reset_index()
    segments = [
        ("pct_smaller", "DFA smaller", SMALLER_COLOR),
        ("pct_equal", "DFA same size", EQUAL_COLOR),
        ("pct_larger", "DFA larger", LARGER_COLOR),
    ]
    labels = [f"{row.nfa_type} ({row.nfas})" for row in outcomes.itertuples()]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    left = [0] * len(outcomes)

    # One horizontal bar per type, split into smaller / same size / larger (each bar sums to 100%)
    for column, label, color in segments:
        bars = ax.barh(labels, outcomes[column], left=left, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.0f}%" if v >= 3 else "" for v in outcomes[column]], label_type="center")
        left = [a + b for a, b in zip(left, outcomes[column])]

    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"% of NFAs ({dfa_name} states vs. NFA states)")
    ax.set_title(title)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- Main

def main():
    parser = argparse.ArgumentParser(description="NFA to DFA blow-up experiment")
    parser.add_argument("--count", type=int, default=1000, help="number of random NFAs to generate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-states", type=int, default=5)
    parser.add_argument("--max-states", type=int, default=12)
    parser.add_argument("--include-json", action="store_true", help="also load data/generated/*_nfas.json")
    parser.add_argument("--out-dir", type=Path, default=EXPERIMENT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    entries, num_failed = generate_experiment_nfas(args.count, args.seed, args.min_states, args.max_states)
    print(f"Generated {len(entries)} unique NFAs ({num_failed} parameter draws had no fully reachable NFA)")

    if args.include_json:
        existing = load_json_nfas()
        known_ids = {get_nfa_id(nfa) for nfa, _ in entries}
        existing = [(nfa, params) for nfa, params in existing if get_nfa_id(nfa) not in known_ids]
        entries += existing
        print(f"Loaded {len(existing)} NFAs from {GENERATED_DIR}")

    save_nfas_json(entries, out_dir / "nfas.json")

    df = build_dataframe(entries)
    con = save_to_duckdb(df, out_dir / DATABASE_NAME)

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

    con.close()
    print()
    print(f"Wrote {out_dir / DATABASE_NAME} and figures to {figures_dir}")


if __name__ == "__main__":
    main()
