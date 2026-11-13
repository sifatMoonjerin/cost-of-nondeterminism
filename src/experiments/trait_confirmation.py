"""Which structural traits make an NFA's DFA unusually large, and do they really cause it?

Works on the 10k run (data/experiment_10k/) and writes CSVs to data/experiment_10k/trait_confirmation/:
  1. feature_z_differences.csv  traits of "extreme" NFAs vs. the rest, compared within each (class, NFA size) group
  2. trait_rates.csv            share of NFAs in the top 5% of their group, by trait value
  3. replication.csv            does the trait profile still pick out extreme NFAs on 12,000 fresh NFAs?
  4. interventions.csv          change one trait of an NFA and recompute its DFA (paired, same NFA before and after)

"Extreme" means the minimized DFA is in the largest 5% of its group, where a group is the NFAs with the same
class and the same number of states. That removes the obvious effects (dense NFAs are small, big NFAs are bigger).
"""
import argparse
import json
import random
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from core.automata import EPSILON, minimize_dfa, nfa_from_dict, nfa_to_dfa, nfa_to_dict
from experiments.blowup_experiment_10k import EXPERIMENT_10K_DIR, generate_balanced_nfas
from experiments.blowup_experiment import DATABASE_NAME

TOP_SHARE = 0.95

# Order matters: the intervention RNG is shared, so changing it changes which edges get thinned
INTERVENTION_SEED = 11
FRESH_SEED = 7
# The interventions are repeated on independent fresh samples (replication r uses FRESH_SEED + r and INTERVENTION_SEED + r)
REPLICATIONS = 10
INTERVENTION_COUNT = 4_000

# Traits compared between extreme and ordinary NFAs
TRAIT_COLUMNS = [
    "num_accept_states", "num_symbol_transitions", "num_epsilon_transitions", "avg_out_degree", "max_out_degree",
    "nondeterministic_pairs", "avg_branching", "symbol_coverage", "ecc_start", "mean_depth", "min_accept_depth",
    "max_accept_depth", "num_scc", "largest_scc_frac", "nondet_state_frac", "missing_slot_frac", "start_self_loops",
    "start_out_degree", "start_nondet", "accept_out_degree", "max_in_degree", "in_degree_1_frac", "zero_in_frac",
    "nondet_per_state",
]


# ---------------------------------------------------------------- Graph features

def get_graph_features(nfa):
    # Structure of the NFA as a graph (ε edges included), on top of the columns blowup_experiment already stores
    num_states = len(nfa.states)
    start = nfa.start_state
    out_edges = {state: set() for state in nfa.states}
    in_degree = {state: 0 for state in nfa.states}
    symbol_edges = {state: {} for state in nfa.states}

    for state, symbol_dict in nfa.transitions.items():
        for symbol, destinations in symbol_dict.items():
            for destination in destinations:
                out_edges[state].add(destination)
                in_degree[destination] += 1

            if symbol != EPSILON:
                symbol_edges[state][symbol] = set(destinations)

    # Shortest distance from the start state
    distance = {start: 0}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        for next_state in out_edges[current]:
            if next_state not in distance:
                distance[next_state] = distance[current] + 1
                queue.append(next_state)

    accept_distances = [distance[state] for state in nfa.accept_states if state in distance]
    component = get_strongly_connected_components(nfa.states, out_edges)
    component_sizes = np.bincount(list(component.values()))
    symbols = sorted(nfa.alphabet)

    nondeterministic_states = sum(
        1 for state in nfa.states if any(len(dests) > 1 for dests in symbol_edges[state].values())
    )
    missing_slots = sum(1 for state in nfa.states for symbol in symbols if symbol not in symbol_edges[state])

    return {
        "ecc_start": max(distance.values()),
        "mean_depth": float(np.mean(list(distance.values()))),
        "min_accept_depth": min(accept_distances) if accept_distances else -1,
        "max_accept_depth": max(accept_distances) if accept_distances else -1,
        "num_scc": len(component_sizes),
        "largest_scc_frac": component_sizes.max() / num_states,
        "nondet_state_frac": nondeterministic_states / num_states,
        "missing_slot_frac": missing_slots / (num_states * len(symbols)),
        "start_self_loops": sum(1 for symbol in symbols if start in symbol_edges[start].get(symbol, set())),
        "start_out_degree": len(out_edges[start]),
        "start_nondet": int(any(len(dests) > 1 for dests in symbol_edges[start].values())),
        "accept_out_degree": float(np.mean([len(out_edges[state]) for state in nfa.accept_states])),
        "max_in_degree": max(in_degree.values()),
        "in_degree_1_frac": sum(1 for value in in_degree.values() if value == 1) / num_states,
        "zero_in_frac": sum(1 for value in in_degree.values() if value == 0) / num_states,
    }


def get_strongly_connected_components(states, out_edges):
    # Kosaraju: finish order on the graph, then sweep the reversed graph in reverse finish order
    finish_order = []
    seen = set()

    for root in sorted(states):
        if root in seen:
            continue

        seen.add(root)
        stack = [(root, iter(sorted(out_edges[root])))]

        while stack:
            node, neighbours = stack[-1]
            for neighbour in neighbours:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append((neighbour, iter(sorted(out_edges[neighbour]))))
                    break
            else:
                finish_order.append(node)
                stack.pop()

    reverse_edges = {state: set() for state in states}
    for source, destinations in out_edges.items():
        for destination in destinations:
            reverse_edges[destination].add(source)

    component = {}
    count = 0

    for root in reversed(finish_order):
        if root in component:
            continue

        component[root] = count
        stack = [root]

        while stack:
            node = stack.pop()
            for neighbour in reverse_edges[node]:
                if neighbour not in component:
                    component[neighbour] = count
                    stack.append(neighbour)

        count += 1

    return component


def get_distances(nfa):
    distance = {nfa.start_state: 0}
    queue = deque([nfa.start_state])

    while queue:
        current = queue.popleft()
        for symbol in list(nfa.alphabet) + [EPSILON]:
            for next_state in nfa.get_transitions(current, symbol):
                if next_state not in distance:
                    distance[next_state] = distance[current] + 1
                    queue.append(next_state)

    return distance


def count_nondeterministic_pairs(nfa):
    return sum(
        1
        for symbol_dict in nfa.transitions.values()
        for symbol, destinations in symbol_dict.items()
        if symbol != EPSILON and len(destinations) > 1
    )


def minimized_size(nfa):
    return len(minimize_dfa(nfa_to_dfa(nfa)).states)


def clone(nfa):
    return nfa_from_dict(nfa_to_dict(nfa))


# ---------------------------------------------------------------- Load the 10k run

def load_features(experiment_dir):
    con = duckdb.connect(str(experiment_dir / DATABASE_NAME), read_only=True)
    df = con.sql("SELECT * EXCLUDE (state_info) FROM random_nfas").df()
    con.close()

    with open(experiment_dir / "nfas.json") as file:
        nfas = {record["id"]: record for record in json.load(file)}

    graph = pd.DataFrame([get_graph_features(nfa_from_dict(nfas[nfa_id])) for nfa_id in df["nfa_id"]])
    df = pd.concat([df, graph], axis=1)
    df["nondet_per_state"] = df["nondeterministic_pairs"] / df["original_states"]
    df["group"] = df["nfa_type"] + "|" + df["original_states"].astype(str)
    df["log_dfa"] = np.log2(df["dfa_states"])
    # Top 5% of the group by minimized DFA size
    df["extreme"] = df.groupby("group")["log_dfa"].rank(pct=True) > TOP_SHARE
    return df


# ---------------------------------------------------------------- 1. Which traits differ?

def feature_z_differences(df):
    # Each trait is z-scored inside its group first, so class and NFA size cannot explain a difference
    rows = []
    groups = df.groupby("group")

    for column in TRAIT_COLUMNS:
        std = groups[column].transform("std").replace(0, np.nan)
        z = ((df[column] - groups[column].transform("mean")) / std).fillna(0)
        rows.append({
            "trait": column,
            "z_difference": z[df["extreme"]].mean() - z[~df["extreme"]].mean(),
            "extreme_mean": df.loc[df["extreme"], column].mean(),
            "rest_mean": df.loc[~df["extreme"], column].mean(),
        })

    return pd.DataFrame(rows).sort_values("z_difference", key=abs, ascending=False).round(3)


# ---------------------------------------------------------------- 2. Extreme rate by trait value

def trait_rates(df):
    df = df.assign(
        single_component=df["largest_scc_frac"] == 1,
        epsilon_edges=df["num_epsilon_transitions"].clip(upper=3),
        accept_distance=df["min_accept_depth"].clip(upper=5),
    )
    specs = [
        ("avg_branching", pd.cut(df["avg_branching"], [0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 10])),
        ("nondeterministic pairs per state", pd.cut(df["nondet_per_state"], [-0.01, 0.25, 0.5, 0.75, 1.0, 1.5, 5])),
        ("accepting states", df["num_accept_states"]),
        ("accept distance (steps)", df["accept_distance"]),
        ("epsilon edges", df["epsilon_edges"]),
        ("single strongly connected component", df["single_component"]),
    ]
    rows = []

    for name, values in specs:
        for value, group in df.groupby(values, observed=True):
            rows.append({"trait": name, "value": str(value), "nfas": len(group), "pct_extreme": 100 * group["extreme"].mean()})

    return pd.DataFrame(rows).round(1), 100 * df["extreme"].mean()


# ---------------------------------------------------------------- Shared: thresholds and the trait profile

def get_thresholds(df):
    # 95th percentile of minimized DFA size per (class, NFA size); "extreme" on fresh NFAs means larger than this
    return df.groupby(["nfa_type", "original_states"])["dfa_states"].quantile(TOP_SHARE).to_dict()


def matches_profile(nfa):
    # One accepting state, at least two steps from the start, 0.25 to 1.0 non-deterministic pairs per state
    features = get_graph_features(nfa)
    pairs_per_state = count_nondeterministic_pairs(nfa) / len(nfa.states)

    return len(nfa.accept_states) == 1 and features["min_accept_depth"] >= 2 and 0.25 <= pairs_per_state <= 1.0


# ---------------------------------------------------------------- 3. Replication on fresh NFAs

def replicate(thresholds, entries, rng_seed=0):
    rows = []

    for nfa, params in entries:
        size = minimized_size(nfa)
        rows.append({
            "nfa_type": params["nfa_type"],
            "states": len(nfa.states),
            "dfa_states": size,
            "profile": matches_profile(nfa),
            "extreme": size > thresholds[(params["nfa_type"], len(nfa.states))],
        })

    df = pd.DataFrame(rows)
    with_profile = df.loc[df["profile"], "extreme"].to_numpy()
    without_profile = df.loc[~df["profile"], "extreme"].to_numpy()

    rng = np.random.default_rng(rng_seed)
    differences = [
        rng.choice(with_profile, len(with_profile)).mean() - rng.choice(without_profile, len(without_profile)).mean()
        for _ in range(2000)
    ]

    summary = pd.DataFrame([{
        "fresh_nfas": len(df),
        "overall_pct_extreme": 100 * df["extreme"].mean(),
        "profile_nfas": int(df["profile"].sum()),
        "profile_pct_extreme": 100 * with_profile.mean(),
        "rest_pct_extreme": 100 * without_profile.mean(),
        "difference_points": 100 * (with_profile.mean() - without_profile.mean()),
        "ci_low_points": 100 * np.percentile(differences, 2.5),
        "ci_high_points": 100 * np.percentile(differences, 97.5),
    }])

    by_class = df.groupby("nfa_type").apply(
        lambda group: pd.Series({
            "pct_extreme_all": 100 * group["extreme"].mean(),
            "pct_extreme_profile": 100 * group.loc[group["profile"], "extreme"].mean(),
            "profile_nfas": int(group["profile"].sum()),
        }),
        include_groups=False,
    ).reset_index()

    return summary.round(2), by_class.round(1)


# ---------------------------------------------------------------- 4. Interventions

def keep_deepest_accept(nfa):
    if len(nfa.accept_states) < 2:
        return None

    distance = get_distances(nfa)
    reachable = [state for state in nfa.accept_states if state in distance]

    if not reachable:
        return None

    changed = clone(nfa)
    changed.accept_states = {max(reachable, key=lambda state: (distance[state], state))}
    return changed


def move_accept_to_deepest_state(nfa):
    distance = get_distances(nfa)
    changed = clone(nfa)
    changed.accept_states = {max(distance, key=lambda state: (distance[state], state))}
    return changed


def drop_epsilon(nfa):
    if not any(EPSILON in symbol_dict for symbol_dict in nfa.transitions.values()):
        return None

    changed = clone(nfa)
    for symbol_dict in changed.transitions.values():
        symbol_dict.pop(EPSILON, None)

    return changed


def cap_branching(nfa, rng):
    changed = clone(nfa)
    modified = False

    for symbol_dict in changed.transitions.values():
        for symbol, destinations in symbol_dict.items():
            if symbol != EPSILON and len(destinations) > 2:
                symbol_dict[symbol] = set(rng.sample(sorted(destinations), 2))
                modified = True

    return changed if modified else None


def add_accept_states(nfa, rng):
    if len(nfa.accept_states) >= 3:
        return None

    changed = clone(nfa)
    others = sorted(nfa.states - nfa.accept_states)
    changed.accept_states = set(nfa.accept_states) | set(rng.sample(others, 3 - len(nfa.accept_states)))
    return changed


def add_random_edges(nfa, rng):
    changed = clone(nfa)

    for state in sorted(changed.states):
        symbol = rng.choice(sorted(changed.alphabet))
        changed.add_transition(state, symbol, rng.choice(sorted(changed.states)))

    return changed


def no_epsilon_and_deep_accept(nfa):
    return move_accept_to_deepest_state(drop_epsilon(nfa) or clone(nfa))


def run_interventions(thresholds, entries, seed=INTERVENTION_SEED):
    rng = random.Random(seed)
    changes = [
        ("single accepting state (keep deepest)", keep_deepest_accept, False),
        ("move accepting state to the deepest state", move_accept_to_deepest_state, False),
        ("delete all ε edges", drop_epsilon, True),
        ("thin branching (cap at 2 destinations)", lambda nfa: cap_branching(nfa, rng), False),
        ("[control] add accepting states up to 3", lambda nfa: add_accept_states(nfa, rng), False),
        ("[control] add one extra random edge per state", lambda nfa: add_random_edges(nfa, rng), False),
        ("combined: no ε + one deep accepting state", no_epsilon_and_deep_accept, True),
    ]
    rows = []

    for name, change, removes_epsilon in changes:
        log_ratios = []
        extreme_before = []
        extreme_after = []

        for nfa, params in entries:
            changed = change(nfa)

            if changed is None:
                continue

            before = minimized_size(nfa)
            after = minimized_size(changed)
            num_states = len(nfa.states)
            nfa_type = params["nfa_type"]
            # Without ε edges the NFA behaves like a non-ε one, so judge it against that class
            after_type = nfa_type.replace("-epsilon", "") if removes_epsilon else nfa_type

            log_ratios.append(np.log2(after / before))
            extreme_before.append(before > thresholds[(nfa_type, num_states)])
            extreme_after.append(after > thresholds[(after_type, num_states)])

        log_ratios = np.array(log_ratios)
        rows.append({
            "change": name,
            "nfas": len(log_ratios),
            "mean_size_ratio": np.mean(2 ** log_ratios),
            "median_size_ratio": 2 ** np.median(log_ratios),
            "pct_grew": 100 * np.mean(log_ratios > 0),
            "pct_shrank": 100 * np.mean(log_ratios < 0),
            "pct_extreme_before": 100 * np.mean(extreme_before),
            "pct_extreme_after": 100 * np.mean(extreme_after),
        })

    return pd.DataFrame(rows).round(2)


def run_one_replication(task):
    # One independent replication: a fresh balanced sample and its own intervention seed
    thresholds, replication = task
    entries, _, _ = generate_balanced_nfas(INTERVENTION_COUNT, seed=FRESH_SEED + replication)
    rows = run_interventions(thresholds, entries, seed=INTERVENTION_SEED + replication)
    rows.insert(0, "replication", replication)
    return rows


INTERVAL_METRICS = ["mean_size_ratio", "median_size_ratio", "pct_grew", "pct_shrank", "pct_extreme_before", "pct_extreme_after"]


def summarize_replications(by_replication):
    # Mean over replications with a 95 % t interval; the interval reflects both which NFAs were drawn and which random
    # edges / accepting states each change picked
    rows = []
    for change, group in by_replication.groupby("change", sort=False):
        count = len(group)
        row = {"change": change, "replications": count, "nfas_per_replication": group["nfas"].mean()}
        for metric in INTERVAL_METRICS:
            mean = group[metric].mean()
            half_width = student_t.ppf(0.975, count - 1) * group[metric].std(ddof=1) / np.sqrt(count)
            row[metric] = mean
            row[metric + "_ci_low"] = mean - half_width
            row[metric + "_ci_high"] = mean + half_width
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def run_replicated_interventions(thresholds, replications=REPLICATIONS):
    with Pool() as pool:
        frames = pool.map(run_one_replication, [(thresholds, replication) for replication in range(replications)])
    by_replication = pd.concat(frames, ignore_index=True)
    return by_replication, summarize_replications(by_replication)


def plot_interventions(summary, path):
    # Share of NFAs whose DFA grew / shrank under each change, with 95 % intervals over the replications
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.blowup_experiment_10k import use_georgia

    use_georgia()
    labels = {
        "single accepting state (keep deepest)": "One accepting state (deepest)",
        "move accepting state to the deepest state": "Accepting state moved deep",
        "thin branching (cap at 2 destinations)": "Branching capped at 2",
        "delete all ε edges": "No ε edges*",
        "combined: no ε + one deep accepting state": "No ε + deep accepting state*",
        "[control] add accepting states up to 3": "Control: add accepting states",
        "[control] add one extra random edge per state": "Control: add random edges",
    }
    order = list(labels)
    rows = summary.set_index("change").loc[order]
    y = np.arange(len(order))[::-1]
    height = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.6))
    for offset, column, color, name in [(height / 2, "pct_grew", "#D9534F", "DFA grew"), (-height / 2, "pct_shrank", "#9E9E9E", "DFA shrank")]:
        values = rows[column].to_numpy()
        errors = np.array([values - rows[column + "_ci_low"].to_numpy(), rows[column + "_ci_high"].to_numpy() - values])
        ax.barh(y + offset, values, height, color=color, label=name, xerr=errors, error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "black"})
        for position, value, high in zip(y + offset, values, rows[column + "_ci_high"]):
            ax.text(high + 0.8, position, f"{value:.1f}", va="center", fontsize=9)

    ax.set_yticks(y, [labels[name] for name in order])
    ax.set_xlim(0, 80)
    ax.set_xticks(range(0, 81, 10), [f"{tick}%" for tick in range(0, 81, 10)])
    ax.grid(axis="x", color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Share of NFAs whose DFA grew or shrank (same NFAs, before vs after)", fontsize=12)
    ax.legend(frameon=False, loc="lower right")
    count = int(rows["replications"].iloc[0])
    fig.text(0.01, 0.01, f"Mean of {count} independent replications of ~{int(rows['nfas_per_replication'].min()):,}-{INTERVENTION_COUNT:,} NFAs each; "
             "bars show 95% intervals across replications. *Removing ε edges also moves an NFA into the larger non-ε class.", fontsize=7.5, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=250)
    plt.close(fig)


# ---------------------------------------------------------------- Main

def run_interventions_step(thresholds, out_dir, experiment_dir, replications):
    by_replication, summary = run_replicated_interventions(thresholds, replications)
    by_replication.to_csv(out_dir / "interventions_by_replication.csv", index=False)
    summary.to_csv(out_dir / "interventions.csv", index=False)
    figures_dir = experiment_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_interventions(summary, figures_dir / "interventions.png")
    print(f"\n4. Paired interventions, {replications} independent replications (mean and 95% interval):")
    print(summary[["change", "nfas_per_replication", "mean_size_ratio", "median_size_ratio", "pct_grew", "pct_grew_ci_low", "pct_grew_ci_high",
                   "pct_shrank", "pct_extreme_before", "pct_extreme_after"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Confirm which NFA traits lead to large DFAs")
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_10K_DIR)
    parser.add_argument("--fresh-count", type=int, default=12_000, help="fresh random NFAs for the replication")
    parser.add_argument("--replications", type=int, default=REPLICATIONS, help="independent repeats of the intervention test")
    parser.add_argument("--interventions-only", action="store_true", help="skip steps 1-3 and only (re)run the intervention test and its chart")
    args = parser.parse_args()

    out_dir = args.experiment_dir / "trait_confirmation"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_features(args.experiment_dir)
    thresholds = get_thresholds(df)

    if args.interventions_only:
        run_interventions_step(thresholds, out_dir, args.experiment_dir, args.replications)
        return

    print(f"{len(df)} random NFAs, {int(df['extreme'].sum())} extreme (top 5% of their class and size)")

    z_differences = feature_z_differences(df)
    z_differences.to_csv(out_dir / "feature_z_differences.csv", index=False)
    print("\n1. Traits of extreme NFAs vs. the rest (z-score difference within group):")
    print(z_differences.head(12).to_string(index=False))

    rates, baseline = trait_rates(df)
    rates.to_csv(out_dir / "trait_rates.csv", index=False)
    print(f"\n2. Share of NFAs in the top 5% of their group, by trait (overall {baseline:.1f}%):")
    print(rates.to_string(index=False))

    # Fresh NFAs: different seed from both earlier runs, balanced over the six classes
    entries, _, _ = generate_balanced_nfas(args.fresh_count, seed=FRESH_SEED)
    summary, by_class = replicate(thresholds, entries)
    summary.to_csv(out_dir / "replication.csv", index=False)
    by_class.to_csv(out_dir / "replication_by_class.csv", index=False)
    print(f"\n3. Replication on {args.fresh_count} fresh NFAs:")
    print(summary.T.to_string(header=False))
    print(by_class.to_string(index=False))

    run_interventions_step(thresholds, out_dir, args.experiment_dir, args.replications)

    print(f"\nWrote CSVs to {out_dir}")


if __name__ == "__main__":
    main()
