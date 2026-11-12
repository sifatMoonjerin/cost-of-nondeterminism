"""Hunt for random 6-state NFAs whose minimized DFA reaches the 2^n ceiling (64 states for n = 6).

Generates distinct random NFAs (same parameter distribution as the earlier sweeps, fixed at 6 states) until TARGET of them
reach the ceiling. No NFA is generated twice within this run: every NFA's key is checked against everything seen so far.
Two NFAs that differ only by swapping the symbols 0 and 1 count as the same NFA (generate_nfa picks symbols from a set, so
the same seed can give either labeling in different runs). The first hit comes from the earlier 6-state run as a head start;
that run's other NFAs can't be excluded, because only their keys were stored and a key can't be turned back into an NFA.

    python -m experiments.ceiling_hunt                 # until 10 hits
    python -m experiments.ceiling_hunt --target 3      # quick check

Outputs go to data/experiment_ceiling_hunt_6_states/.
"""
import argparse
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

import duckdb
import pandas as pd

from core.automata import generate_nfa, minimize_dfa, nfa_from_dict, nfa_to_dfa, nfa_to_dict
from experiments.blowup_experiment import DATABASE_NAME, ROOT_DIR, classify_sweep_nfa, get_structural_features
from experiments.blowup_experiment_1m import get_dedup_key

NUM_STATES = 6
CEILING = 2 ** NUM_STATES
PREVIOUS_RUN_DIR = ROOT_DIR / "data" / "experiment_1m_6_states"
OUTPUT_DIR = ROOT_DIR / "data" / "experiment_ceiling_hunt_6_states"
SHARD_SIZE = 20_000
SEEDS_PER_SHARD = 1_000_000_000


def swap_symbols(nfa):
    # The same NFA with the symbols 0 and 1 exchanged
    swapped = nfa_from_dict(nfa_to_dict(nfa))
    swapped.transitions = {
        state: {{"0": "1", "1": "0"}.get(symbol, symbol): destinations for symbol, destinations in symbol_dict.items()}
        for state, symbol_dict in nfa.transitions.items()
    }
    return swapped


def get_canonical_key(nfa):
    # Same key for an NFA and its 0/1 mirror image
    return min(get_dedup_key(nfa), get_dedup_key(swap_symbols(nfa)))


def key_to_int(key):
    # 64 bits of the sha256 key: for ten million NFAs the chance of any collision is about 3 in a million
    return int(key[:16], 16)


# ---------------------------------------------------------------- Worker

def run_shard(task):
    # Draws `size` NFAs from a seed block of its own. Returns every NFA's 64-bit key (so the parent can drop duplicates)
    # and full details for the ones that reach the ceiling.
    shard_id, size, run_seed = task
    param_rng = random.Random(run_seed * 1_000_003 + shard_id)
    nfa_seed = (shard_id + 1) * SEEDS_PER_SHARD
    keys = []
    hits = []
    num_failed = 0
    seen_here = set()

    while len(keys) < size:
        params = {
            "seed": nfa_seed,
            "num_states": NUM_STATES,
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

        key = get_canonical_key(nfa)

        if key in seen_here:
            continue

        seen_here.add(key)
        keys.append(key_to_int(key))
        subset_dfa = nfa_to_dfa(nfa)

        # Minimization can only shrink the DFA, so skip it unless the subset DFA already has all 2^n states
        if len(subset_dfa.states) < CEILING:
            continue

        min_dfa = minimize_dfa(subset_dfa)

        if len(min_dfa.states) == CEILING:
            hits.append({
                "nfa_key": key,
                "nfa_type": classify_sweep_nfa(params),
                "seed": params["seed"],
                "min_transitions_per_state": params["min_transitions_per_state"],
                "max_transitions_per_state": params["max_transitions_per_state"],
                "allow_epsilon": params["allow_epsilon"],
                "subset_states": len(subset_dfa.states),
                "dfa_states": len(min_dfa.states),
                **get_structural_features(nfa),
                "nfa": nfa_to_dict(nfa),
            })

    return keys, hits, num_failed


# ---------------------------------------------------------------- Head start

def load_head_start():
    # NFAs from the earlier 6-state run that already reach the ceiling, rebuilt from their seeds. generate_nfa may give the
    # 0/1-mirrored labeling of the stored NFA; that is the same NFA up to renaming, with the same DFA size.
    con = duckdb.connect(str(PREVIOUS_RUN_DIR / DATABASE_NAME), read_only=True)
    rows = con.sql(f"SELECT * FROM nfa_results WHERE dfa_states = {CEILING}").df()
    con.close()

    hits = []
    for row in rows.itertuples():
        nfa = generate_nfa(
            seed=int(row.seed), num_states=NUM_STATES,
            min_transitions_per_state=int(row.min_transitions_per_state),
            max_transitions_per_state=int(row.max_transitions_per_state),
            allow_epsilon=row.nfa_type.endswith("epsilon"),
        )
        dfa = minimize_dfa(nfa_to_dfa(nfa))
        assert len(dfa.states) == CEILING
        hits.append({
            "nfa_key": get_canonical_key(nfa), "nfa_type": row.nfa_type, "seed": int(row.seed),
            "min_transitions_per_state": int(row.min_transitions_per_state),
            "max_transitions_per_state": int(row.max_transitions_per_state),
            "allow_epsilon": row.nfa_type.endswith("epsilon"),
            "subset_states": len(nfa_to_dfa(nfa).states), "dfa_states": len(dfa.states),
            **get_structural_features(nfa),
            "nfa": nfa_to_dict(nfa),
            "found_in": "earlier 1M run (head start)",
        })

    return hits


# ---------------------------------------------------------------- Main

def main():
    parser = argparse.ArgumentParser(description="Find distinct random 6-state NFAs that reach the 2^n ceiling")
    parser.add_argument("--target", type=int, default=10)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-nfas", type=int, default=100_000_000, help="give up after this many NFAs")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers or __import__("os").cpu_count()

    hits = load_head_start()
    seen = {key_to_int(hit["nfa_key"]) for hit in hits}
    print(f"Head start: {len(hits)} NFA from the earlier run reaches the ceiling "
          f"({', '.join(hit['nfa_key'][:8] for hit in hits)})", flush=True)

    scanned = 0
    duplicates = 0
    failed = 0
    start = time.time()
    next_shard = 0

    def tasks():
        nonlocal next_shard
        while True:
            yield (next_shard, SHARD_SIZE, args.seed)
            next_shard += 1

    with Pool(workers) as pool:
        for keys, shard_hits, num_failed in pool.imap_unordered(run_shard, tasks()):
            failed += num_failed
            new_keys = set()

            for key in keys:
                if key in seen or key in new_keys:
                    duplicates += 1
                else:
                    new_keys.add(key)

            seen |= new_keys
            scanned += len(new_keys)

            for hit in shard_hits:
                if key_to_int(hit["nfa_key"]) in new_keys and len(hits) < args.target:
                    hit["found_in"] = f"this run, after {scanned:,} new NFAs"
                    hits.append(hit)
                    print(f"  hit {len(hits)}/{args.target}: {hit['nfa_type']:15s} {hit['nfa_key'][:12]}  "
                          f"(after {scanned:,} new NFAs, {time.time() - start:.0f}s)", flush=True)

            if len(hits) >= args.target or scanned >= args.max_nfas:
                break

            if next_shard % 25 == 0:
                print(f"  scanned {scanned:,} new NFAs, {len(hits)} hits ({time.time() - start:.0f}s)", flush=True)

        pool.terminate()

    elapsed = time.time() - start
    print(f"\nStopped after {scanned:,} new distinct NFAs in {elapsed:.0f}s "
          f"({duplicates:,} duplicates dropped, {failed:,} draws with no reachable NFA)")

    assert len({hit["nfa_key"] for hit in hits}) == len(hits), "duplicate hits"

    table = pd.DataFrame([{key: value for key, value in hit.items() if key != "nfa"} for hit in hits])
    table.to_csv(args.out_dir / "hits.csv", index=False)

    with open(args.out_dir / "hits_nfas.json", "w") as file:
        json.dump([{"id": hit["nfa_key"], **hit["nfa"]} for hit in hits], file, indent=4)

    summary = {
        "target": args.target, "hits": len(hits), "num_states": NUM_STATES, "ceiling": CEILING,
        "head_start_hits": sum(1 for hit in hits if "head start" in hit["found_in"]),
        "new_distinct_nfas_scanned": scanned, "duplicates_dropped": duplicates,
        "new_hits": sum(1 for hit in hits if "head start" not in hit["found_in"]),
        "hit_rate_per_million": 1e6 * sum(1 for hit in hits if "head start" not in hit["found_in"]) / max(scanned, 1),
        "seconds": round(elapsed),
    }
    with open(args.out_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print()
    print(table[["nfa_key", "nfa_type", "min_transitions_per_state", "max_transitions_per_state", "num_accept_states",
                 "nondeterministic_pairs", "num_symbol_transitions", "num_epsilon_transitions", "found_in"]].to_string(index=False))
    print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
