"""Run the three correctness checks (core/validate.py) over the NFAs of the experiments and write one results file per run.

    cd src
    python -m experiments.validate_runs stored               # 10k run, first-experiment JSON files, ceiling hits
    python -m experiments.validate_runs structured --states 6      # regenerate the structured NFAs from their stored seeds
    python -m experiments.validate_runs baseline --states 6        # same for the unrestricted 2M run

Each run writes data/validation/<run>.json: how many NFAs were checked exactly (NFA vs subset DFA, subset DFA vs minimized DFA),
how many also got the random-string check, strings per NFA, the test seed and the number of failures. The first failure stops
the run with the NFA's label, the string and the three automata's answers.
"""
import argparse
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

import duckdb

from core.automata import generate_nfa, minimize_dfa, nfa_from_dict, nfa_to_dfa
from core.validate import ValidationFailure, validate_nfa
from experiments.blowup_experiment import DATABASE_NAME, GENERATED_DIR, ROOT_DIR
from experiments.blowup_experiment_1m import get_dedup_key
from experiments.ceiling_hunt import get_canonical_key
from experiments.structured_nfa_experiment import generate_structured_nfa

VALIDATION_DIR = ROOT_DIR / "data" / "validation"
STRINGS_PER_NFA = 500
MAX_LENGTH = 20
TEST_SEED = 20261120
CHUNK = 2_000


def check_one(task):
    # task: (label, nfa_dict, random_seed or None, expected_key or None, key_function name)
    label, spec, random_seed, expected_key, key_kind = task
    if isinstance(spec, dict):
        nfa = nfa_from_dict(spec)
    elif spec[0] == "structured":
        # Regenerated inside the worker from the seed stored in the database
        nfa = generate_structured_nfa(spec[1], spec[2])[0]
    else:
        _, seed, num_states, low, high, with_epsilon = spec
        nfa = generate_nfa(seed=seed, num_states=num_states, min_transitions_per_state=low, max_transitions_per_state=high, allow_epsilon=with_epsilon)
    subset_dfa = nfa_to_dfa(nfa)
    min_dfa = minimize_dfa(subset_dfa)
    ran_random = validate_nfa(nfa, subset_dfa, min_dfa, label, random_seed, STRINGS_PER_NFA)
    key_ok = True
    if expected_key is not None:
        key_ok = (get_canonical_key(nfa) if key_kind == "canonical" else get_dedup_key(nfa)) == expected_key
    return ran_random, key_ok


def check_chunk(tasks):
    return [check_one(task) for task in tasks]


def run_checks(name, items, random_share, description, extra=None):
    # items: list of (label, nfa_dict, expected_key, key_kind); random_share 1.0 = every NFA also gets the random-string check
    sampler = random.Random(TEST_SEED)
    tasks = [(label, nfa_dict, (TEST_SEED + index) if sampler.random() < random_share else None, key, kind)
             for index, (label, nfa_dict, key, kind) in enumerate(items)]
    chunks = [tasks[i:i + CHUNK] for i in range(0, len(tasks), CHUNK)]
    start = time.time()
    random_checked = 0
    key_mismatches = 0
    with Pool() as pool:
        for results in pool.imap_unordered(check_chunk, chunks):
            random_checked += sum(1 for ran, _ in results if ran)
            key_mismatches += sum(1 for _, ok in results if not ok)
    summary = {
        "run": name, "description": description, "nfas_checked_exact": len(tasks), "nfas_checked_random_strings": random_checked,
        "strings_per_nfa": STRINGS_PER_NFA, "max_string_length": MAX_LENGTH, "test_seed": TEST_SEED,
        "exact_checks": ["NFA vs subset DFA", "subset DFA vs minimized DFA"], "failures": 0,
        "regenerated_nfa_key_mismatches": key_mismatches if any(item[2] for item in items) else None,
        "seconds": round(time.time() - start, 1), **(extra or {}),
    }
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    (VALIDATION_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return summary


def stored_items(path):
    return [(f"{path.name}[{index}] {record.get('id', '')}", record, None, None) for index, record in enumerate(json.loads(path.read_text()))]


def run_stored():
    run_checks("balanced_10k", stored_items(ROOT_DIR / "data" / "experiment_10k" / "nfas.json"), 1.0,
               "data/experiment_10k/nfas.json: 10,000 random NFAs plus the 7 blow-up NFAs, all checks on every NFA")
    run_checks("first_sweep_3k", stored_items(ROOT_DIR / "data" / "experiment_3k" / "nfas.json"), 1.0,
               "data/experiment_3k/nfas.json: the first sweep's 3,200 random NFAs plus 7 blow-up NFAs, all checks on every NFA")
    items = [item for path in sorted(GENERATED_DIR.glob("*.json")) for item in stored_items(path)]
    run_checks("first_experiment_generated_json", items, 1.0,
               "data/generated/*.json: sparse, dense, sparse-epsilon, dense-epsilon (50 each), blow-up (7) and the old epsilon_nfas.json (50)")
    run_checks("ceiling_hunt_hits", stored_items(ROOT_DIR / "data" / "experiment_ceiling_hunt_6_states" / "hits_nfas.json"), 1.0,
               "the 10 NFAs of the ceiling hunt, all checks")


def regenerate_items(kind, states, limit=None):
    # Only the recipe (seed and parameters) is built here; the workers regenerate each NFA
    if kind == "structured":
        path = ROOT_DIR / "data" / f"experiment_structured_{states}_states" / DATABASE_NAME
        rows = duckdb.connect(str(path), read_only=True).sql("SELECT seed, nfa_key FROM nfa_results" + (f" LIMIT {limit}" if limit else "")).fetchall()
        return [(f"structured n={states} seed {seed}", ("structured", seed, states), key, "canonical") for seed, key in rows]
    path = ROOT_DIR / "data" / f"experiment_unrestricted_{states}_states_2m" / DATABASE_NAME
    rows = duckdb.connect(str(path), read_only=True).sql(
        "SELECT seed, nfa_key, original_states, min_transitions_per_state, max_transitions_per_state, nfa_type FROM nfa_results"
        + (f" LIMIT {limit}" if limit else "")).fetchall()
    return [(f"unrestricted n={states} seed {seed}", ("generate", seed, n, low, high, nfa_type.endswith("epsilon")), key, "dedup")
            for seed, key, n, low, high, nfa_type in rows]


def main():
    parser = argparse.ArgumentParser(description="Validate the NFA -> DFA pipeline on the experiments' NFAs")
    parser.add_argument("target", choices=["stored", "structured", "baseline"])
    parser.add_argument("--states", type=int, default=6)
    parser.add_argument("--random-share", type=float, default=0.01, help="share of regenerated NFAs that also get the random-string check")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.target == "stored":
        run_stored()
        return
    kind = "structured" if args.target == "structured" else "baseline"
    items = regenerate_items("structured" if kind == "structured" else "baseline", args.states, args.limit)
    run_checks(f"{kind}_{args.states}_states", items, args.random_share,
               f"{kind} run for n = {args.states}, NFAs regenerated from the seeds stored in the database; exact checks on all, "
               f"random strings on a {args.random_share:.0%} seeded sample; the regenerated NFA's key is compared with the stored key")


if __name__ == "__main__":
    main()
