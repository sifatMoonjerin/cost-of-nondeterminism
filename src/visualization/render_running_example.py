"""Draw the subset-construction DFA of the running example (the blow-up family) for n = 1 ... 10.

The n-th symbol from the end is 1: an NFA with n + 1 states (q0 ... qn) whose DFA has 2^n states.
Outputs go to data/visualizations/running_example/.
"""
import time
from pathlib import Path

from core.automata import NFA, nfa_to_dfa
from visualization.visualize_dfa import render_dfa

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "data" / "visualizations" / "running_example"


def build_blowup_nfa(n):
    # Same construction as generate_blowup_nfas in the notebook: q0 loops and guesses, a chain of n - 1 counts out the rest
    nfa = NFA(
        states=set(range(n + 1)),
        alphabet={"0", "1"},
        transitions={},
        start_state=0,
        accept_states={n},
    )

    nfa.add_transition(0, "0", 0)
    nfa.add_transition(0, "1", 0)
    nfa.add_transition(0, "1", 1)

    for i in range(1, n):
        nfa.add_transition(i, "0", i + 1)
        nfa.add_transition(i, "1", i + 1)

    return nfa


def main(max_n=10):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for n in range(1, max_n + 1):
        dfa = nfa_to_dfa(build_blowup_nfa(n))
        assert len(dfa.states) == 2 ** n

        # Small DFAs get letter names and their NFA-state sets; big ones are drawn plainly as SVG (PNG gets enormous). dot is fine to n = 3, neato to n = 6, sfdp beyond
        small = n <= 4
        options = {
            "names": "letters",
            "show_sets": small,
            "layout": "dot" if n <= 3 else "neato" if n <= 6 else "sfdp",
            "title": f"n = {n}: NFA with {n + 1} states, DFA with {2 ** n} states",
        }
        extension = "png" if n <= 6 else "svg"
        path = OUTPUT_DIR / f"n{n:02d}_{2 ** n}_states.{extension}"

        start = time.time()
        render_dfa(dfa, path, **options)
        print(f"n = {n:2d}: {len(dfa.states):5d} DFA states -> {path.name} ({time.time() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
