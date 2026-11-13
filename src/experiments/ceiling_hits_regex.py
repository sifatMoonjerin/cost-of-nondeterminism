"""Regular expressions for the ceiling-hunt NFAs, for finding shared sub-patterns. Each is checked against its NFA on all strings up to length 12.

The minimized DFAs (64 states) are not converted: state elimination on 64 states gives expressions far too large to read.

    cd src && python -m experiments.ceiling_hits_regex
Writes data/experiment_ceiling_hunt_6_states/hits_regex.md
"""
import json
from pathlib import Path

from core.automata import minimize_dfa, nfa_from_dict, nfa_to_dfa
from core.regex import automaton_to_regex, check_against_automaton

HUNT_DIR = Path(__file__).resolve().parents[2] / "data" / "experiment_ceiling_hunt_6_states"


def main():
    entries = json.loads((HUNT_DIR / "hits_nfas.json").read_text())
    lines = ["# Regular expressions of the 10 ceiling NFAs", "",
             "`|` union, `*` star, `ε` empty string. Read straight off the NFA by state elimination (`core/regex.py`); verified on all strings up to length 12.", ""]

    for number, entry in enumerate(entries, 1):
        nfa = nfa_from_dict(entry)
        from_nfa = automaton_to_regex(nfa)
        mismatch = check_against_automaton(nfa, from_nfa, 12)
        assert mismatch is None, (number, mismatch)
        dfa_states = len(minimize_dfa(nfa_to_dfa(nfa)).states)
        lines += [f"## {number}. `{entry['id'][:8]}`  ({dfa_states} DFA states)", "",
                  f"- accepting {entry['accept_states']}, transitions {entry['transitions']}",
                  f"- regex ({len(from_nfa)} chars): `{from_nfa}`", ""]
        print(number, len(from_nfa))

    (HUNT_DIR / "hits_regex.md").write_text("\n".join(lines))
    print(f"Wrote {HUNT_DIR / 'hits_regex.md'}")


if __name__ == "__main__":
    main()
