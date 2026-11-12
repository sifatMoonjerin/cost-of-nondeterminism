"""Draw any NFA with Graphviz (the counterpart of visualize_dfa.py).

    from visualization.visualize_nfa import render_nfa
    render_nfa(nfa, "nfa.png")

Red edges are the non-determinism: several destinations for the same (state, symbol). Dashed blue edges are ε moves.
Needs the Graphviz `dot` binary on the PATH.
"""
import shutil
import subprocess
from pathlib import Path

from core.automata import EPSILON

FILL_COLOR = "#C9DAF8"
EDGE_COLOR = "#555555"
CHOICE_COLOR = "#C0392B"
EPSILON_COLOR = "#2F5DA8"


def quote(text):
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'


def nfa_to_dot(nfa, rankdir="LR", font="Georgia", title=None):
    """Graphviz DOT source for an NFA. Handles states that only appear in transitions, and ε edges."""
    states = sorted(set(nfa.states) | set(nfa.accept_states) | {nfa.start_state}
                    | {t for d in nfa.transitions.values() for ds in d.values() for t in ds})

    lines = ["digraph NFA {", f"  rankdir={rankdir}; nodesep=0.45; ranksep=0.8; bgcolor=\"white\"; pad=0.2;"]

    if title:
        lines.append(f"  labelloc=t; fontname={quote(font)}; fontsize=16; label={quote(title)};")

    lines.append(f'  node [fontname={quote(font)}, fontsize=18, shape=circle, style=filled, fillcolor="{FILL_COLOR}", '
                 f'color="{EDGE_COLOR}", width=0.7, fixedsize=true];')
    lines.append(f'  edge [fontname={quote(font)}, fontsize=15, color="{EDGE_COLOR}", arrowsize=0.8];')
    lines.append('  start [shape=none, label="", width=0, height=0];')

    for state in states:
        shape = ", shape=doublecircle" if state in nfa.accept_states else ""
        lines.append(f'  q{state} [label="q{state}"{shape}];')

    lines.append(f"  start -> q{nfa.start_state};")

    # One arrow per (from, to) pair listing its symbols; red when that pair is one of several choices for a symbol
    for state in states:
        by_target = {}
        for symbol, destinations in sorted(nfa.transitions.get(state, {}).items()):
            for target in destinations:
                by_target.setdefault(target, []).append((symbol, len(destinations) > 1))

        for target, items in sorted(by_target.items()):
            label = ", ".join(sorted(symbol for symbol, _ in items))
            is_choice = any(choice for symbol, choice in items if symbol != EPSILON)

            if all(symbol == EPSILON for symbol, _ in items):
                style = f'style=dashed, color="{EPSILON_COLOR}", fontcolor="{EPSILON_COLOR}"'
            elif is_choice:
                style = f'color="{CHOICE_COLOR}", fontcolor="{CHOICE_COLOR}", penwidth=2.2'
            else:
                style = ""

            lines.append(f"  q{state} -> q{target} [label={quote(label)}{', ' + style if style else ''}];")

    lines.append("}")
    return "\n".join(lines)


def render_nfa(nfa, path, dpi=200, **options):
    """Render an NFA to an image; the format comes from the extension (png, svg, pdf ...). Returns the path."""
    path = Path(path)

    if shutil.which("dot") is None:
        raise RuntimeError("Graphviz 'dot' was not found. Install Graphviz (e.g. `brew install graphviz`).")

    path.parent.mkdir(parents=True, exist_ok=True)
    command = ["dot", f"-T{path.suffix.lstrip('.') or 'png'}", f"-Gdpi={dpi}", "-o", str(path)]
    result = subprocess.run(command, input=nfa_to_dot(nfa, **options), text=True, capture_output=True)

    if result.returncode != 0:
        raise RuntimeError(f"Graphviz failed: {result.stderr.strip()}")

    return path
