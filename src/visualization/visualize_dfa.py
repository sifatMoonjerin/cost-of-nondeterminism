"""Draw any DFA with Graphviz.

Works with the DFA class in automata.py, and with anything that has the same attributes:
states, alphabet, start_state, accept_states and get_transition(state, symbol).

    from visualization.visualize_dfa import render_dfa
    render_dfa(dfa, "dfa.png")                           # small DFA
    render_dfa(dfa, "dfa.svg", names="letters")          # A, B, C ... for subset-construction DFAs
    render_dfa(big_dfa, "big.png", layout="neato")       # organic layout for dense graphs
    render_dfa(huge_dfa, "huge.svg", layout="sfdp")      # 100s of states (dot and neato get very slow)

Needs the Graphviz binaries (`dot`, `neato`, ...) on the PATH; no extra Python package.
"""
import argparse
import json
import shutil
import subprocess
from collections import deque
from pathlib import Path

FILL_COLOR = "#C9DAF8"
EDGE_COLOR = "#555555"
DEAD_COLOR = "#999999"
HIGHLIGHT_COLOR = "#2E8B57"

LAYOUTS = ["dot", "neato", "fdp", "sfdp", "circo", "twopi"]

# Every state is drawn as the same size circle, big enough for the longest label.
# Sizes are in points / inches; label widths are estimated from the font size.
NAME_SIZE = 16
SET_SIZE = 11
CHAR_WIDTH = 0.6
PADDING = 0.12
MIN_DIAMETER = 0.5
RING_GAP = 8 / 72      # a doublecircle's outer ring adds 4pt on each side of its node size


# ---------------------------------------------------------------- Labels

def format_state(state, set_prefix="q"):
    # A subset-construction state is a frozenset of NFA states, shown as {q0,q1}; anything else uses str()
    if isinstance(state, (set, frozenset)):
        if not state:
            return "∅"

        items = sorted(state, key=lambda item: (str(type(item)), item))
        return "{" + ",".join(f"{set_prefix}{item}" if isinstance(item, int) else str(item) for item in items) + "}"

    return str(state)


def get_reachable_order(dfa):
    # Breadth-first from the start state, then any unreachable states, so letter names are stable
    order = [dfa.start_state]
    seen = {dfa.start_state}
    queue = deque(order)

    while queue:
        state = queue.popleft()
        for symbol in sorted(dfa.alphabet):
            next_state = dfa.get_transition(state, symbol)
            if next_state is not None and next_state not in seen:
                seen.add(next_state)
                order.append(next_state)
                queue.append(next_state)

    order += sorted((state for state in dfa.states if state not in seen), key=format_state)
    return order


def get_letter_name(index):
    # A ... Z, then AA, AB ...
    name = ""
    index += 1

    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name

    return name


def get_state_labels(dfa, names, show_sets, set_prefix):
    # Returns {state: label}. names: None (format the state itself), "letters", or a dict {state: name}
    if names is None:
        return {state: format_state(state, set_prefix) for state in dfa.states}

    if names == "letters":
        order = get_reachable_order(dfa)
        names = {state: get_letter_name(index) for index, state in enumerate(order)}

    labels = {}
    for state in dfa.states:
        label = str(names.get(state, format_state(state, set_prefix)))

        if show_sets and isinstance(state, (set, frozenset)):
            label += "\n" + format_state(state, set_prefix)

        labels[state] = label

    return labels


def quote(text):
    # DOT double-quoted string; \n stays a line break
    return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def get_label_sizes(label):
    # First line is the state name; any further lines (the NFA-state set) are drawn smaller
    return [(line, NAME_SIZE if index == 0 else SET_SIZE) for index, line in enumerate(label.split("\n"))]


def format_label(label):
    # Plain one-line label, or an HTML-like one so the set can be a smaller font
    parts = get_label_sizes(label)

    if len(parts) == 1:
        return quote(label)

    escaped = [(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), size) for line, size in parts]
    return "<" + "<BR/>".join(f'<FONT POINT-SIZE="{size}">{line}</FONT>' for line, size in escaped) + ">"


def get_diameter(labels):
    # Smallest circle (inside the ring) that holds the widest label, for every state
    needed = MIN_DIAMETER

    for label in labels.values():
        parts = get_label_sizes(label)
        width = max(len(line) * size * CHAR_WIDTH for line, size in parts) / 72
        height = sum(size * 1.2 for _, size in parts) / 72
        needed = max(needed, (width ** 2 + height ** 2) ** 0.5 + PADDING)

    return needed


# ---------------------------------------------------------------- DOT source

def is_dead_state(dfa, state):
    # Non-accepting and every symbol loops back: the machine can never accept from here
    return state not in dfa.accept_states and all(
        dfa.get_transition(state, symbol) == state for symbol in dfa.alphabet
    )


def trace_edges(dfa, string):
    # The (state, symbol) steps taken on a string, to highlight; stops if a transition is missing
    edges = set()
    state = dfa.start_state

    for symbol in string:
        next_state = dfa.get_transition(state, symbol)
        if next_state is None:
            break

        edges.add((state, next_state))
        state = next_state

    return edges


def dfa_to_dot(dfa, names=None, show_sets=False, set_prefix="q", layout="dot", rankdir="LR",
               dim_dead=True, highlight_string=None, font="Georgia", title=None):
    """Graphviz DOT source for a DFA.

    names:            None to label each state with itself, "letters" for A, B, C ..., or a {state: name} dict
    show_sets:        with names, add the underlying NFA-state set under the name
    layout:           Graphviz engine; "dot" for layered graphs, "neato" for dense ones, "sfdp" for hundreds of states
    rankdir:          "LR" or "TB" (dot only)
    dim_dead:         draw dead (trap) states dashed and gray
    highlight_string: highlight the path this input string takes through the DFA
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")

    labels = get_state_labels(dfa, names, show_sets, set_prefix)
    order = get_reachable_order(dfa)
    ids = {state: f"s{index}" for index, state in enumerate(order)}
    highlighted = trace_edges(dfa, highlight_string) if highlight_string else set()

    lines = ["digraph DFA {"]

    if layout == "dot":
        lines.append(f"  rankdir={rankdir}; nodesep=0.4; ranksep=0.7;")
    elif layout == "sfdp":
        # The only engine that stays fast on hundreds of states; straight edges keep it that way
        lines.append('  overlap=prism; splines=false; sep="+4";')
    else:
        lines.append('  overlap=false; splines=true; sep="+10";')

    lines.append('  bgcolor="white"; pad=0.2;')

    if title:
        lines.append(f'  labelloc=t; fontname={quote(font)}; fontsize=16; label={quote(title)};')

    lines.append(
        f'  node [fontname={quote(font)}, fontsize={NAME_SIZE}, shape=circle, style=filled, '
        f'fillcolor="{FILL_COLOR}", color="{EDGE_COLOR}", fixedsize=true];'
    )
    lines.append(f'  edge [fontname={quote(font)}, fontsize=14, color="{EDGE_COLOR}", arrowsize=0.8];')
    lines.append('  start [shape=none, label="", width=0, height=0, fixedsize=false];')

    # Accepting states are the same size as the rest: their node is smaller by the width of the outer ring
    inner = get_diameter(labels)
    outer = inner + RING_GAP

    for state in order:
        attributes = [f"label={format_label(labels[state])}"]

        if state in dfa.accept_states:
            attributes.append(f"shape=doublecircle, width={inner:.3f}, height={inner:.3f}")
        else:
            attributes.append(f"width={outer:.3f}, height={outer:.3f}")

        if dim_dead and is_dead_state(dfa, state):
            attributes.append(f'style="filled,dashed", fillcolor="#EEEEEE", color="{DEAD_COLOR}", fontcolor="{DEAD_COLOR}"')

        lines.append(f"  {ids[state]} [{', '.join(attributes)}];")

    # Force-directed layouts otherwise leave the start arrow very long
    start_edge = "" if layout == "dot" else " [len=0.4]"
    lines.append(f"  start -> {ids[dfa.start_state]}{start_edge};")

    # One arrow per (from, to) pair, with all its symbols in the label
    for state in order:
        symbols_by_target = {}
        for symbol in sorted(dfa.alphabet):
            target = dfa.get_transition(state, symbol)
            if target is not None:
                symbols_by_target.setdefault(target, []).append(symbol)

        for target, symbols in symbols_by_target.items():
            attributes = [f"label={quote(', '.join(symbols))}"]

            if (state, target) in highlighted:
                attributes.append(f'color="{HIGHLIGHT_COLOR}", fontcolor="{HIGHLIGHT_COLOR}", penwidth=2.4')
            elif dim_dead and is_dead_state(dfa, target):
                attributes.append(f'color="{DEAD_COLOR}", fontcolor="{DEAD_COLOR}", style=dashed')

            lines.append(f"  {ids[state]} -> {ids[target]} [{', '.join(attributes)}];")

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------- Rendering

def render_dfa(dfa, path, dpi=200, **options):
    """Render a DFA to an image. The format comes from the extension (png, svg, pdf ...).

    options are passed to dfa_to_dot. Returns the output path.
    """
    path = Path(path)
    layout = options.get("layout", "dot")

    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")

    if shutil.which(layout) is None:
        raise RuntimeError(f"Graphviz '{layout}' was not found. Install Graphviz (e.g. `brew install graphviz`).")

    output_format = path.suffix.lstrip(".") or "png"
    path.parent.mkdir(parents=True, exist_ok=True)

    command = [layout, f"-T{output_format}", f"-Gdpi={dpi}", "-o", str(path)]
    result = subprocess.run(command, input=dfa_to_dot(dfa, **options), text=True, capture_output=True)

    if result.returncode != 0:
        raise RuntimeError(f"Graphviz failed: {result.stderr.strip()}")

    return path


# ---------------------------------------------------------------- Command line

def main():
    # Draw the subset-construction (or minimized) DFA of one NFA from a data/generated JSON file
    from core.automata import minimize_dfa, nfa_from_dict, nfa_to_dfa

    parser = argparse.ArgumentParser(description="Draw the DFA built from one NFA in a JSON file")
    parser.add_argument("json_file", type=Path, help="e.g. data/generated/blowup_nfas.json")
    parser.add_argument("--index", type=int, default=0, help="which NFA in the file")
    parser.add_argument("--minimize", action="store_true", help="draw the minimized DFA instead")
    parser.add_argument("--string", help="highlight the path this input string takes")
    parser.add_argument("--layout", default="dot", choices=LAYOUTS)
    parser.add_argument("-o", "--output", type=Path, default=Path("dfa.png"))
    args = parser.parse_args()

    with open(args.json_file) as file:
        nfa = nfa_from_dict(json.load(file)[args.index])

    dfa = nfa_to_dfa(nfa)
    names = "letters"
    show_sets = True

    if args.minimize:
        dfa = minimize_dfa(dfa)
        names = None
        show_sets = False

    render_dfa(dfa, args.output, names=names, show_sets=show_sets, layout=args.layout, highlight_string=args.string)
    print(f"Wrote {args.output} ({len(dfa.states)} states)")


if __name__ == "__main__":
    main()
