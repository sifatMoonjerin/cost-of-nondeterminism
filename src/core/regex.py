"""Convert a finite automaton to a regular expression by state elimination (the GNFA method).

Works on the project's DFA and NFA classes (ε edges of an NFA are handled). The result is a string using
`|` (union), `*` (star), juxtaposition (concatenation) and `ε` (empty string); `∅` means the empty language.
`to_python_pattern` turns it into a pattern for Python's `re`, and `check_against_automaton` compares the
expression with the automaton on every string up to a given length so the conversion can be verified.

    from core.regex import automaton_to_regex, check_against_automaton
    pattern = automaton_to_regex(dfa)
"""
import itertools
import re

from core.automata import EPSILON

# Expression trees: None (empty language), ("eps",), ("sym", c), ("cat", [..]), ("alt", [..]), ("star", x)
EPS = ("eps",)


def make_alt(left, right):
    if left is None:
        return right
    if right is None:
        return left

    parts = []
    for part in (left, right):
        for item in (part[1] if part[0] == "alt" else [part]):
            if item not in parts:
                parts.append(item)

    # x|ε with x already able to match ε needs no ε; x* absorbs ε
    if EPS in parts and any(p[0] == "star" for p in parts):
        parts.remove(EPS)
    return parts[0] if len(parts) == 1 else ("alt", parts)


def make_cat(left, right):
    if left is None or right is None:
        return None
    if left == EPS:
        return right
    if right == EPS:
        return left

    parts = (left[1] if left[0] == "cat" else [left]) + (right[1] if right[0] == "cat" else [right])
    return ("cat", parts)


def make_star(expr):
    if expr is None or expr == EPS:
        return EPS
    if expr[0] == "star":
        return expr
    if expr[0] == "alt" and EPS in expr[1]:
        rest = [p for p in expr[1] if p != EPS]
        expr = rest[0] if len(rest) == 1 else ("alt", rest)
    return ("star", expr)


def render(expr, parent=None):
    if expr is None:
        return "∅"
    kind = expr[0]
    if kind == "eps":
        return "ε"
    if kind == "sym":
        return expr[1]
    if kind == "star":
        inner = render(expr[1], "star")
        return inner + "*"
    if kind == "cat":
        text = "".join(render(p, "cat") for p in expr[1])
        return f"({text})" if parent == "star" else text
    text = "|".join(render(p, "alt") for p in expr[1])
    return f"({text})" if parent in ("star", "cat") else text


# ---------------------------------------------------------------- automaton -> expression

def edges_of(automaton):
    """Edge labels {(from, to): expression} for a DFA or NFA, as the starting GNFA."""
    edges = {}

    def add(source, target, label):
        edges[(source, target)] = make_alt(edges.get((source, target)), label)

    for source, by_symbol in automaton.transitions.items():
        for symbol, targets in by_symbol.items():
            label = EPS if symbol == EPSILON else ("sym", symbol)
            for target in (targets if isinstance(targets, (set, frozenset, list)) else [targets]):
                add(source, target, label)
    return edges


def useful_states(automaton, edges):
    # Keep states reachable from the start AND able to reach an accepting state; dead states add only ∅ branches
    forward, backward = {}, {}
    for source, target in edges:
        forward.setdefault(source, set()).add(target)
        backward.setdefault(target, set()).add(source)

    def reach(seeds, graph):
        seen, stack = set(seeds), list(seeds)
        while stack:
            for nxt in graph.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    return reach({automaton.start_state}, forward) & reach(automaton.accept_states, backward)


def automaton_to_regex(automaton):
    states = list(useful_states(automaton, edges_of(automaton)))
    if automaton.start_state not in states:
        return "∅"

    edges = {key: label for key, label in edges_of(automaton).items() if key[0] in states and key[1] in states}
    start, final = "start", "final"
    edges[(start, automaton.start_state)] = EPS
    for accept in automaton.accept_states:
        if accept in states:
            edges[(accept, final)] = make_alt(edges.get((accept, final)), EPS)

    # Eliminate the state with the fewest in x out edges first; this keeps the expression small
    remaining = list(states)
    while remaining:
        def cost(state):
            ins = sum(1 for (a, b) in edges if b == state and a != state)
            outs = sum(1 for (a, b) in edges if a == state and b != state)
            return ins * outs

        victim = min(remaining, key=cost)
        remaining.remove(victim)
        loop = make_star(edges.get((victim, victim)))
        ins = [(a, label) for (a, b), label in edges.items() if b == victim and a != victim]
        outs = [(b, label) for (a, b), label in edges.items() if a == victim and b != victim]

        for source, into in ins:
            for target, out in outs:
                path = make_cat(make_cat(into, loop), out)
                edges[(source, target)] = make_alt(edges.get((source, target)), path)

        edges = {key: label for key, label in edges.items() if victim not in key}

    return render(edges.get((start, final)))


# ---------------------------------------------------------------- checking

def to_python_pattern(regex):
    return "(?:)" if regex == "ε" else regex.replace("ε", "(?:)").replace("∅", "(?!)")


def automaton_accepts(automaton, string):
    # Works for a DFA, or for an NFA without ε edges
    current = {automaton.start_state}
    if hasattr(automaton, "get_transition"):
        for symbol in string:
            current = {automaton.get_transition(s, symbol) for s in current} - {None}
    else:
        for symbol in string:
            current = {t for s in current for t in automaton.get_transitions(s, symbol)}
    return bool(current & automaton.accept_states)


def check_against_automaton(automaton, regex, max_length=12):
    """Compare regex and automaton on every string over the alphabet up to max_length. Returns the first mismatch or None."""
    pattern = re.compile(to_python_pattern(regex))
    alphabet = sorted(automaton.alphabet - {EPSILON})
    for length in range(max_length + 1):
        for chars in itertools.product(alphabet, repeat=length):
            string = "".join(chars)
            if bool(pattern.fullmatch(string)) != automaton_accepts(automaton, string):
                return string
    return None
