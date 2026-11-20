"""Independent correctness checks for the NFA -> subset DFA -> minimized DFA pipeline.

Nothing here calls the construction code (nfa_to_dfa, minimize_dfa, epsilon_closure): the NFA is simulated with its own
ε-closure, and the exact checks explore pairs of states directly. Three checks, each raising ValidationFailure with the
NFA's label, the evidence, and every automaton's answer on the first disagreement:

  check_random_strings         random strings: the NFA, the subset DFA and the minimized DFA must agree on each
  check_subset_vs_minimized    exact: explore every pair of states reachable together; accepting status must always agree
  check_nfa_vs_subset          exact: explore every (NFA state set, DFA state) pair reachable together; same requirement

The two exact checks prove equivalence (not just agreement on a sample): two deterministic automata, or an NFA and a DFA,
accept the same language exactly when no reachable pair of states disagrees on acceptance.
"""
import random
from collections import deque

from core.automata import EPSILON


class ValidationFailure(Exception):
    def __init__(self, label, check, detail):
        self.label = label
        self.check = check
        self.detail = detail
        super().__init__(f"{check} failed for {label}: {detail}")

    def __reduce__(self):
        # Keep the three fields when the exception crosses a process boundary (multiprocessing pickles it)
        return (ValidationFailure, (self.label, self.check, self.detail))


# ---------------------------------------------------------------- Simulation (independent of the construction code)

def closure(nfa, states):
    # ε-closure by search over ε edges
    seen = set(states)
    stack = list(states)
    while stack:
        state = stack.pop()
        for nxt in nfa.transitions.get(state, {}).get(EPSILON, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return frozenset(seen)


def step(nfa, states, symbol):
    moved = set()
    for state in states:
        moved |= nfa.transitions.get(state, {}).get(symbol, set())
    return closure(nfa, moved)


def nfa_accepts(nfa, string):
    current = closure(nfa, {nfa.start_state})
    for symbol in string:
        current = step(nfa, current, symbol)
    return bool(current & nfa.accept_states)


def dfa_accepts(dfa, string):
    state = dfa.start_state
    for symbol in string:
        state = dfa.transitions.get(state, {}).get(symbol)
        if state is None:
            return False
    return state in dfa.accept_states


# ---------------------------------------------------------------- Checks

def check_random_strings(nfa, subset_dfa, min_dfa, label, rng, count=500, max_length=20):
    symbols = sorted(nfa.alphabet)
    for _ in range(count):
        string = "".join(rng.choice(symbols) for _ in range(rng.randint(0, max_length)))
        answers = (nfa_accepts(nfa, string), dfa_accepts(subset_dfa, string), dfa_accepts(min_dfa, string))
        if len(set(answers)) != 1:
            raise ValidationFailure(label, "random strings", f"string {string!r}: NFA={answers[0]}, subset DFA={answers[1]}, minimized DFA={answers[2]}")


def check_subset_vs_minimized(subset_dfa, min_dfa, label):
    symbols = sorted(subset_dfa.alphabet)
    start = (subset_dfa.start_state, min_dfa.start_state)
    seen = {start}
    queue = deque([(start, "")])
    while queue:
        (a, b), path = queue.popleft()
        if (a in subset_dfa.accept_states) != (b in min_dfa.accept_states):
            raise ValidationFailure(label, "subset vs minimized", f"string {path!r}: subset DFA={a in subset_dfa.accept_states}, minimized DFA={b in min_dfa.accept_states}")
        for symbol in symbols:
            pair = (subset_dfa.transitions[a][symbol], min_dfa.transitions[b][symbol])
            if pair not in seen:
                seen.add(pair)
                queue.append((pair, path + symbol))
    return len(seen)


def check_nfa_vs_subset(nfa, subset_dfa, label):
    symbols = sorted(nfa.alphabet)
    start = (closure(nfa, {nfa.start_state}), subset_dfa.start_state)
    seen = {start}
    queue = deque([(start, "")])
    while queue:
        (nfa_set, dfa_state), path = queue.popleft()
        if bool(nfa_set & nfa.accept_states) != (dfa_state in subset_dfa.accept_states):
            raise ValidationFailure(label, "NFA vs subset", f"string {path!r}: NFA={bool(nfa_set & nfa.accept_states)}, subset DFA={dfa_state in subset_dfa.accept_states}")
        for symbol in symbols:
            pair = (step(nfa, nfa_set, symbol), subset_dfa.transitions[dfa_state][symbol])
            if pair not in seen:
                seen.add(pair)
                queue.append((pair, path + symbol))
    return len(seen)


def validate_nfa(nfa, subset_dfa, min_dfa, label, random_seed=None, strings=500):
    # Both exact checks always; the random-string check when random_seed is given. Returns True if the random check ran.
    check_nfa_vs_subset(nfa, subset_dfa, label)
    check_subset_vs_minimized(subset_dfa, min_dfa, label)
    if random_seed is None:
        return False
    check_random_strings(nfa, subset_dfa, min_dfa, label, random.Random(random_seed), strings)
    return True
