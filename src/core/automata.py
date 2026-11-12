import hashlib
import json
import random
from collections import deque

EPSILON = "ε"


class NFA:
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions or {}
        self.start_state = start_state
        self.accept_states = accept_states

    def add_transition(self, from_state, input_symbol, to_state):
        if from_state not in self.transitions:
            self.transitions[from_state] = {}
        if input_symbol not in self.transitions[from_state]:
            self.transitions[from_state][input_symbol] = set()
        self.transitions[from_state][input_symbol].add(to_state)

    def get_transitions(self, state, input_symbol):
        if state in self.transitions and input_symbol in self.transitions[state]:
            return self.transitions[state][input_symbol]
        return set()

    def is_accepting(self, state):
        return state in self.accept_states


class DFA:
    def __init__(self, states, alphabet, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.start_state = start_state
        self.accept_states = accept_states
        self.transitions = {}

    def add_transition(self, from_state, input_symbol, to_state):
        if from_state not in self.states or to_state not in self.states:
            raise ValueError("Invalid state.")

        if input_symbol not in self.alphabet:
            raise ValueError("Invalid input symbol.")

        if from_state not in self.transitions:
            self.transitions[from_state] = {}

        self.transitions[from_state][input_symbol] = to_state

    def get_transition(self, state, input_symbol):
        return self.transitions.get(state, {}).get(input_symbol)


# ---------------------------------------------------------------- NFA generation

def initialize_nfa(num_states, rng):
    if num_states < 3:
        raise ValueError("Number of states must be at least 3.")

    states = set(range(num_states))
    alphabet = {"0", "1"}

    start_state = 0

    num_accept_states = rng.randint(1, 3)

    accept_states = set(
        rng.sample(list(states), num_accept_states)
    )

    nfa = NFA(
        states=states,
        alphabet=alphabet,
        transitions={},
        start_state=start_state,
        accept_states=accept_states
    )

    return nfa


def get_reachable_states(nfa, start_state = 0):
    reachable = set()
    stack = [start_state]

    while stack:
        current_state = stack.pop()
        if current_state not in reachable:
            reachable.add(current_state)
            for symbol in nfa.alphabet | {EPSILON}:
                for next_state in nfa.get_transitions(current_state, symbol):
                    stack.append(next_state)

    return reachable


def generate_random_transitions(nfa, min_transitions_per_state, max_transitions_per_state, rng, allow_epsilon):
    for state in nfa.states:
        num_transitions = rng.randint(min_transitions_per_state, max_transitions_per_state)
        for _ in range(num_transitions):
            # sorted: a set's order changes between runs, which made the same seed give different NFAs
            input_symbol = rng.choice(sorted(nfa.alphabet))
            to_state = rng.choice(list(nfa.states))
            nfa.add_transition(state, input_symbol, to_state)

    if allow_epsilon:
        num_epsilon_transitions = rng.randint(1, 3)
        for _ in range(num_epsilon_transitions):
            from_state = rng.choice(list(nfa.states))
            to_state = rng.choice(list(nfa.states - {from_state}))
            nfa.add_transition(from_state, EPSILON, to_state)


def all_states_reachable(nfa):
    reachable_states = get_reachable_states(nfa, nfa.start_state)

    return reachable_states == nfa.states


def generate_nfa(seed= 0, num_states=10, min_transitions_per_state=1, max_transitions_per_state=3, allow_epsilon=False):
    rng = random.Random(seed)
    max_attempts = 1000

    for _ in range(max_attempts):
        nfa = initialize_nfa(num_states, rng)
        generate_random_transitions(
            nfa,
            min_transitions_per_state,
            max_transitions_per_state,
            rng,
            allow_epsilon
        )

        if all_states_reachable(nfa):
            return nfa

    raise RuntimeError("Could not generate a fully reachable NFA.")


# ---------------------------------------------------------------- NFA identity and serialization

def get_nfa_signature(nfa):
    transitions = frozenset(
        (from_state, symbol, to_state)
        for from_state, symbol_dict in nfa.transitions.items()
        for symbol, destinations in symbol_dict.items()
        for to_state in destinations
    )

    return (
        frozenset(nfa.states),
        frozenset(nfa.alphabet),
        nfa.start_state,
        frozenset(nfa.accept_states),
        transitions
    )


def get_nfa_id(nfa):
    # Hash of a canonical (sorted) form, so the same NFA gets the same id on every run.
    # Python's built-in hash() is salted per process, so it can't be used here.
    canonical = json.dumps([
        sorted(nfa.states),
        sorted(nfa.alphabet),
        nfa.start_state,
        sorted(nfa.accept_states),
        sorted(
            (from_state, symbol, to_state)
            for from_state, symbol_dict in nfa.transitions.items()
            for symbol, destinations in symbol_dict.items()
            for to_state in destinations
        ),
    ])

    return "nfa_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]


def nfa_to_dict(nfa):
    return {
        "states": sorted(nfa.states),
        "alphabet": sorted(nfa.alphabet),
        "start_state": nfa.start_state,
        "accept_states": sorted(nfa.accept_states),
        "transitions": {
            str(state): {
                symbol: sorted(destinations)
                for symbol, destinations in symbol_dict.items()
            }
            for state, symbol_dict in nfa.transitions.items()
        }
    }


def nfa_from_dict(data):
    nfa = NFA(
        states=set(data["states"]),
        alphabet=set(data["alphabet"]),
        transitions={},
        start_state=data["start_state"],
        accept_states=set(data["accept_states"])
    )

    for from_state, symbol_dict in data["transitions"].items():
        for symbol, destinations in symbol_dict.items():
            for to_state in destinations:
                nfa.add_transition(int(from_state), symbol, to_state)

    return nfa


# ---------------------------------------------------------------- NFA to DFA

def epsilon_closure(nfa, states):
    closure = set(states)
    stack = list(states)

    while stack:
        current_state = stack.pop()
        for next_state in nfa.get_transitions(current_state, EPSILON):
            if next_state not in closure:
                closure.add(next_state)
                stack.append(next_state)

    return closure


def move(nfa, states, input_symbol):
    next_states = set()
    for state in states:
        next_states.update(nfa.get_transitions(state, input_symbol))
    return next_states


def nfa_to_dfa(nfa):
    start_state = frozenset(
        epsilon_closure(nfa, {nfa.start_state})
    )

    dfa_states = {start_state}
    dfa_accept_states = set()
    dfa_transitions = {}

    queue = deque([start_state])

    while queue:
        current_state = queue.popleft()

        if current_state & nfa.accept_states:
            dfa_accept_states.add(current_state)

        for symbol in nfa.alphabet:
            next_state = epsilon_closure(
                nfa,
                move(
                    nfa,
                    epsilon_closure(nfa, current_state),
                    symbol
                )
            )

            next_state = frozenset(next_state)

            if next_state not in dfa_states:
                dfa_states.add(next_state)
                queue.append(next_state)

            if current_state not in dfa_transitions:
                dfa_transitions[current_state] = {}

            dfa_transitions[current_state][symbol] = next_state

    dfa = DFA(
        states=dfa_states,
        alphabet=nfa.alphabet,
        start_state=start_state,
        accept_states=dfa_accept_states
    )

    dfa.transitions = dfa_transitions

    return dfa


# ---------------------------------------------------------------- DFA minimization
# Moore's partition refinement:
# E. F. Moore, "Gedanken-experiments on sequential machines", Automata Studies, 1956.

def initial_partition(dfa):
    # Accepting and non-accepting states are distinguishable by the empty string
    return {state: int(state in dfa.accept_states) for state in dfa.states}


def refine_partition(dfa, block_of):
    # Split blocks: states stay together only if every symbol leads to the same block
    symbols = sorted(dfa.alphabet)
    block_ids = {}
    new_block_of = {}

    for state in dfa.states:
        key = (
            block_of[state],
            tuple(block_of[dfa.get_transition(state, symbol)] for symbol in symbols)
        )

        if key not in block_ids:
            block_ids[key] = len(block_ids)

        new_block_of[state] = block_ids[key]

    return new_block_of


def minimize_dfa(dfa):
    block_of = initial_partition(dfa)
    num_blocks = len(set(block_of.values()))

    # Refinement only splits blocks, so an unchanged block count means a fixed point
    while True:
        block_of = refine_partition(dfa, block_of)
        new_num_blocks = len(set(block_of.values()))

        if new_num_blocks == num_blocks:
            break

        num_blocks = new_num_blocks

    # Each block of equivalent states becomes one state of the minimized DFA
    min_dfa = DFA(
        states=set(block_of.values()),
        alphabet=dfa.alphabet,
        start_state=block_of[dfa.start_state],
        accept_states={block_of[state] for state in dfa.accept_states}
    )

    # Equivalent states agree on successor blocks, so repeated writes are identical
    for state in dfa.states:
        for symbol in dfa.alphabet:
            min_dfa.add_transition(
                block_of[state],
                symbol,
                block_of[dfa.get_transition(state, symbol)]
            )

    return min_dfa
