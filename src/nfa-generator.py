import random

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


def initialize_nfa(num_states=10, seed=None):

    rng = random.Random(seed)

    states = set(range(num_states))
    alphabet = {"0", "1"}

    start_state = rng.choice(list(states))

    num_accept_states = rng.randint(2, 3)

    accept_states = set(
        rng.sample(list(states), num_accept_states)
    )

    nfa = NFA(
        states=states,
        alphabet=alphabet,
        start_state=start_state,
        accept_states=accept_states
    )

    return nfa