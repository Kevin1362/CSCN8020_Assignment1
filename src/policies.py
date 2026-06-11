"""
policies.py
Policy classes for CSCN8020 Assignment 1.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

State = Tuple[int, int]


class RandomPolicy:
    """Uniform random behavior policy b(a|s)."""

    def __init__(self, actions: List[str], seed: int = 42) -> None:
        self.actions = actions
        self.rng = np.random.default_rng(seed)

    def action_prob(self, state: State, action: str) -> float:
        return 1.0 / len(self.actions)

    def sample_action(self, state: State) -> str:
        return str(self.rng.choice(self.actions))


class GreedyPolicy:
    """Greedy policy using Q-values or state values."""

    def __init__(self, actions: List[str]) -> None:
        self.actions = actions
        self.policy: Dict[State, str] = {}

    def set_action(self, state: State, action: str) -> None:
        self.policy[state] = action

    def get_action(self, state: State) -> str:
        return self.policy.get(state, self.actions[0])

    def action_prob(self, state: State, action: str) -> float:
        return 1.0 if self.get_action(state) == action else 0.0

    def update_from_q(self, q_values: Dict[Tuple[State, str], float], states: List[State]) -> None:
        for state in states:
            best_action = max(self.actions, key=lambda a: q_values[(state, a)])
            self.policy[state] = best_action
