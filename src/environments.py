"""
environments.py
GridWorld environment classes for CSCN8020 Assignment 1.

This file keeps the environment logic separate from the agents.
The GridWorldEnvironment class uses a simple deterministic transition model.
Required course reference used: ProfEspinosaAIML/HelloGymMaze.
This class follows the Gymnasium environment pattern with reset(), step(),
observation_space, action_space, reward, and terminal state logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # gymnasium is optional for import safety
    gym = object
    spaces = None


State = Tuple[int, int]


@dataclass(frozen=True)
class StepResult:
    """Small container for environment step output."""
    next_state: State
    reward: float
    done: bool


class GridWorldEnvironment(gym.Env if hasattr(gym, "Env") else object):
    """
    Deterministic gridworld.

    reward_mode:
        "next" means reward is based on next state after action.
        "current" means reward is based on current state before action.
    """

    metadata = {"render_modes": ["human"]}

    ACTIONS: Dict[str, State] = {
        "right": (0, 1),
        "down": (1, 0),
        "left": (0, -1),
        "up": (-1, 0),
    }

    def __init__(
        self,
        rows: int,
        cols: int,
        terminal_states: Optional[Sequence[State]] = None,
        grey_states: Optional[Sequence[State]] = None,
        reward_regular: float = -1.0,
        reward_grey: float = -5.0,
        reward_goal: float = 10.0,
        reward_map: Optional[Dict[State, float]] = None,
        start_state: State = (0, 0),
        reward_mode: str = "next",
        seed: int = 42,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.states: List[State] = [(r, c) for r in range(rows) for c in range(cols)]
        self.terminal_states = set(terminal_states or [])
        self.grey_states = set(grey_states or [])
        self.reward_regular = reward_regular
        self.reward_grey = reward_grey
        self.reward_goal = reward_goal
        self.reward_map = reward_map or {}
        self.start_state = start_state
        self.current_state = start_state
        self.reward_mode = reward_mode
        self.rng = np.random.default_rng(seed)

        if spaces is not None:
            self.observation_space = spaces.Discrete(rows * cols)
            self.action_space = spaces.Discrete(len(self.ACTIONS))

    @property
    def action_names(self) -> List[str]:
        return list(self.ACTIONS.keys())

    def state_to_index(self, state: State) -> int:
        return state[0] * self.cols + state[1]

    def index_to_state(self, index: int) -> State:
        return (index // self.cols, index % self.cols)

    def is_inside_grid(self, state: State) -> bool:
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_terminal(self, state: State) -> bool:
        return state in self.terminal_states

    def get_non_terminal_states(self) -> List[State]:
        return [s for s in self.states if not self.is_terminal(s)]

    def reward(self, state: State) -> float:
        """State reward R(s)."""
        if state in self.reward_map:
            return float(self.reward_map[state])
        if state in self.terminal_states:
            return float(self.reward_goal)
        if state in self.grey_states:
            return float(self.reward_grey)
        return float(self.reward_regular)

    def transition(self, state: State, action: str) -> State:
        """If action hits wall, agent stays in same state."""
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        if self.is_terminal(state):
            return state

        dr, dc = self.ACTIONS[action]
        candidate = (state[0] + dr, state[1] + dc)
        return candidate if self.is_inside_grid(candidate) else state

    def transition_reward(self, state: State, next_state: State) -> float:
        """Immediate reward used after a transition."""
        if self.reward_mode == "current":
            return self.reward(state)
        return self.reward(next_state)

    def reset(self, seed: Optional[int] = None, start_state: Optional[State] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_state = start_state if start_state is not None else self.start_state
        if spaces is not None:
            return self.state_to_index(self.current_state), {}
        return self.current_state

    def step(self, action):
        """Gymnasium-style step. Action may be string or integer index."""
        action_name = self.action_names[action] if isinstance(action, int) else action
        next_state = self.transition(self.current_state, action_name)
        reward = self.transition_reward(self.current_state, next_state)
        done = self.is_terminal(next_state)
        self.current_state = next_state

        if spaces is not None:
            return self.state_to_index(next_state), reward, done, False, {}
        return StepResult(next_state=next_state, reward=reward, done=done)

    def random_non_terminal_state(self) -> State:
        states = self.get_non_terminal_states()
        idx = int(self.rng.integers(0, len(states)))
        return states[idx]

    def render(self) -> None:
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                s = (r, c)
                if s == self.current_state:
                    row.append("A")
                elif s in self.terminal_states:
                    row.append("G")
                elif s in self.grey_states:
                    row.append("X")
                else:
                    row.append(".")
            print(" ".join(row))

    @classmethod
    def make_2x2(cls) -> "GridWorldEnvironment":
        reward_map = {(0, 0): 5.0, (0, 1): 10.0, (1, 0): 1.0, (1, 1): 2.0}
        return cls(rows=2, cols=2, reward_map=reward_map, reward_mode="current", start_state=(0, 0))

    @classmethod
    def make_5x5(cls) -> "GridWorldEnvironment":
        return cls(
            rows=5,
            cols=5,
            terminal_states=[(4, 4)],
            grey_states=[(2, 2), (3, 0), (0, 4)],
            reward_regular=-1.0,
            reward_grey=-5.0,
            reward_goal=10.0,
            reward_mode="next",
            start_state=(0, 0),
        )
