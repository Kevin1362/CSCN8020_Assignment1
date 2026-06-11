"""
agents.py
Value Iteration, In-Place Value Iteration, and Off-policy Monte Carlo agents.
Required course references used: CSCN8020 playground lec3_DP for value iteration
and lec4_MC for Monte Carlo algorithm structure. The code below is written
in object-oriented form for this assignment.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import numpy as np

from .environments import GridWorldEnvironment, State
from .policies import GreedyPolicy, RandomPolicy


class ValueIterationAgent:
    """
    Standard/synchronous value iteration.

    This maps to the Bellman optimality update:
    V(s) <- max_a [ R + gamma * V(s') ]
    """

    def __init__(
        self,
        env: GridWorldEnvironment,
        gamma: float = 0.9,
        theta: float = 1e-6,
        max_iterations: int = 1000,
        logger: logging.Logger | None = None,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.logger = logger or logging.getLogger(__name__)
        self.V: Dict[State, float] = {state: 0.0 for state in env.states}
        self.iterations: int = 0
        self.runtime_seconds: float = 0.0

    def q_value(self, state: State, action: str, V: Dict[State, float]) -> float:
        next_state = self.env.transition(state, action)
        reward = self.env.transition_reward(state, next_state)
        future = 0.0 if self.env.is_terminal(next_state) else self.gamma * V[next_state]
        return reward + future

    def run(self) -> Dict[State, float]:
        start = time.perf_counter()
        self.logger.info("Standard Value Iteration started.")
        self.logger.info("gamma=%s, theta=%s, max_iterations=%s", self.gamma, self.theta, self.max_iterations)

        for i in range(1, self.max_iterations + 1):
            delta = 0.0
            new_V = self.V.copy()

            for state in self.env.states:
                if self.env.is_terminal(state):
                    new_V[state] = 0.0
                    continue

                old_value = self.V[state]
                action_values = [self.q_value(state, action, self.V) for action in self.env.action_names]
                new_V[state] = max(action_values)
                delta = max(delta, abs(old_value - new_V[state]))

            self.V = new_V
            self.iterations = i
            self.logger.info("Standard VI iteration=%s delta=%.8f", i, delta)

            if delta < self.theta:
                break

        self.runtime_seconds = time.perf_counter() - start
        self.logger.info("Standard Value Iteration finished in %.6f seconds and %s iterations.", self.runtime_seconds, self.iterations)
        return self.V

    def get_policy(self) -> Dict[State, str]:
        policy = {}
        for state in self.env.states:
            if self.env.is_terminal(state):
                policy[state] = "G"
            else:
                best_action = max(self.env.action_names, key=lambda action: self.q_value(state, action, self.V))
                policy[state] = best_action
        return policy


class InPlaceValueIterationAgent(ValueIterationAgent):
    """In-place value iteration using one value table."""

    def run(self) -> Dict[State, float]:
        start = time.perf_counter()
        self.logger.info("In-Place Value Iteration started.")
        self.logger.info("gamma=%s, theta=%s, max_iterations=%s", self.gamma, self.theta, self.max_iterations)

        for i in range(1, self.max_iterations + 1):
            delta = 0.0

            for state in self.env.states:
                if self.env.is_terminal(state):
                    self.V[state] = 0.0
                    continue

                old_value = self.V[state]
                action_values = [self.q_value(state, action, self.V) for action in self.env.action_names]
                self.V[state] = max(action_values)
                delta = max(delta, abs(old_value - self.V[state]))

            self.iterations = i
            self.logger.info("In-Place VI iteration=%s delta=%.8f", i, delta)

            if delta < self.theta:
                break

        self.runtime_seconds = time.perf_counter() - start
        self.logger.info("In-Place Value Iteration finished in %.6f seconds and %s iterations.", self.runtime_seconds, self.iterations)
        return self.V


class MonteCarloOffPolicyAgent:
    """
    Off-policy Monte Carlo control with weighted importance sampling.

    Behavior policy b(a|s): random policy.
    Target policy pi(a|s): greedy policy learned from Q(s,a).
    """

    def __init__(
        self,
        env: GridWorldEnvironment,
        gamma: float = 0.9,
        episodes: int = 5000,
        max_steps: int = 100,
        seed: int = 42,
        logger: logging.Logger | None = None,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.episodes = episodes
        self.max_steps = max_steps
        self.logger = logger or logging.getLogger(__name__)
        self.behavior_policy = RandomPolicy(env.action_names, seed=seed)
        self.target_policy = GreedyPolicy(env.action_names)
        self.rng = np.random.default_rng(seed)

        self.Q: Dict[Tuple[State, str], float] = {
            (state, action): 0.0 for state in env.states for action in env.action_names
        }
        self.C: Dict[Tuple[State, str], float] = {
            (state, action): 0.0 for state in env.states for action in env.action_names
        }
        for state in env.get_non_terminal_states():
            self.target_policy.set_action(state, env.action_names[0])

        self.runtime_seconds: float = 0.0

    def generate_episode(self) -> List[Tuple[State, str, float]]:
        """Generate one episode using behavior policy."""
        start_state = self.env.random_non_terminal_state()
        self.env.reset(start_state=start_state)
        episode: List[Tuple[State, str, float]] = []

        for _ in range(self.max_steps):
            state = self.env.current_state
            action = self.behavior_policy.sample_action(state)
            next_state = self.env.transition(state, action)
            reward = self.env.transition_reward(state, next_state)
            episode.append((state, action, reward))
            self.env.current_state = next_state

            if self.env.is_terminal(next_state):
                break

        return episode

    def run(self) -> Dict[State, float]:
        start = time.perf_counter()
        self.logger.info("Off-policy Monte Carlo started.")
        self.logger.info("gamma=%s, episodes=%s, max_steps=%s", self.gamma, self.episodes, self.max_steps)

        for episode_number in range(1, self.episodes + 1):
            episode = self.generate_episode()
            G = 0.0
            W = 1.0

            for state, action, reward in reversed(episode):
                G = self.gamma * G + reward
                key = (state, action)

                self.C[key] += W
                if self.C[key] != 0:
                    self.Q[key] += (W / self.C[key]) * (G - self.Q[key])

                best_action = max(self.env.action_names, key=lambda a: self.Q[(state, a)])
                self.target_policy.set_action(state, best_action)

                if action != self.target_policy.get_action(state):
                    break

                behavior_probability = self.behavior_policy.action_prob(state, action)
                if behavior_probability == 0:
                    break
                W = W / behavior_probability

            if episode_number % max(1, self.episodes // 10) == 0:
                self.logger.info("MC progress: episode %s/%s", episode_number, self.episodes)

        self.runtime_seconds = time.perf_counter() - start
        self.logger.info("Off-policy Monte Carlo finished in %.6f seconds.", self.runtime_seconds)
        return self.get_value_function()

    def get_value_function(self) -> Dict[State, float]:
        V: Dict[State, float] = {}
        for state in self.env.states:
            if self.env.is_terminal(state):
                V[state] = 0.0
            else:
                action = self.target_policy.get_action(state)
                V[state] = self.Q[(state, action)]
        return V

    def get_policy(self) -> Dict[State, str]:
        policy = {}
        for state in self.env.states:
            if self.env.is_terminal(state):
                policy[state] = "G"
            else:
                policy[state] = self.target_policy.get_action(state)
        return policy
