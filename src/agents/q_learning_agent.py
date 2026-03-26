from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.games.connect4 import Connect4
from src.games.tictactoe import TicTacToe

GameType = TicTacToe | Connect4
StateKey = tuple[int, ...]


class QLearningAgent(BaseAgent):
    """
    Tabular Q-learning agent

    Q-values are stored as:
        Q[state_key][action] = value

    Agent supports both training-time epsilon-greedy behaviour and
    evaluation-time greedy action selection
    """

    def __init__(
        self,
        name: str = "q_learning",
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name=name)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.rng = random.Random(seed)

        self.q_table: DefaultDict[StateKey, Dict[int, float]] = defaultdict(dict)

    def select_action(self, game: GameType, training: bool = False) -> int:
        """
        Select action using epsilon-greedy exploration during training,
        or greedy selection during evaluation
        """
        legal_actions = game.available_actions()
        if not legal_actions:
            raise ValueError("No legal actions available.")

        state_key = game.get_state_key()
        self._ensure_state_actions_exist(state_key, legal_actions)

        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)

        return self._best_action(state_key, legal_actions)

    def update(
        self,
        state_key: StateKey,
        action: int,
        reward: float,
        next_state_key: Optional[StateKey],
        next_legal_actions: Optional[List[int]],
        done: bool,
    ) -> None:
        """
        Standard Q-learning update:

        Q(s, a) <- Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0

        current_q = self.q_table[state_key][action]

        if done or next_state_key is None or not next_legal_actions:
            target = reward
        else:
            self._ensure_state_actions_exist(next_state_key, next_legal_actions)
            best_next_q = max(self.q_table[next_state_key][a] for a in next_legal_actions)
            target = reward + self.discount_factor * best_next_q

        self.q_table[state_key][action] = current_q + self.learning_rate * (target - current_q)

    def get_q_value(self, state_key: StateKey, action: int) -> float:
        return self.q_table[state_key].get(action, 0.0)

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def decay_epsilon(self, decay_rate: float, min_epsilon: float) -> None:
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def save(self, filepath: str) -> None:
        """
        Save Q-table and agent metadata to disk as JSON
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        serialisable_q_table = {}
        for state_key, action_values in self.q_table.items():
            serialisable_q_table[str(state_key)] = {
                str(action): value for action, value in action_values.items()
            }

        payload = {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "q_table": serialisable_q_table,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load(self, filepath: str) -> None:
        """
        Load Q-table and agent metadata from disk
        """
        path = Path(filepath)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self.name = payload["name"]
        self.learning_rate = payload["learning_rate"]
        self.discount_factor = payload["discount_factor"]
        self.epsilon = payload["epsilon"]

        self.q_table.clear()

        for state_str, action_values in payload["q_table"].items():
            state_key = tuple(
                int(x.strip())
                for x in state_str.strip("()").split(",")
                if x.strip()
            )
            self.q_table[state_key] = {
                int(action): float(value)
                for action, value in action_values.items()
            }

    def _ensure_state_actions_exist(self, state_key: StateKey, legal_actions: List[int]) -> None:
        for action in legal_actions:
            if action not in self.q_table[state_key]:
                self.q_table[state_key][action] = 0.0

    def _best_action(self, state_key: StateKey, legal_actions: List[int]) -> int:
        """
        Return greedy action

        If multiple actions tie, choose randomly among best ones
        """
        best_q = max(self.q_table[state_key][action] for action in legal_actions)
        best_actions = [
            action
            for action in legal_actions
            if self.q_table[state_key][action] == best_q
        ]
        return self.rng.choice(best_actions)

    def policy_size(self) -> int:
        """
        Return number of distinct states with at least one stored Q-value
        """
        return len(self.q_table)