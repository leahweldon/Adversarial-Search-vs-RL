from __future__ import annotations

import random
from typing import Optional

from src.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Simple random baseline agent"""

    def __init__(self, name: str = "random", seed: Optional[int] = None) -> None:
        super().__init__(name=name)
        self.rng = random.Random(seed)

    def select_action(self, game) -> int:
        legal_actions = game.available_actions()
        if not legal_actions:
            raise ValueError("No legal actions available.")
        return self.rng.choice(legal_actions)