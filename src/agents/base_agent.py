from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """
    Small shared interface for all agents.
    
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def select_action(self, game: Any) -> int:
        """
        Return the chosen action for the current game state.
        """
        raise NotImplementedError