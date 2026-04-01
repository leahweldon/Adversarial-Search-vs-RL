from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """
    Small shared interface for all agents
    
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def select_action(self, game: Any) -> int:
        """
        Return chosen action for current game state
        """
        raise NotImplementedError