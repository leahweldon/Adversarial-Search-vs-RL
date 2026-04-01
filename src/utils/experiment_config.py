from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class SearchExperimentConfig:
    game_name: str
    evaluation_games: int
    alternate_starting_player: bool
    depth_limits: List[int]


@dataclass(frozen=True)
class RLExperimentConfig:
    game_name: str
    training_episodes: int
    evaluation_games: int
    alternate_starting_player: bool
    seeds: List[int]
    learner_players: List[int] = field(default_factory=lambda: [1, -1])

TICTACTOE_SEARCH_CONFIG = SearchExperimentConfig(
    game_name="tictactoe",
    evaluation_games=150,
    alternate_starting_player=True,
    depth_limits=[None],
)

CONNECT4_SEARCH_CONFIG = SearchExperimentConfig(
    game_name="connect4",
    evaluation_games=150,
    alternate_starting_player=True,
    depth_limits=[3, 4],
)

TICTACTOE_RL_CONFIG = RLExperimentConfig(
    game_name="tictactoe",
    training_episodes=20000,
    evaluation_games=1000,
    alternate_starting_player=True,
    seeds=[7, 21, 42],
)

CONNECT4_RL_CONFIG = RLExperimentConfig(
    game_name="connect4",
    training_episodes=80000,
    evaluation_games=1000,
    alternate_starting_player=True,
    seeds=[7, 21, 42],
)