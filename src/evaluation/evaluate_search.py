from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List

from src.agents.base_agent import BaseAgent
from src.agents.minimax_agent import MinimaxAgent
from src.games.connect4 import Connect4
from src.games.tictactoe import TicTacToe

GameFactory = Callable[[], TicTacToe | Connect4]


@dataclass
class SearchEvaluationStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_nodes_visited: int = 0
    total_cutoffs: int = 0
    max_depth_seen: int = 0
    move_times: List[float] | None = None

    def __post_init__(self) -> None:
        if self.move_times is None:
            self.move_times = []

    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_games if self.total_games else 0.0

    @property
    def average_move_time(self) -> float:
        return sum(self.move_times) / len(self.move_times) if self.move_times else 0.0


def evaluate_search_agent(
    agent: MinimaxAgent,
    game_factory: GameFactory,
    opponent: BaseAgent,
    games: int,
    learner_player: int = 1,
) -> SearchEvaluationStats:
    stats = SearchEvaluationStats()

    for _ in range(games):
        game = game_factory()

        while not game.is_terminal():
            if game.current_player == learner_player:
                start_time = time.perf_counter()
                action = agent.select_action(game)
                elapsed = time.perf_counter() - start_time

                stats.move_times.append(elapsed)
                stats.total_nodes_visited += agent.last_search_stats.nodes_visited
                stats.total_cutoffs += agent.last_search_stats.cutoffs
                stats.max_depth_seen = max(stats.max_depth_seen, agent.last_search_stats.max_depth_reached)
            else:
                action = opponent.select_action(game)

            game.apply_action(action)

        result = game.result()
        if result == learner_player:
            stats.wins += 1
        elif result == 0:
            stats.draws += 1
        else:
            stats.losses += 1

    return stats