from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from src.agents.base_agent import BaseAgent
from src.agents.minimax_agent import MinimaxAgent
from src.games.connect4 import Connect4
from src.games.tictactoe import TicTacToe

GameFactory = Callable[[], TicTacToe | Connect4]


@dataclass
class MatchupStats:
    games: int = 0
    player1_wins: int = 0
    player2_wins: int = 0
    draws: int = 0
    player1_move_times: List[float] | None = None
    player2_move_times: List[float] | None = None
    player1_nodes: int = 0
    player2_nodes: int = 0
    player1_cutoffs: int = 0
    player2_cutoffs: int = 0
    player1_max_depth: int = 0
    player2_max_depth: int = 0

    def __post_init__(self) -> None:
        if self.player1_move_times is None:
            self.player1_move_times = []
        if self.player2_move_times is None:
            self.player2_move_times = []

    @property
    def player1_win_rate(self) -> float:
        return self.player1_wins / self.games if self.games else 0.0

    @property
    def player2_win_rate(self) -> float:
        return self.player2_wins / self.games if self.games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games else 0.0

    @property
    def player1_avg_move_time(self) -> float:
        return sum(self.player1_move_times) / len(self.player1_move_times) if self.player1_move_times else 0.0

    @property
    def player2_avg_move_time(self) -> float:
        return sum(self.player2_move_times) / len(self.player2_move_times) if self.player2_move_times else 0.0


def _record_search_stats(agent: BaseAgent, stats: MatchupStats, is_player1: bool, elapsed: float) -> None:
    if is_player1:
        stats.player1_move_times.append(elapsed)
    else:
        stats.player2_move_times.append(elapsed)

    if isinstance(agent, MinimaxAgent):
        if is_player1:
            stats.player1_nodes += agent.last_search_stats.nodes_visited
            stats.player1_cutoffs += agent.last_search_stats.cutoffs
            stats.player1_max_depth = max(stats.player1_max_depth, agent.last_search_stats.max_depth_reached)
        else:
            stats.player2_nodes += agent.last_search_stats.nodes_visited
            stats.player2_cutoffs += agent.last_search_stats.cutoffs
            stats.player2_max_depth = max(stats.player2_max_depth, agent.last_search_stats.max_depth_reached)


def play_single_game(
    game_factory: GameFactory,
    agent_player1: BaseAgent,
    agent_player2: BaseAgent,
) -> tuple[int, MatchupStats]:
    game = game_factory()
    stats = MatchupStats()

    while not game.is_terminal():
        if game.current_player == 1:
            agent = agent_player1
            is_player1 = True
        else:
            agent = agent_player2
            is_player1 = False

        start = time.perf_counter()
        action = agent.select_action(game)
        elapsed = time.perf_counter() - start

        _record_search_stats(agent, stats, is_player1, elapsed)
        game.apply_action(action)

    result = game.result()
    stats.games = 1

    if result == 1:
        stats.player1_wins = 1
    elif result == -1:
        stats.player2_wins = 1
    else:
        stats.draws = 1

    return result, stats


def run_matchup(
    game_factory: GameFactory,
    agent_a: BaseAgent,
    agent_b: BaseAgent,
    games: int,
    alternate_starting_player: bool = True,
) -> MatchupStats:
    """
    Runs matchup and reports results from perspective of the seat assignment,
    not the algorithm names. If alternating is enabled, agent_a and agent_b swap seats
    every other game.
    """
    aggregate = MatchupStats()

    for game_index in range(games):
        if alternate_starting_player and game_index % 2 == 1:
            _, single = play_single_game(game_factory, agent_b, agent_a)

            aggregate.games += 1
            aggregate.player1_wins += single.player2_wins
            aggregate.player2_wins += single.player1_wins
            aggregate.draws += single.draws

            aggregate.player1_move_times.extend(single.player2_move_times)
            aggregate.player2_move_times.extend(single.player1_move_times)

            aggregate.player1_nodes += single.player2_nodes
            aggregate.player2_nodes += single.player1_nodes
            aggregate.player1_cutoffs += single.player2_cutoffs
            aggregate.player2_cutoffs += single.player1_cutoffs
            aggregate.player1_max_depth = max(aggregate.player1_max_depth, single.player2_max_depth)
            aggregate.player2_max_depth = max(aggregate.player2_max_depth, single.player1_max_depth)
        else:
            _, single = play_single_game(game_factory, agent_a, agent_b)

            aggregate.games += 1
            aggregate.player1_wins += single.player1_wins
            aggregate.player2_wins += single.player2_wins
            aggregate.draws += single.draws

            aggregate.player1_move_times.extend(single.player1_move_times)
            aggregate.player2_move_times.extend(single.player2_move_times)

            aggregate.player1_nodes += single.player1_nodes
            aggregate.player2_nodes += single.player2_nodes
            aggregate.player1_cutoffs += single.player1_cutoffs
            aggregate.player2_cutoffs += single.player2_cutoffs
            aggregate.player1_max_depth = max(aggregate.player1_max_depth, single.player1_max_depth)
            aggregate.player2_max_depth = max(aggregate.player2_max_depth, single.player2_max_depth)

    return aggregate