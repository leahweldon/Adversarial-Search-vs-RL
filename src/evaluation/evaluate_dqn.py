from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

from src.agents.base_agent import BaseAgent
from src.agents.dqn_agent import DQNAgent
from src.games.connect4 import Connect4
from src.games.tictactoe import TicTacToe

GameFactory = Callable[[], TicTacToe | Connect4]


@dataclass
class DQNEvaluationStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_games if self.total_games else 0.0


def evaluate_dqn_agent(
    agent: DQNAgent,
    game_factory: GameFactory,
    opponent: BaseAgent,
    games: int,
    learner_player: int = 1,
) -> DQNEvaluationStats:
    stats = DQNEvaluationStats()
    original_epsilon = agent.epsilon
    agent.set_epsilon(0.0)

    try:
        for _ in range(games):
            game = game_factory()

            while not game.is_terminal():
                if game.current_player == learner_player:
                    action = agent.select_action(game, training=False)
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
    finally:
        agent.set_epsilon(original_epsilon)

    return stats


def evaluate_dqn_agent_by_seat(
    agent: DQNAgent,
    game_factory: GameFactory,
    opponent: BaseAgent,
    games: int,
) -> Tuple[DQNEvaluationStats, DQNEvaluationStats]:
    """evaluate separately as P1 and P2, returns (p1_stats, p2_stats)"""
    p1_stats = evaluate_dqn_agent(agent, game_factory, opponent, games, learner_player=1)
    p2_stats = evaluate_dqn_agent(agent, game_factory, opponent, games, learner_player=-1)
    return p1_stats, p2_stats