from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.agents.base_agent import BaseAgent
from src.evaluation.matchup_runner import run_matchup
from src.games.connect4 import Connect4
from src.games.tictactoe import TicTacToe

GameFactory = Callable[[], TicTacToe | Connect4]


@dataclass
class RLMatchupSummary:
    game: str
    agent_name: str
    opponent_name: str
    games: int
    win_rate: float
    loss_rate: float
    draw_rate: float
    avg_move_time: float


def evaluate_rl_agent_in_matchup(
    game_factory: GameFactory,
    agent: BaseAgent,
    opponent: BaseAgent,
    games: int,
    alternate_starting_player: bool = True,
) -> RLMatchupSummary:
    stats = run_matchup(
        game_factory=game_factory,
        agent_a=agent,
        agent_b=opponent,
        games=games,
        alternate_starting_player=alternate_starting_player,
    )

    game_name = game_factory.__name__.lower()

    return RLMatchupSummary(
        game=game_name,
        agent_name=agent.name,
        opponent_name=opponent.name,
        games=games,
        win_rate=stats.player1_win_rate,
        loss_rate=stats.player2_win_rate,
        draw_rate=stats.draw_rate,
        avg_move_time=stats.player1_avg_move_time,
    )