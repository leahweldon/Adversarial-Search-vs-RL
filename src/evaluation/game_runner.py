from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseAgent


def play_game(game: Any, agent_x: BaseAgent, agent_o: BaseAgent, render: bool = False) -> int:
    """
    Play one full game and return result from Player 1's perspective

    Returns:
        1  -> Player 1 wins
        -1 -> Player -1 wins
        0  -> draw
    """
    if render:
        print("\nInitial state:")
        game.render()

    while not game.is_terminal():
        current_agent = agent_x if game.current_player == 1 else agent_o
        action = current_agent.select_action(game)
        game.apply_action(action)

        if render:
            print(f"\n{current_agent.name} played action {action}")
            game.render()

    return game.result()