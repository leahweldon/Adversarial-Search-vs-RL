from __future__ import annotations

from src.agents.minimax_agent import MinimaxAgent
from src.evaluation.game_runner import play_game
from src.games.tictactoe import TicTacToe
from src.utils.io_utils import save_dict_rows_to_csv


def run_matchup(agent_x, agent_o, games: int) -> dict:
    wins_x = 0
    wins_o = 0
    draws = 0

    for _ in range(games):
        game = TicTacToe()
        result = play_game(game, agent_x, agent_o, render=False)

        if result == 1:
            wins_x += 1
        elif result == -1:
            wins_o += 1
        else:
            draws += 1

    return {
        "agent_x": agent_x.name,
        "agent_o": agent_o.name,
        "games": games,
        "wins_x": wins_x,
        "wins_o": wins_o,
        "draws": draws,
    }


def main() -> None:
    minimax_agent = MinimaxAgent(name="minimax", use_alpha_beta=False)
    alphabeta_agent = MinimaxAgent(name="alpha_beta", use_alpha_beta=True)

    rows = [
        run_matchup(minimax_agent, alphabeta_agent, 50),
        run_matchup(alphabeta_agent, minimax_agent, 50),
    ]

    for row in rows:
        print("\nMatchup")
        for key, value in row.items():
            print(f"{key}: {value}")

    save_dict_rows_to_csv(rows, "results/search/tictactoe/head_to_head.csv")


if __name__ == "__main__":
    main()