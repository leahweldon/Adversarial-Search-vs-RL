from __future__ import annotations

from src.agents.default_opponent import DefaultOpponent
from src.agents.minimax_agent import MinimaxAgent
from src.evaluation.evaluate_search import evaluate_search_agent
from src.games.tictactoe import TicTacToe
from src.utils.io_utils import save_dict_rows_to_csv


def run_experiment(use_alpha_beta: bool) -> dict:
    agent_name = "alpha_beta" if use_alpha_beta else "minimax"

    agent = MinimaxAgent(
        name=agent_name,
        use_alpha_beta=use_alpha_beta,
        depth_limit=None,
    )
    opponent = DefaultOpponent(name="default_opponent", seed=42)

    stats = evaluate_search_agent(
        agent=agent,
        game_factory=TicTacToe,
        opponent=opponent,
        games=100,
        learner_player=1,
    )

    return {
        "algorithm": agent_name,
        "game": "tictactoe",
        "depth_limit": "full",
        "games": stats.total_games,
        "wins": stats.wins,
        "draws": stats.draws,
        "losses": stats.losses,
        "win_rate": stats.win_rate,
        "total_nodes_visited": stats.total_nodes_visited,
        "total_cutoffs": stats.total_cutoffs,
        "max_depth_seen": stats.max_depth_seen,
        "average_move_time_seconds": stats.average_move_time,
    }


def main() -> None:
    rows = [
        run_experiment(use_alpha_beta=False),
        run_experiment(use_alpha_beta=True),
    ]

    for row in rows:
        print("\nResult")
        for key, value in row.items():
            print(f"{key}: {value}")

    save_dict_rows_to_csv(rows, "results/search/tictactoe/minimax_vs_alphabeta.csv")


if __name__ == "__main__":
    main()