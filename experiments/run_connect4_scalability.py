from __future__ import annotations

import time

from src.agents.minimax_agent import MinimaxAgent
from src.games.connect4 import Connect4
from src.utils.io_utils import save_dict_rows_to_csv


def run_single_search(use_alpha_beta: bool, depth_limit: int | None) -> dict:
    agent_name = "alpha_beta" if use_alpha_beta else "minimax"
    depth_label = "full" if depth_limit is None else depth_limit

    game = Connect4()
    agent = MinimaxAgent(
        name=agent_name,
        use_alpha_beta=use_alpha_beta,
        depth_limit=depth_limit,
    )

    start_time = time.perf_counter()
    action = agent.select_action(game)
    elapsed = time.perf_counter() - start_time

    return {
        "algorithm": agent_name,
        "depth_limit": depth_label,
        "selected_action": action,
        "elapsed_seconds": elapsed,
        "nodes_visited": agent.last_search_stats.nodes_visited,
        "cutoffs": agent.last_search_stats.cutoffs,
        "max_depth_reached": agent.last_search_stats.max_depth_reached,
    }


def main() -> None:
    rows = []

    rows.append(run_single_search(use_alpha_beta=False, depth_limit=3))
    rows.append(run_single_search(use_alpha_beta=True, depth_limit=3))
    rows.append(run_single_search(use_alpha_beta=False, depth_limit=4))
    rows.append(run_single_search(use_alpha_beta=True, depth_limit=4))
    rows.append(run_single_search(use_alpha_beta=True, depth_limit=5))

    for row in rows:
        print("\nSearch result")
        for key, value in row.items():
            print(f"{key}: {value}")

    save_dict_rows_to_csv(rows, "results/search/connect4/scalability.csv")


if __name__ == "__main__":
    main()