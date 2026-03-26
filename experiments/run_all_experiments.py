from __future__ import annotations

from pathlib import Path

from src.agents.default_opponent import DefaultOpponent
from src.agents.dqn_agent import DQNAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.aggregate_results import aggregate_rows
from src.evaluation.evaluate_rl_matchups import evaluate_rl_agent_in_matchup
from src.evaluation.matchup_runner import run_matchup
from src.games.connect4 import Connect4
from src.games.tictactoe import TicTacToe
from src.training.train_dqn import train_dqn_agent_mixed_seats
from src.training.train_q_learning import train_q_learning_agent_mixed_seats
from src.utils.experiment_config import (
    CONNECT4_RL_CONFIG,
    CONNECT4_SEARCH_CONFIG,
    TICTACTOE_RL_CONFIG,
    TICTACTOE_SEARCH_CONFIG,
)
from src.utils.io_utils import save_dict_rows_to_csv


def run_search_experiments() -> tuple[list[dict], list[dict]]:
    raw_rows: list[dict] = []

    # Tic Tac Toe search vs default opponent
    for use_alpha_beta in [False, True]:
        agent = MinimaxAgent(
            name="alpha_beta" if use_alpha_beta else "minimax",
            use_alpha_beta=use_alpha_beta,
            depth_limit=None,
        )
        opponent = DefaultOpponent(name="default_opponent", seed=42)

        stats = run_matchup(
            game_factory=TicTacToe,
            agent_a=agent,
            agent_b=opponent,
            games=TICTACTOE_SEARCH_CONFIG.evaluation_games,
            alternate_starting_player=TICTACTOE_SEARCH_CONFIG.alternate_starting_player,
        )

        raw_rows.append(
            {
                "category": "search_vs_default",
                "game": "tictactoe",
                "algorithm": agent.name,
                "depth_limit": "full",
                "player1_win_rate": stats.player1_win_rate,
                "player2_win_rate": stats.player2_win_rate,
                "draw_rate": stats.draw_rate,
                "player1_avg_move_time": stats.player1_avg_move_time,
                "player1_nodes": stats.player1_nodes,
                "player1_cutoffs": stats.player1_cutoffs,
                "player1_max_depth": stats.player1_max_depth,
            }
        )

    # Connect 4 search vs default opponent
    for use_alpha_beta, depth_limit in [(False, 3), (True, 3), (True, 4)]:
        agent = MinimaxAgent(
            name="alpha_beta" if use_alpha_beta else "minimax",
            use_alpha_beta=use_alpha_beta,
            depth_limit=depth_limit,
        )
        opponent = DefaultOpponent(name="default_opponent", seed=42)

        stats = run_matchup(
            game_factory=Connect4,
            agent_a=agent,
            agent_b=opponent,
            games=CONNECT4_SEARCH_CONFIG.evaluation_games,
            alternate_starting_player=CONNECT4_SEARCH_CONFIG.alternate_starting_player,
        )

        raw_rows.append(
            {
                "category": "search_vs_default",
                "game": "connect4",
                "algorithm": agent.name,
                "depth_limit": depth_limit,
                "player1_win_rate": stats.player1_win_rate,
                "player2_win_rate": stats.player2_win_rate,
                "draw_rate": stats.draw_rate,
                "player1_avg_move_time": stats.player1_avg_move_time,
                "player1_nodes": stats.player1_nodes,
                "player1_cutoffs": stats.player1_cutoffs,
                "player1_max_depth": stats.player1_max_depth,
            }
        )

    aggregated_rows = raw_rows[:]  # deterministic search runs, so raw = summary here
    return raw_rows, aggregated_rows


def run_rl_experiments() -> tuple[list[dict], list[dict]]:
    raw_rows: list[dict] = []

    # Tic Tac Toe Q-learning
    for seed in TICTACTOE_RL_CONFIG.seeds:
        agent = QLearningAgent(
            name="tabular_q_learning",
            learning_rate=0.2,
            discount_factor=0.95,
            epsilon=1.0,
            seed=seed,
        )
        opponent = DefaultOpponent(name="default_opponent", seed=seed)

        train_q_learning_agent_mixed_seats(
            agent=agent,
            game_factory=TicTacToe,
            opponent=opponent,
            episodes=TICTACTOE_RL_CONFIG.training_episodes,
            epsilon_decay=0.9993,
            min_epsilon=0.05,
        )

        eval_summary = evaluate_rl_agent_in_matchup(
            game_factory=TicTacToe,
            agent=agent,
            opponent=opponent,
            games=TICTACTOE_RL_CONFIG.evaluation_games,
            alternate_starting_player=True,
        )

        agent.save(f"results/models/tictactoe_q_learning_seed{seed}.json")

        raw_rows.append(
            {
                "category": "rl_vs_opponent",
                "game": "tictactoe",
                "algorithm": "tabular_q_learning",
                "opponent": "default_opponent",
                "seed": seed,
                "win_rate": eval_summary.win_rate,
                "wins": eval_summary.win_rate * eval_summary.games,
                "draws": eval_summary.draw_rate * eval_summary.games,
                "losses": eval_summary.loss_rate * eval_summary.games,
                "policy_size": agent.policy_size(),
            }
        )

    # Tic Tac Toe DQN
    for seed in TICTACTOE_RL_CONFIG.seeds:
        agent = DQNAgent(
            name="dqn",
            state_size=10,
            action_size=9,
            learning_rate=5e-4,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.9997,
            min_epsilon=0.05,
            batch_size=64,
            replay_buffer_size=20000,
            target_update_frequency=100,
            seed=seed,
        )
        opponent = DefaultOpponent(name="default_opponent", seed=seed)

        train_dqn_agent_mixed_seats(
            agent=agent,
            game_factory=TicTacToe,
            opponent=opponent,
            episodes=15000,
        )

        eval_summary = evaluate_rl_agent_in_matchup(
            game_factory=TicTacToe,
            agent=agent,
            opponent=opponent,
            games=TICTACTOE_RL_CONFIG.evaluation_games,
            alternate_starting_player=True,
        )


        agent.save(f"results/models/tictactoe_dqn_seed{seed}.pt")

        raw_rows.append(
            {
                "category": "rl_vs_opponent",
                "game": "tictactoe",
                "algorithm": "dqn",
                "opponent": "default_opponent",
                "seed": seed,
                "win_rate": eval_summary.win_rate,
                "wins": eval_summary.win_rate * eval_summary.games,
                "draws": eval_summary.draw_rate * eval_summary.games,
                "losses": eval_summary.loss_rate * eval_summary.games,
                "policy_size": 0,
            }
        )

    # Connect 4 Q-learning
    for seed in CONNECT4_RL_CONFIG.seeds:
        agent = QLearningAgent(
            name="tabular_q_learning",
            learning_rate=0.15,
            discount_factor=0.99,
            epsilon=1.0,
            seed=seed,
        )
        opponent = RandomAgent(name="random_opponent", seed=seed)

        train_q_learning_agent_mixed_seats(
            agent=agent,
            game_factory=Connect4,
            opponent=opponent,
            episodes=CONNECT4_RL_CONFIG.training_episodes,
            epsilon_decay=0.9997,
            min_epsilon=0.05,
        )

        eval_summary = evaluate_rl_agent_in_matchup(
            game_factory=Connect4,
            agent=agent,
            opponent=opponent,
            games=CONNECT4_RL_CONFIG.evaluation_games,
            alternate_starting_player=True,
        )

        agent.save(f"results/models/connect4_q_learning_seed{seed}.json")

        raw_rows.append(
            {
                "category": "rl_vs_opponent",
                "game": "connect4",
                "algorithm": "tabular_q_learning",
                "opponent": "random_opponent",
                "seed": seed,
                "win_rate": eval_summary.win_rate,
                "wins": eval_summary.win_rate * eval_summary.games,
                "draws": eval_summary.draw_rate * eval_summary.games,
                "losses": eval_summary.loss_rate * eval_summary.games,
                "policy_size": agent.policy_size(),
            }
        )

    # Connect 4 DQN
    for seed in CONNECT4_RL_CONFIG.seeds:
        agent = DQNAgent(
            name="dqn",
            state_size=43,
            action_size=7,
            learning_rate=5e-4,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.9998,
            min_epsilon=0.05,
            batch_size=128,
            replay_buffer_size=50000,
            target_update_frequency=250,
            seed=seed,
        )
        opponent = RandomAgent(name="random_opponent", seed=seed)

        train_dqn_agent_mixed_seats(
            agent=agent,
            game_factory=Connect4,
            opponent=opponent,
            episodes=30000,
        )

        eval_summary = evaluate_rl_agent_in_matchup(
            game_factory=Connect4,
            agent=agent,
            opponent=opponent,
            games=CONNECT4_RL_CONFIG.evaluation_games,
            alternate_starting_player=True,
        )

        agent.save(f"results/models/connect4_dqn_seed{seed}.pt")

        raw_rows.append(
            {
                "category": "rl_vs_opponent",
                "game": "connect4",
                "algorithm": "dqn",
                "opponent": "random_opponent",
                "seed": seed,
                "win_rate": eval_summary.win_rate,
                "wins": eval_summary.win_rate * eval_summary.games,
                "draws": eval_summary.draw_rate * eval_summary.games,
                "losses": eval_summary.loss_rate * eval_summary.games,
                "policy_size": 0,
            }
        )

    aggregated_rows = aggregate_rows(
        rows=raw_rows,
        group_keys=["category", "game", "algorithm", "opponent"],
        metric_keys=["win_rate", "wins", "draws", "losses", "policy_size"],
    )
    return raw_rows, aggregated_rows


def build_master_summary(search_rows: list[dict], rl_summary_rows: list[dict]) -> list[dict]:
    summary_rows: list[dict] = []

    for row in search_rows:
        summary_rows.append(
            {
                "game": row["game"],
                "algorithm": row["algorithm"],
                "setting": row["category"],
                "opponent": "default_opponent",
                "depth_limit": row["depth_limit"],
                "win_rate": row["player1_win_rate"],
                "draw_rate": row["draw_rate"],
                "avg_move_time": row["player1_avg_move_time"],
                "notes": "Deterministic search evaluation",
            }
        )

    for row in rl_summary_rows:
        summary_rows.append(
            {
                "game": row["game"],
                "algorithm": row["algorithm"],
                "setting": row["category"],
                "opponent": row["opponent"],
                "depth_limit": "",
                "win_rate": row["win_rate_mean"],
                "draw_rate": row["draws_mean"] / (row["wins_mean"] + row["draws_mean"] + row["losses_mean"])
                if (row["wins_mean"] + row["draws_mean"] + row["losses_mean"]) > 0
                else 0.0,
                "avg_move_time": "",
                "notes": f"Mean over {row['runs']} seeded RL runs",
            }
        )

    return summary_rows


def main() -> None:
    output_dir = Path("results/final")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running search experiments...")
    search_raw, search_summary = run_search_experiments()

    print("Running RL experiments...")
    rl_raw, rl_summary = run_rl_experiments()

    master_summary = build_master_summary(search_summary, rl_summary)

    save_dict_rows_to_csv(search_raw, output_dir / "search_raw_results.csv")
    save_dict_rows_to_csv(search_summary, output_dir / "search_summary_results.csv")
    save_dict_rows_to_csv(rl_raw, output_dir / "rl_raw_results.csv")
    save_dict_rows_to_csv(rl_summary, output_dir / "rl_summary_results.csv")
    save_dict_rows_to_csv(master_summary, output_dir / "master_summary.csv")

    print("\nSaved final result files:")
    print("- results/final/search_raw_results.csv")
    print("- results/final/search_summary_results.csv")
    print("- results/final/rl_raw_results.csv")
    print("- results/final/rl_summary_results.csv")
    print("- results/final/master_summary.csv")


if __name__ == "__main__":
    main()