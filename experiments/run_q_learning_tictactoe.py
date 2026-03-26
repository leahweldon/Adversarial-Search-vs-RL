from __future__ import annotations

from src.agents.default_opponent import DefaultOpponent
from src.agents.q_learning_agent import QLearningAgent
from src.evaluation.evaluate_q_learning import evaluate_q_agent
from src.games.tictactoe import TicTacToe
from src.training.train_q_learning import train_q_learning_agent_mixed_seats
from src.utils.io_utils import save_dict_rows_to_csv
from src.utils.plotting import plot_training_curve


def main() -> None:
    agent = QLearningAgent(
        name="q_learning_ttt",
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon=1.0,
        seed=42,
    )

    opponent = DefaultOpponent(name="default_opponent", seed=42)

    episodes = 10000

    train_stats = train_q_learning_agent_mixed_seats(
        agent=agent,
        game_factory=TicTacToe,
        opponent=opponent,
        episodes=episodes,
        epsilon_decay=0.9993,
        min_epsilon=0.05,
    )

    eval_stats = evaluate_q_agent(
        agent=agent,
        game_factory=TicTacToe,
        opponent=opponent,
        games=500,
        learner_player=1,
    )

    agent.save("results/models/tictactoe_q_learning_seed42.json")

    print("\nTraining summary")
    print(f"Wins:   {train_stats.wins}")
    print(f"Draws:  {train_stats.draws}")
    print(f"Losses: {train_stats.losses}")
    print(f"Stored states: {agent.policy_size()}")

    print("\nEvaluation summary")
    print(f"Wins:   {eval_stats.wins}")
    print(f"Draws:  {eval_stats.draws}")
    print(f"Losses: {eval_stats.losses}")
    print(f"Win rate: {eval_stats.win_rate:.3f}")

    rows = []
    for episode_index, (reward, epsilon, outcome) in enumerate(
        zip(
            train_stats.episode_rewards,
            train_stats.epsilon_history,
            train_stats.outcome_history,
        ),
        start=1,
    ):
        rows.append(
            {
                "episode": episode_index,
                "reward": reward,
                "epsilon": epsilon,
                "outcome": outcome,
            }
        )

    save_dict_rows_to_csv(
        rows,
        "results/q_learning/tictactoe/training_log.csv",
    )

    summary_rows = [
        {
            "algorithm": "tabular_q_learning",
            "game": "tictactoe",
            "training_episodes": episodes,
            "eval_games": 500,
            "eval_wins": eval_stats.wins,
            "eval_draws": eval_stats.draws,
            "eval_losses": eval_stats.losses,
            "eval_win_rate": eval_stats.win_rate,
            "policy_size": agent.policy_size(),
        }
    ]

    save_dict_rows_to_csv(
        summary_rows,
        "results/q_learning/tictactoe/evaluation_summary.csv",
    )

    plot_training_curve(
        values=train_stats.episode_rewards,
        title="Q-Learning Training Rewards for Tic Tac Toe",
        y_label="Episode Reward",
        output_path="results/q_learning/tictactoe/training_rewards.png",
        smoothing_window=200,
    )

    plot_training_curve(
        values=train_stats.epsilon_history,
        title="Q-Learning Epsilon Decay for Tic Tac Toe",
        y_label="Epsilon",
        output_path="results/q_learning/tictactoe/epsilon_decay.png",
    )

    plot_training_curve(
        values=train_stats.outcome_history,
        title="Q-Learning Episode Outcomes for Tic Tac Toe",
        y_label="Outcome (-1 loss, 0 draw, 1 win)",
        output_path="results/q_learning/tictactoe/outcomes.png",
        smoothing_window=200,
    )


if __name__ == "__main__":
    main()