from __future__ import annotations

from src.agents.default_opponent import DefaultOpponent
from src.agents.dqn_agent import DQNAgent
from src.evaluation.evaluate_dqn import evaluate_dqn_agent, evaluate_dqn_agent_by_seat
from src.games.tictactoe import TicTacToe
from src.training.train_dqn import train_dqn_agent_mixed_seats
from src.utils.io_utils import save_dict_rows_to_csv
from src.utils.plotting import plot_training_curve


def main() -> None:
    agent = DQNAgent(
        name="dqn_ttt",
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
        seed=42,
    )

    opponent = DefaultOpponent(name="default_opponent", seed=42)

    episodes = 50000

    train_stats = train_dqn_agent_mixed_seats(
    agent=agent,
    game_factory=TicTacToe,
    opponent=opponent,
    episodes=episodes,
    )

    eval_stats = evaluate_dqn_agent(
        agent=agent,
        game_factory=TicTacToe,
        opponent=opponent,
        games=500,
        learner_player=1,
    )

    p1_stats, p2_stats = evaluate_dqn_agent_by_seat(
        agent=agent,
        game_factory=TicTacToe,
        opponent=opponent,
        games=250,
    )

    agent.save("results/models/tictactoe_dqn_seed42.pt")

    print("\nTraining summary")
    print(f"Wins:   {train_stats.wins}")
    print(f"Draws:  {train_stats.draws}")
    print(f"Losses: {train_stats.losses}")

    print("\nEvaluation summary (as P1)")
    print(f"Wins:   {eval_stats.wins}")
    print(f"Draws:  {eval_stats.draws}")
    print(f"Losses: {eval_stats.losses}")
    print(f"Win rate: {eval_stats.win_rate:.3f}")

    print("\nEvaluation by seat")
    print(f"  As P1 - Wins: {p1_stats.wins}  Draws: {p1_stats.draws}  Losses: {p1_stats.losses}  Win rate: {p1_stats.win_rate:.3f}")
    print(f"  As P2 - Wins: {p2_stats.wins}  Draws: {p2_stats.draws}  Losses: {p2_stats.losses}  Win rate: {p2_stats.win_rate:.3f}")

    rows = []
    for episode_index, (reward, loss, epsilon, outcome) in enumerate(
        zip(
            train_stats.episode_rewards,
            train_stats.losses_history,
            train_stats.epsilon_history,
            train_stats.outcome_history,
        ),
        start=1,
    ):
        rows.append(
            {
                "episode": episode_index,
                "reward": reward,
                "mean_loss": loss,
                "epsilon": epsilon,
                "outcome": outcome,
            }
        )

    save_dict_rows_to_csv(rows, "results/dqn/tictactoe/training_log.csv")

    summary_rows = [
        {
            "algorithm": "dqn",
            "game": "tictactoe",
            "training_episodes": episodes,
            "eval_games": 500,
            "eval_wins": eval_stats.wins,
            "eval_draws": eval_stats.draws,
            "eval_losses": eval_stats.losses,
            "eval_win_rate": eval_stats.win_rate,
            "p1_win_rate": p1_stats.win_rate,
            "p2_win_rate": p2_stats.win_rate,
        }
    ]

    save_dict_rows_to_csv(summary_rows, "results/dqn/tictactoe/evaluation_summary.csv")

    plot_training_curve(
        values=train_stats.episode_rewards,
        title="DQN Training Rewards for Tic Tac Toe",
        y_label="Episode Reward",
        output_path="results/dqn/tictactoe/training_rewards.png",
        smoothing_window=200,
    )

    plot_training_curve(
        values=train_stats.losses_history,
        title="DQN Training Loss for Tic Tac Toe",
        y_label="Mean Episode Loss",
        output_path="results/dqn/tictactoe/training_loss.png",
        smoothing_window=200,
    )

    plot_training_curve(
        values=train_stats.epsilon_history,
        title="DQN Epsilon Decay for Tic Tac Toe",
        y_label="Epsilon",
        output_path="results/dqn/tictactoe/epsilon_decay.png",
    )

    plot_training_curve(
        values=train_stats.outcome_history,
        title="DQN Episode Outcomes for Tic Tac Toe",
        y_label="Outcome (-1 loss, 0 draw, 1 win)",
        output_path="results/dqn/tictactoe/outcomes.png",
        smoothing_window=200,
    )


if __name__ == "__main__":
    main()