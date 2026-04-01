from __future__ import annotations

from src.agents.q_learning_agent import QLearningAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.evaluate_q_learning import evaluate_q_agent, evaluate_q_agent_by_seat
from src.games.connect4 import Connect4
from src.training.train_q_learning import train_q_learning_agent_mixed_seats
from src.utils.io_utils import save_dict_rows_to_csv
from src.utils.plotting import plot_training_curve


def main() -> None:
    agent = QLearningAgent(
        name="q_learning_connect4",
        learning_rate=0.15,
        discount_factor=0.99,
        epsilon=1.0,
        seed=42,
    )

    opponent = RandomAgent(name="random_opponent", seed=42)

    episodes = 50000

    train_stats = train_q_learning_agent_mixed_seats(
    agent=agent,
    game_factory=Connect4,
    opponent=opponent,
    episodes=episodes,
    epsilon_decay=0.99985,
    min_epsilon=0.05,
    use_reward_shaping=True,
    )

    eval_stats = evaluate_q_agent(
        agent=agent,
        game_factory=Connect4,
        opponent=opponent,
        games=500,
        learner_player=1,
    )

    p1_stats, p2_stats = evaluate_q_agent_by_seat(
        agent=agent,
        game_factory=Connect4,
        opponent=opponent,
        games=250,
    )

    agent.save("results/models/connect4_q_learning_seed42.json")
    
    print("\nTraining summary")
    print(f"Wins:   {train_stats.wins}")
    print(f"Draws:  {train_stats.draws}")
    print(f"Losses: {train_stats.losses}")
    print(f"Stored states: {agent.policy_size()}")

    print("\nEvaluation summary (as P1)")
    print(f"Wins:   {eval_stats.wins}")
    print(f"Draws:  {eval_stats.draws}")
    print(f"Losses: {eval_stats.losses}")
    print(f"Win rate: {eval_stats.win_rate:.3f}")

    print("\nEvaluation by seat")
    print(f"  As P1 - Wins: {p1_stats.wins}  Draws: {p1_stats.draws}  Losses: {p1_stats.losses}  Win rate: {p1_stats.win_rate:.3f}")
    print(f"  As P2 - Wins: {p2_stats.wins}  Draws: {p2_stats.draws}  Losses: {p2_stats.losses}  Win rate: {p2_stats.win_rate:.3f}")

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
        "results/q_learning/connect4/training_log.csv",
    )

    summary_rows = [
        {
            "algorithm": "tabular_q_learning",
            "game": "connect4",
            "opponent": "random",
            "training_episodes": episodes,
            "eval_games": 500,
            "eval_wins": eval_stats.wins,
            "eval_draws": eval_stats.draws,
            "eval_losses": eval_stats.losses,
            "eval_win_rate": eval_stats.win_rate,
            "policy_size": agent.policy_size(),
            "p1_win_rate": p1_stats.win_rate,
            "p2_win_rate": p2_stats.win_rate,
        }
    ]

    save_dict_rows_to_csv(
        summary_rows,
        "results/q_learning/connect4/evaluation_summary.csv",
    )

    plot_training_curve(
        values=train_stats.episode_rewards,
        title="Q-Learning Training Rewards for Connect 4",
        y_label="Episode Reward",
        output_path="results/q_learning/connect4/training_rewards.png",
        smoothing_window=500,
    )

    plot_training_curve(
        values=train_stats.epsilon_history,
        title="Q-Learning Epsilon Decay for Connect 4",
        y_label="Epsilon",
        output_path="results/q_learning/connect4/epsilon_decay.png",
    )

    plot_training_curve(
        values=train_stats.outcome_history,
        title="Q-Learning Episode Outcomes for Connect 4",
        y_label="Outcome (-1 loss, 0 draw, 1 win)",
        output_path="results/q_learning/connect4/outcomes.png",
        smoothing_window=500,
    )


if __name__ == "__main__":
    main()