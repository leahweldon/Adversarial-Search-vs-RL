from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from src.agents.base_agent import BaseAgent
from src.agents.q_learning_agent import QLearningAgent
from src.games.connect4 import Connect4
from src.games.tictactoe import TicTacToe

GameFactory = Callable[[], TicTacToe | Connect4]


@dataclass
class TrainingStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    episode_rewards: List[float] | None = None
    epsilon_history: List[float] | None = None
    outcome_history: List[int] | None = None

    def __post_init__(self) -> None:
        if self.episode_rewards is None:
            self.episode_rewards = []
        if self.epsilon_history is None:
            self.epsilon_history = []
        if self.outcome_history is None:
            self.outcome_history = []


def _reward_from_result(result: int, learner_player: int) -> float:
    if result == learner_player:
        return 1.0
    if result == 0:
        return 0.3
    return -1.0


def _connect4_shape_reward(game: Connect4, player: int) -> float:
    """small bonus for 3-in-a-row threats, penalty if opponent has one"""
    board = game.board
    rows, cols = game.rows, game.cols
    opponent = -player
    reward = 0.0
    windows = []
    for row in range(rows):
        for col in range(cols - 3):
            windows.append([board[row][col + i] for i in range(4)])
    for row in range(rows - 3):
        for col in range(cols):
            windows.append([board[row + i][col] for i in range(4)])
    for row in range(rows - 3):
        for col in range(cols - 3):
            windows.append([board[row + i][col + i] for i in range(4)])
    for row in range(3, rows):
        for col in range(cols - 3):
            windows.append([board[row - i][col + i] for i in range(4)])
    for window in windows:
        pc, oc, ec = window.count(player), window.count(opponent), window.count(0)
        if pc == 3 and ec == 1:
            reward += 0.05
        elif oc == 3 and ec == 1:
            reward -= 0.05
    return reward


def train_q_learning_agent(
    agent: QLearningAgent,
    game_factory: GameFactory,
    opponent: BaseAgent,
    episodes: int,
    learner_player: int = 1,
    epsilon_decay: float = 0.9995,
    min_epsilon: float = 0.05,
    use_reward_shaping: bool = False,
) -> TrainingStats:
    """Train learner against a fixed opponent """
    stats = TrainingStats()

    for _ in range(episodes):
        game = game_factory()
        total_episode_reward = 0.0

        while not game.is_terminal():
            if game.current_player != learner_player:
                opponent_action = opponent.select_action(game)
                game.apply_action(opponent_action)
                continue

            state_key = game.get_state_key()
            action = agent.select_action(game, training=True)
            game.apply_action(action)

            # shaping measured here - after learner's move, before opponent acts
            step_reward = _connect4_shape_reward(game, learner_player) if use_reward_shaping and isinstance(game, Connect4) else 0.0

            # If learner ends the game immediately, update with terminal reward
            if game.is_terminal():
                result = game.result()
                reward = _reward_from_result(result, learner_player)
                total_episode_reward += reward

                agent.update(
                    state_key=state_key,
                    action=action,
                    reward=reward,
                    next_state_key=None,
                    next_legal_actions=None,
                    done=True,
                )
                break

            # Opponent now responds
            opponent_action = opponent.select_action(game)
            game.apply_action(opponent_action)

            if game.is_terminal():
                result = game.result()
                reward = _reward_from_result(result, learner_player)
                total_episode_reward += reward

                agent.update(
                    state_key=state_key,
                    action=action,
                    reward=reward,
                    next_state_key=None,
                    next_legal_actions=None,
                    done=True,
                )
                break

            # Otherwise  game continues and it is the learner's turn again
            next_state_key = game.get_state_key()
            next_legal_actions = game.available_actions()

            total_episode_reward += step_reward

            agent.update(
                state_key=state_key,
                action=action,
                reward=step_reward,
                next_state_key=next_state_key,
                next_legal_actions=next_legal_actions,
                done=False,
            )

        final_result = game.result()
        if final_result == learner_player:
            stats.wins += 1
            stats.outcome_history.append(1)
        elif final_result == 0:
            stats.draws += 1
            stats.outcome_history.append(0)
        else:
            stats.losses += 1
            stats.outcome_history.append(-1)

        stats.episode_rewards.append(total_episode_reward)
        stats.epsilon_history.append(agent.epsilon)

        agent.decay_epsilon(epsilon_decay, min_epsilon)

    return stats

def train_q_learning_agent_mixed_seats(
    agent: QLearningAgent,
    game_factory: GameFactory,
    opponent: BaseAgent,
    episodes: int,
    epsilon_decay: float = 0.9995,
    min_epsilon: float = 0.05,
    use_reward_shaping: bool = False,
) -> TrainingStats:
    """
    Train same Q-learning agent while alternating whether it plays as
    Player 1 or Player -1 across episodes

    Reduces first-player bias and makes learned policy more robust
    """
    combined_stats = TrainingStats()

    for episode_index in range(episodes):
        learner_player = 1 if episode_index % 2 == 0 else -1

        episode_stats = train_q_learning_agent(
            agent=agent,
            game_factory=game_factory,
            opponent=opponent,
            episodes=1,
            learner_player=learner_player,
            epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon,
            use_reward_shaping=use_reward_shaping,
        )

        combined_stats.wins += episode_stats.wins
        combined_stats.draws += episode_stats.draws
        combined_stats.losses += episode_stats.losses
        combined_stats.episode_rewards.extend(episode_stats.episode_rewards)
        combined_stats.epsilon_history.extend(episode_stats.epsilon_history)
        combined_stats.outcome_history.extend(episode_stats.outcome_history)

    return combined_stats