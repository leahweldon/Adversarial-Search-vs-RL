from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

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

    def __post_init__(self) -> None:
        if self.episode_rewards is None:
            self.episode_rewards = []


def _reward_from_result(result: int, learner_player: int) -> float:
    """

    Result is always from Player 1's perspective:
        1  -> Player 1 won
        -1 -> Player -1 won
        0  -> draw
    """
    if result == learner_player:
        return 1.0
    if result == 0:
        return 0.3
    return -1.0


def train_q_learning_agent(
    agent: QLearningAgent,
    game_factory: GameFactory,
    opponent: BaseAgent,
    episodes: int,
    learner_player: int = 1,
    epsilon_decay: float = 0.9995,
    min_epsilon: float = 0.05,
) -> TrainingStats:
    """
    Train tabular Q-learning agent through self-contained game episodes

    The learner plays as either:
    - Player 1 if learner_player = 1
    - Player -1 if learner_player = -1

    The opponent is assumed to already implement select_action(game)
    """
    stats = TrainingStats()

    for _ in range(episodes):
        game = game_factory()
        total_episode_reward = 0.0

        state_key: Optional[tuple[int, ...]] = None
        action: Optional[int] = None

        while not game.is_terminal():
            if game.current_player == learner_player:
                state_key = game.get_state_key()
                action = agent.select_action(game, training=True)
                game.apply_action(action)

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

                next_state_key = game.get_state_key()
                next_legal_actions = game.available_actions()

                # Small step reward keeps learning stable without overengineering it
                step_reward = 0.0
                total_episode_reward += step_reward

                agent.update(
                    state_key=state_key,
                    action=action,
                    reward=step_reward,
                    next_state_key=next_state_key,
                    next_legal_actions=next_legal_actions,
                    done=False,
                )

            else:
                opponent_action = opponent.select_action(game)
                game.apply_action(opponent_action)

                if game.is_terminal() and state_key is not None and action is not None:
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

        final_result = game.result()
        if final_result == learner_player:
            stats.wins += 1
        elif final_result == 0:
            stats.draws += 1
        else:
            stats.losses += 1

        stats.episode_rewards.append(total_episode_reward)
        agent.decay_epsilon(epsilon_decay, min_epsilon)

    return stats