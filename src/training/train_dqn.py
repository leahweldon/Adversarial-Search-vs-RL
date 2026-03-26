from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from src.agents.base_agent import BaseAgent
from src.agents.dqn_agent import DQNAgent
from src.games.connect4 import Connect4
from src.games.tictactoe import TicTacToe

GameFactory = Callable[[], TicTacToe | Connect4]


@dataclass
class DQNTrainingStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    episode_rewards: List[float] | None = None
    losses_history: List[float] | None = None
    epsilon_history: List[float] | None = None
    outcome_history: List[int] | None = None

    def __post_init__(self) -> None:
        if self.episode_rewards is None:
            self.episode_rewards = []
        if self.losses_history is None:
            self.losses_history = []
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


def train_dqn_agent(
    agent: DQNAgent,
    game_factory: GameFactory,
    opponent: BaseAgent,
    episodes: int,
    learner_player: int = 1,
) -> DQNTrainingStats:
    """
    Train DQN against fixed opponent

    Each stored transition spans:
    learner move -> opponent reply -> next learner turn (or terminal)
    """
    stats = DQNTrainingStats()

    for _ in range(episodes):
        game = game_factory()
        total_episode_reward = 0.0
        episode_loss_values: List[float] = []

        while not game.is_terminal():
            if game.current_player != learner_player:
                opponent_action = opponent.select_action(game)
                game.apply_action(opponent_action)
                continue

            state = agent.game_to_state_vector(game)
            action = agent.select_action(game, training=True)
            game.apply_action(action)

            # Learner wins or draws immediately
            if game.is_terminal():
                result = game.result()
                reward = _reward_from_result(result, learner_player)
                total_episode_reward += reward

                next_state = agent.game_to_state_vector(game)
                agent.store_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=True,
                    next_legal_actions=[],
                )

                loss = agent.train_step()
                if loss is not None:
                    episode_loss_values.append(loss)
                break

            # Opponent reply is part of same transition
            opponent_action = opponent.select_action(game)
            game.apply_action(opponent_action)

            if game.is_terminal():
                result = game.result()
                reward = _reward_from_result(result, learner_player)
                total_episode_reward += reward

                next_state = agent.game_to_state_vector(game)
                agent.store_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=True,
                    next_legal_actions=[],
                )

                loss = agent.train_step()
                if loss is not None:
                    episode_loss_values.append(loss)
                break

            next_state = agent.game_to_state_vector(game)
            next_legal_actions = game.available_actions()

            reward = 0.0
            total_episode_reward += reward

            agent.store_experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=False,
                next_legal_actions=next_legal_actions,
            )

            loss = agent.train_step()
            if loss is not None:
                episode_loss_values.append(loss)

        result = game.result()
        if result == learner_player:
            stats.wins += 1
            stats.outcome_history.append(1)
        elif result == 0:
            stats.draws += 1
            stats.outcome_history.append(0)
        else:
            stats.losses += 1
            stats.outcome_history.append(-1)

        stats.episode_rewards.append(total_episode_reward)
        stats.epsilon_history.append(agent.epsilon)

        mean_loss = sum(episode_loss_values) / len(episode_loss_values) if episode_loss_values else 0.0
        stats.losses_history.append(mean_loss)

        agent.decay_epsilon()

    return stats

def train_dqn_agent_mixed_seats(
    agent: DQNAgent,
    game_factory: GameFactory,
    opponent: BaseAgent,
    episodes: int,
) -> DQNTrainingStats:
    """
    Train same DQN agent while alternating whether it plays as
    Player 1 or Player -1 across episodes
    """
    combined_stats = DQNTrainingStats()

    for episode_index in range(episodes):
        learner_player = 1 if episode_index % 2 == 0 else -1

        episode_stats = train_dqn_agent(
            agent=agent,
            game_factory=game_factory,
            opponent=opponent,
            episodes=1,
            learner_player=learner_player,
        )

        combined_stats.wins += episode_stats.wins
        combined_stats.draws += episode_stats.draws
        combined_stats.losses += episode_stats.losses
        combined_stats.episode_rewards.extend(episode_stats.episode_rewards)
        combined_stats.losses_history.extend(episode_stats.losses_history)
        combined_stats.epsilon_history.extend(episode_stats.epsilon_history)
        combined_stats.outcome_history.extend(episode_stats.outcome_history)

    return combined_stats