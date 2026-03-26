from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.base_agent import BaseAgent
from src.games.connect4 import Connect4
from src.games.tictactoe import TicTacToe

GameType = TicTacToe | Connect4


@dataclass
class Experience:
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    done: bool
    next_legal_actions: List[int]


class DQNetwork(nn.Module):
    """
    Small feedforward network for board-value estimation
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent with:
    - epsilon-greedy exploration
    - replay buffer
    - target network
    - legal-action masking
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        name: str = "dqn",
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
        batch_size: int = 64,
        replay_buffer_size: int = 10000,
        target_update_frequency: int = 200,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)

        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.rng = random.Random(seed)

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer: Deque[Experience] = deque(maxlen=replay_buffer_size)
        self.training_steps = 0

    def select_action(self, game: GameType, training: bool = False) -> int:
        legal_actions = game.available_actions()
        if not legal_actions:
            raise ValueError("No legal actions available.")

        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)

        state = self.game_to_state_vector(game)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0)

        masked_q_values = torch.full_like(q_values, float("-inf"))
        for action in legal_actions:
            masked_q_values[action] = q_values[action]

        return int(torch.argmax(masked_q_values).item())

    def store_experience(
        self,
        state: List[float],
        action: int,
        reward: float,
        next_state: List[float],
        done: bool,
        next_legal_actions: List[int],
    ) -> None:
        self.replay_buffer.append(
            Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                next_legal_actions=next_legal_actions,
            )
        )

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.rng.sample(list(self.replay_buffer), self.batch_size)

        states = torch.tensor([exp.state for exp in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([exp.action for exp in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([exp.next_state for exp in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32, device=self.device)

        current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_values_all = self.target_net(next_states)
            next_q_values = []

            for i, exp in enumerate(batch):
                if exp.done or not exp.next_legal_actions:
                    next_q_values.append(0.0)
                else:
                    legal_qs = next_q_values_all[i][exp.next_legal_actions]
                    next_q_values.append(torch.max(legal_qs).item())

            next_q_values_tensor = torch.tensor(next_q_values, dtype=torch.float32, device=self.device)
            targets = rewards + (1 - dones) * self.discount_factor * next_q_values_tensor

        loss = self.loss_fn(current_q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def save(self, filepath: str) -> None:
        """
        Save trained DQN model and metadata
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "name": self.name,
                "state_size": self.state_size,
                "action_size": self.action_size,
                "discount_factor": self.discount_factor,
                "epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "min_epsilon": self.min_epsilon,
                "training_steps": self.training_steps,
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, filepath: str) -> None:
        """
        Load trained DQN model and metadata
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.name = checkpoint["name"]
        self.discount_factor = checkpoint["discount_factor"]
        self.epsilon = checkpoint["epsilon"]
        self.epsilon_decay = checkpoint["epsilon_decay"]
        self.min_epsilon = checkpoint["min_epsilon"]
        self.training_steps = checkpoint["training_steps"]

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    @staticmethod
    def game_to_state_vector(game: GameType) -> List[float]:
        """
        Convert game state into a flat numeric vector for neural network
        """
        if isinstance(game, TicTacToe):
            return [float(cell) for cell in game.board] + [float(game.current_player)]

        if isinstance(game, Connect4):
            flat_board = [float(cell) for row in game.board for cell in row]
            return flat_board + [float(game.current_player)]

        raise TypeError(f"Unsupported game type: {type(game).__name__}")