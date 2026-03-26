from pathlib import Path

from src.agents.dqn_agent import DQNAgent
from src.agents.q_learning_agent import QLearningAgent
from src.games.tictactoe import TicTacToe


def test_q_learning_save_and_load(tmp_path):
    filepath = tmp_path / "q_table.json"

    agent = QLearningAgent(seed=42)
    state_key = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    agent.q_table[state_key][4] = 0.75
    agent.save(str(filepath))

    loaded_agent = QLearningAgent(seed=7)
    loaded_agent.load(str(filepath))

    assert loaded_agent.get_q_value(state_key, 4) == 0.75


def test_dqn_save_and_load(tmp_path):
    filepath = tmp_path / "dqn_model.pt"

    agent = DQNAgent(
        state_size=10,
        action_size=9,
        seed=42,
        batch_size=4,
        replay_buffer_size=100,
    )
    agent.save(str(filepath))

    loaded_agent = DQNAgent(
        state_size=10,
        action_size=9,
        seed=7,
        batch_size=4,
        replay_buffer_size=100,
    )
    loaded_agent.load(str(filepath))

    assert loaded_agent.name == agent.name
    assert loaded_agent.discount_factor == agent.discount_factor

def test_loaded_q_learning_agent_can_select_action(tmp_path):
    filepath = tmp_path / "q_agent.json"

    agent = QLearningAgent(seed=42)
    state_key = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    agent.q_table[state_key][0] = 0.5
    agent.q_table[state_key][4] = 1.0
    agent.save(str(filepath))

    loaded_agent = QLearningAgent(seed=7)
    loaded_agent.load(str(filepath))

    game = TicTacToe()
    action = loaded_agent.select_action(game, training=False)

    assert action in game.legal_actions()