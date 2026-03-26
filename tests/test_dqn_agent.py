from src.agents.dqn_agent import DQNAgent
from src.games.tictactoe import TicTacToe


def test_dqn_state_vector_size_for_tictactoe():
    game = TicTacToe()
    state = DQNAgent.game_to_state_vector(game)

    assert len(state) == 10


def test_dqn_selects_legal_action():
    game = TicTacToe()
    agent = DQNAgent(
        state_size=10,
        action_size=9,
        epsilon=0.0,
        batch_size=4,
        replay_buffer_size=100,
        seed=42,
    )

    action = agent.select_action(game, training=False)
    assert action in game.legal_actions()


def test_dqn_stores_experience():
    agent = DQNAgent(
        state_size=10,
        action_size=9,
        batch_size=4,
        replay_buffer_size=100,
        seed=42,
    )

    agent.store_experience(
        state=[0.0] * 10,
        action=0,
        reward=1.0,
        next_state=[0.0] * 10,
        done=False,
        next_legal_actions=[1, 2, 3],
    )

    assert len(agent.replay_buffer) == 1


def test_dqn_train_step_returns_none_if_not_enough_samples():
    agent = DQNAgent(
        state_size=10,
        action_size=9,
        batch_size=4,
        replay_buffer_size=100,
        seed=42,
    )

    loss = agent.train_step()
    assert loss is None