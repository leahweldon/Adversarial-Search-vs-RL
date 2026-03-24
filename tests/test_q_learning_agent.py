from src.agents.default_opponent import DefaultOpponent
from src.agents.q_learning_agent import QLearningAgent
from src.games.tictactoe import TicTacToe
from src.training.train_q_learning import train_q_learning_agent


def test_q_learning_agent_selects_legal_action():
    game = TicTacToe()
    agent = QLearningAgent(seed=42)

    action = agent.select_action(game, training=False)

    assert action in game.legal_actions()


def test_q_learning_update_changes_q_value():
    agent = QLearningAgent(learning_rate=0.5, discount_factor=0.9, epsilon=0.0, seed=42)

    state_key = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    next_state_key = (1, 0, 0, 0, 0, 0, 0, 0, 0, -1)

    agent.update(
        state_key=state_key,
        action=0,
        reward=1.0,
        next_state_key=next_state_key,
        next_legal_actions=[1, 2, 3],
        done=False,
    )

    assert agent.get_q_value(state_key, 0) != 0.0


def test_q_learning_terminal_update_uses_reward_only():
    agent = QLearningAgent(learning_rate=1.0, discount_factor=0.9, epsilon=0.0, seed=42)

    state_key = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)

    agent.update(
        state_key=state_key,
        action=4,
        reward=1.0,
        next_state_key=None,
        next_legal_actions=None,
        done=True,
    )

    assert agent.get_q_value(state_key, 4) == 1.0


def test_q_learning_training_runs_and_collects_stats():
    agent = QLearningAgent(seed=42)
    opponent = DefaultOpponent(seed=42)

    stats = train_q_learning_agent(
        agent=agent,
        game_factory=TicTacToe,
        opponent=opponent,
        episodes=20,
        learner_player=1,
    )

    assert stats.wins + stats.losses + stats.draws == 20
    assert len(stats.episode_rewards) == 20


def test_q_learning_policy_grows_after_training():
    agent = QLearningAgent(seed=42)
    opponent = DefaultOpponent(seed=42)

    initial_size = agent.policy_size()

    train_q_learning_agent(
        agent=agent,
        game_factory=TicTacToe,
        opponent=opponent,
        episodes=50,
        learner_player=1,
    )

    assert agent.policy_size() > initial_size