from src.agents.random_agent import RandomAgent
from src.games.connect4 import Connect4
from src.games.tictactoe import TicTacToe


def test_random_agent_returns_legal_tictactoe_action():
    game = TicTacToe()
    agent = RandomAgent(seed=42)

    action = agent.select_action(game)
    assert action in game.legal_actions()


def test_random_agent_returns_legal_connect4_action():
    game = Connect4()
    agent = RandomAgent(seed=42)

    action = agent.select_action(game)
    assert action in game.legal_actions()