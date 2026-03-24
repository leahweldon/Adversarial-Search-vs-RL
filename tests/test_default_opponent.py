from src.agents.default_opponent import DefaultOpponent
from src.games.connect4 import Connect4
from src.games.tictactoe import TicTacToe


def test_default_opponent_tictactoe_takes_winning_move():
    game = TicTacToe(
        board=[
            1, 1, 0,
            0, -1, 0,
            0, 0, -1,
        ],
        current_player=1,
    )

    agent = DefaultOpponent()
    action = agent.select_action(game)

    assert action == 2


def test_default_opponent_tictactoe_blocks_opponent_win():
    game = TicTacToe(
        board=[
            -1, -1, 0,
            0, 1, 0,
            0, 0, 1,
        ],
        current_player=1,
    )

    agent = DefaultOpponent()
    action = agent.select_action(game)

    assert action == 2


def test_default_opponent_tictactoe_prefers_centre_on_empty_board():
    game = TicTacToe()
    agent = DefaultOpponent()
    action = agent.select_action(game)

    assert action == 4


def test_default_opponent_connect4_takes_winning_move():
    game = Connect4()
    game.board[5][0] = 1
    game.board[5][1] = 1
    game.board[5][2] = 1
    game.current_player = 1

    agent = DefaultOpponent()
    action = agent.select_action(game)

    assert action == 3


def test_default_opponent_connect4_blocks_opponent_win():
    game = Connect4()
    game.board[5][0] = -1
    game.board[5][1] = -1
    game.board[5][2] = -1
    game.current_player = 1

    agent = DefaultOpponent()
    action = agent.select_action(game)

    assert action == 3


def test_default_opponent_connect4_prefers_centre_on_empty_board():
    game = Connect4()
    agent = DefaultOpponent(seed=42)
    action = agent.select_action(game)

    assert action == 3