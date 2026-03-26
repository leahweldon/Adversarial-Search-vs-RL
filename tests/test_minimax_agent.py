import time

import pytest

from src.agents.minimax_agent import MinimaxAgent, SearchTimeoutError
from src.games.connect4 import Connect4
from src.games.tictactoe import TicTacToe


def test_minimax_tictactoe_finds_winning_move():
    game = TicTacToe(
        board=[
            1, 1, 0,
            0, -1, 0,
            0, 0, -1,
        ],
        current_player=1,
    )

    agent = MinimaxAgent()
    action = agent.select_action(game)

    assert action == 2


def test_minimax_tictactoe_blocks_loss():
    game = TicTacToe(
        board=[
            -1, -1, 0,
            0, 1, 0,
            0, 0, 0,
        ],
        current_player=1,
    )

    agent = MinimaxAgent()
    action = agent.select_action(game)

    assert action == 2


def test_alpha_beta_matches_minimax_on_tictactoe_position():
    game = TicTacToe(
        board=[
            1, -1, 1,
            0, -1, 0,
            0, 0, 0,
        ],
        current_player=1,
    )

    minimax_agent = MinimaxAgent(use_alpha_beta=False)
    alphabeta_agent = MinimaxAgent(use_alpha_beta=True)

    minimax_action = minimax_agent.select_action(game)
    alphabeta_action = alphabeta_agent.select_action(game)

    assert minimax_action == alphabeta_action


def test_alpha_beta_visits_no_more_nodes_than_plain_minimax():
    game = TicTacToe(
        board=[
            1, -1, 0,
            0, 1, 0,
            -1, 0, 0,
        ],
        current_player=1,
    )

    minimax_agent = MinimaxAgent(use_alpha_beta=False)
    alphabeta_agent = MinimaxAgent(use_alpha_beta=True)

    minimax_agent.select_action(game)
    alphabeta_agent.select_action(game)

    assert alphabeta_agent.last_search_stats.nodes_visited <= minimax_agent.last_search_stats.nodes_visited


def test_depth_limited_connect4_returns_legal_move():
    game = Connect4()
    agent = MinimaxAgent(use_alpha_beta=True, depth_limit=3)

    action = agent.select_action(game)

    assert action in game.legal_actions()


def test_depth_limited_connect4_takes_immediate_win():
    game = Connect4()
    game.board[5][0] = 1
    game.board[5][1] = 1
    game.board[5][2] = 1
    game.current_player = 1

    agent = MinimaxAgent(use_alpha_beta=True, depth_limit=3)
    action = agent.select_action(game)

    assert action == 3


def test_probe_configuration_can_be_set_and_cleared():
    agent = MinimaxAgent(use_alpha_beta=True, depth_limit=3)

    callback_calls = []

    def callback(stats):
        callback_calls.append(stats.nodes_visited)

    deadline = time.perf_counter() + 1.0
    agent.configure_probe(
        search_deadline=deadline,
        progress_callback=callback,
    )

    assert agent.search_deadline == deadline
    assert agent.progress_callback is callback

    agent.clear_probe_configuration()

    assert agent.search_deadline is None
    assert agent.progress_callback is None


def test_timeout_raises_search_timeout_error():
    game = TicTacToe(
        board=[
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
        ],
        current_player=1,
    )

    agent = MinimaxAgent(use_alpha_beta=False)
    agent.configure_probe(search_deadline=time.perf_counter() - 0.001)

    with pytest.raises(SearchTimeoutError):
        agent.select_action(game)