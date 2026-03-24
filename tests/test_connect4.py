from src.games.connect4 import Connect4


def test_connect4_starts_empty():
    game = Connect4()
    assert len(game.board) == 6
    assert len(game.board[0]) == 7
    assert game.current_player == 1
    assert game.legal_actions() == [0, 1, 2, 3, 4, 5, 6]


def test_connect4_apply_action_drops_to_bottom():
    game = Connect4()
    game.apply_action(3)
    assert game.board[5][3] == 1
    assert game.current_player == -1


def test_connect4_detects_horizontal_win():
    game = Connect4()
    game.board[5][0] = 1
    game.board[5][1] = 1
    game.board[5][2] = 1
    game.board[5][3] = 1
    assert game.winner() == 1
    assert game.is_terminal()


def test_connect4_detects_vertical_win():
    game = Connect4()
    game.board[5][0] = -1
    game.board[4][0] = -1
    game.board[3][0] = -1
    game.board[2][0] = -1
    assert game.winner() == -1
    assert game.is_terminal()