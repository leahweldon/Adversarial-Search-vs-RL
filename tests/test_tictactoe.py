from src.games.tictactoe import TicTacToe


def test_tictactoe_starts_empty():
    game = TicTacToe()
    assert game.board == [0] * 9
    assert game.current_player == 1
    assert game.legal_actions() == list(range(9))


def test_tictactoe_apply_action():
    game = TicTacToe()
    game.apply_action(0)
    assert game.board[0] == 1
    assert game.current_player == -1


def test_tictactoe_detects_row_win():
    game = TicTacToe(board=[1, 1, 1, 0, 0, 0, 0, 0, 0], current_player=-1)
    assert game.winner() == 1
    assert game.is_terminal()


def test_tictactoe_detects_draw():
    game = TicTacToe(board=[1, -1, 1, 1, -1, -1, -1, 1, 1], current_player=1)
    assert game.winner() is None
    assert game.is_draw()
    assert game.is_terminal()