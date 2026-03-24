from __future__ import annotations

import random
from typing import List, Optional

from src.agents.base_agent import BaseAgent
from src.games.connect4 import Connect4
from src.games.tictactoe import TicTacToe


class DefaultOpponent(BaseAgent):
    """
    A opponent for both Tic Tac Toe and Connect 4.

    1. take a winning move immediately if one exists
    2. block the opponent's immediate winning move if needed

    """

    def __init__(self, name: str = "default_opponent", seed: Optional[int] = None) -> None:
        super().__init__(name=name)
        self.rng = random.Random(seed)

    def select_action(self, game: TicTacToe | Connect4) -> int:
        if isinstance(game, TicTacToe):
            return self._select_tictactoe_action(game)

        if isinstance(game, Connect4):
            return self._select_connect4_action(game)

        raise TypeError(f"Unsupported game type: {type(game).__name__}")

    def _select_tictactoe_action(self, game: TicTacToe) -> int:
        legal_actions = game.legal_actions()

        winning_move = self._find_immediate_win(game, game.current_player)
        if winning_move is not None:
            return winning_move

        blocking_move = self._find_immediate_win(game, -game.current_player)
        if blocking_move is not None:
            return blocking_move

        # After immediate tactics, prefer strong board positions.
        preferred_order = [4, 0, 2, 6, 8, 1, 3, 5, 7]
        for action in preferred_order:
            if action in legal_actions:
                return action

        return self.rng.choice(legal_actions)

    def _select_connect4_action(self, game: Connect4) -> int:
        legal_actions = game.legal_actions()

        winning_move = self._find_immediate_win(game, game.current_player)
        if winning_move is not None:
            return winning_move

        blocking_move = self._find_immediate_win(game, -game.current_player)
        if blocking_move is not None:
            return blocking_move

        # In Connect 4, central columns are generally stronger because they are part
        # of more possible four-in-a-row patterns.
        centre_preference = self._ordered_by_centre_bias(legal_actions, game.cols)

        # Among equally legal moves, we slightly prefer actions that create more
        # pressure for the current player on the next turn.
        best_score = None
        best_actions: List[int] = []

        for action in centre_preference:
            next_state = game.next_state(action)
            score = self._score_connect4_position(next_state, player=game.current_player)

            if best_score is None or score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return self.rng.choice(best_actions)

    def _find_immediate_win(self, game: TicTacToe | Connect4, player: int) -> Optional[int]:
        """
        Return a move that gives player an immediate win, if one exists.

        """
        for action in game.legal_actions():
            trial_state = game.clone()
            trial_state.current_player = player
            trial_state.apply_action(action)

            if trial_state.winner() == player:
                return action

        return None

    def _ordered_by_centre_bias(self, legal_actions: List[int], cols: int) -> List[int]:
        """
        Sort legal columns by distance from the centre column.
        """
        centre = cols // 2
        return sorted(legal_actions, key=lambda col: (abs(col - centre), col))

    def _score_connect4_position(self, game: Connect4, player: int) -> int:
        """
        Lightweight evaluation used only by  default opponent.

        The scoring rewards:
        - occupying the centre
        - making 2-in-a-row and 3-in-a-row patterns with room to grow
        - limiting the opponent's good windows
        """
        board = game.board
        rows = game.rows
        cols = game.cols

        score = 0
        opponent = -player

        centre_col = cols // 2
        centre_count = sum(1 for row in range(rows) if board[row][centre_col] == player)
        score += 3 * centre_count

        windows = []

        # Horizontal windows
        for row in range(rows):
            for col in range(cols - 3):
                windows.append([board[row][col + i] for i in range(4)])

        # Vertical windows
        for row in range(rows - 3):
            for col in range(cols):
                windows.append([board[row + i][col] for i in range(4)])

        # Down-right diagonals
        for row in range(rows - 3):
            for col in range(cols - 3):
                windows.append([board[row + i][col + i] for i in range(4)])

        # Up-right diagonals
        for row in range(3, rows):
            for col in range(cols - 3):
                windows.append([board[row - i][col + i] for i in range(4)])

        for window in windows:
            score += self._score_window(window, player, opponent)

        return score

    def _score_window(self, window: List[int], player: int, opponent: int) -> int:
        """
        Give small positional value to a 4-cell window.
        """
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(0)

        if player_count == 4:
            return 10_000
        if player_count == 3 and empty_count == 1:
            return 50
        if player_count == 2 and empty_count == 2:
            return 10

        if opponent_count == 3 and empty_count == 1:
            return -40
        if opponent_count == 2 and empty_count == 2:
            return -8

        return 0