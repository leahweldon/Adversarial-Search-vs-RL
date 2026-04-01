from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Player = int
Action = int


@dataclass
class TicTacToe:
    """
    Simple Tic Tac Toe environment

    Board positions are indexed like this:

        0 | 1 | 2
        ---------
        3 | 4 | 5
        ---------
        6 | 7 | 8

    Player 1 uses X
    Player -1 uses O
    Empty cells are 0
    """

    board: List[int] = field(default_factory=lambda: [0] * 9)
    current_player: Player = 1

    WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    )

    def clone(self) -> "TicTacToe":
        """Return deep copy of game state"""
        return TicTacToe(board=self.board.copy(), current_player=self.current_player)

    def reset(self) -> None:
        """Reset game back to empty board"""
        self.board = [0] * 9
        self.current_player = 1

    def legal_actions(self) -> List[Action]:
        """Return all currently available moves"""
        if self.is_terminal():
            return []
        return [i for i, cell in enumerate(self.board) if cell == 0]
    
    def available_actions(self) -> List[Action]:
        """
        Same as legal_actions, but reads a bit more naturally in some agents

        """
        return self.legal_actions()

    def apply_action(self, action: Action) -> None:
        """Apply move for current player"""
        if self.is_terminal():
            raise ValueError("Cannot play a move in a finished game.")

        if action < 0 or action >= 9:
            raise ValueError(f"Action must be between 0 and 8, got {action}.")

        if self.board[action] != 0:
            raise ValueError(f"Cell {action} is already occupied.")

        self.board[action] = self.current_player
        self.current_player *= -1

    def next_state(self, action: Action) -> "TicTacToe":
        """Return state that results from taking an action"""
        new_state = self.clone()
        new_state.apply_action(action)
        return new_state

    def winner(self) -> Optional[Player]:
        """Return 1 if X has won, -1 if O has won, otherwise None"""
        for a, b, c in self.WIN_LINES:
            line_sum = self.board[a] + self.board[b] + self.board[c]
            if line_sum == 3:
                return 1
            if line_sum == -3:
                return -1
        return None

    def is_draw(self) -> bool:
        """Return True if board is full and neither player has won"""
        return self.winner() is None and all(cell != 0 for cell in self.board)

    def is_terminal(self) -> bool:
        """Return True if game has ended"""
        return self.winner() is not None or self.is_draw()

    def result(self) -> int:
        """
        Return game result from Player 1's perspective

        1  -> Player 1 won
        -1 -> Player -1 won
        0  -> draw or unfinished
        """
        game_winner = self.winner()
        if game_winner is not None:
            return game_winner
        return 0

    def get_state_key(self) -> Tuple[int, ...]:
        """
        Return hashable representation of current state

        Including current_player makes this suitable for tabular RL
        """
        return tuple(self.board + [self.current_player])

    def render(self) -> None:
        """Print current board in human-friendly way"""
        symbols = {1: "X", -1: "O", 0: " "}
        rows = []
        for i in range(0, 9, 3):
            row = " | ".join(symbols[self.board[j]] for j in range(i, i + 3))
            rows.append(row)
        print("\n---------\n".join(rows))

    def __str__(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        rows = []
        for i in range(0, 9, 3):
            rows.append(" ".join(symbols[self.board[j]] for j in range(i, i + 3)))
        return "\n".join(rows)