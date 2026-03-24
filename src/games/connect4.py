from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Player = int
Action = int


@dataclass
class Connect4:
    """
    Standard 6x7 Connect 4 environment.

    Board values:
    -  1 for player X
    - -1 for player O
    -  0 for empty

    Rows are indexed top to bottom.
    Columns are indexed left to right.
    """

    rows: int = 6
    cols: int = 7
    board: List[List[int]] = field(default_factory=lambda: [[0] * 7 for _ in range(6)])
    current_player: Player = 1

    def clone(self) -> "Connect4":
        """Return a deep copy of the game state."""
        return Connect4(
            rows=self.rows,
            cols=self.cols,
            board=[row.copy() for row in self.board],
            current_player=self.current_player,
        )

    def reset(self) -> None:
        """Reset the game back to an empty board."""
        self.board = [[0] * self.cols for _ in range(self.rows)]
        self.current_player = 1

    def legal_actions(self) -> List[Action]:
        """Return columns that are not full."""
        if self.is_terminal():
            return []
        return [col for col in range(self.cols) if self.board[0][col] == 0]
    
    def available_actions(self) -> List[Action]:
        """ Alias for legal_actions"""
        return self.legal_actions()

    def apply_action(self, action: Action) -> None:
        """Drop a piece into the selected column."""
        if self.is_terminal():
            raise ValueError("Cannot play a move in a finished game.")

        if action < 0 or action >= self.cols:
            raise ValueError(f"Column must be between 0 and {self.cols - 1}, got {action}.")

        if self.board[0][action] != 0:
            raise ValueError(f"Column {action} is full.")

        for row in range(self.rows - 1, -1, -1):
            if self.board[row][action] == 0:
                self.board[row][action] = self.current_player
                self.current_player *= -1
                return

        raise RuntimeError("Could not apply action even though the column looked legal.")

    def next_state(self, action: Action) -> "Connect4":
        """Return the state that results from taking an action."""
        new_state = self.clone()
        new_state.apply_action(action)
        return new_state

    def winner(self) -> Optional[Player]:
        """Return the winner if there is one, otherwise None."""
        # Horizontal
        for row in range(self.rows):
            for col in range(self.cols - 3):
                window = [self.board[row][col + i] for i in range(4)]
                if sum(window) == 4:
                    return 1
                if sum(window) == -4:
                    return -1

        # Vertical
        for row in range(self.rows - 3):
            for col in range(self.cols):
                window = [self.board[row + i][col] for i in range(4)]
                if sum(window) == 4:
                    return 1
                if sum(window) == -4:
                    return -1

        # Diagonal down-right
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                window = [self.board[row + i][col + i] for i in range(4)]
                if sum(window) == 4:
                    return 1
                if sum(window) == -4:
                    return -1

        # Diagonal up-right
        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                window = [self.board[row - i][col + i] for i in range(4)]
                if sum(window) == 4:
                    return 1
                if sum(window) == -4:
                    return -1

        return None

    def is_draw(self) -> bool:
        """Return True if the board is full and nobody has won."""
        return self.winner() is None and all(self.board[0][col] != 0 for col in range(self.cols))

    def is_terminal(self) -> bool:
        """Return True if the game has ended."""
        return self.winner() is not None or self.is_draw()

    def result(self) -> int:
        """
        Return the game result from Player 1's perspective.

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
        Return a hashable representation of the state.

        Including current_player is useful for Q-learning.
        """
        flat_board = [cell for row in self.board for cell in row]
        return tuple(flat_board + [self.current_player])

    def render(self) -> None:
        """Print the board in a readable format."""
        symbols = {1: "X", -1: "O", 0: "."}
        for row in self.board:
            print(" ".join(symbols[cell] for cell in row))
        print("0 1 2 3 4 5 6")

    def __str__(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        lines = [" ".join(symbols[cell] for cell in row) for row in self.board]
        lines.append("0 1 2 3 4 5 6")
        return "\n".join(lines)