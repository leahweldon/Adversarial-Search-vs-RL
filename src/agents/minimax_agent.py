from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.games.connect4 import Connect4
from src.games.tictactoe import TicTacToe


@dataclass
class SearchStats:
    """
    Stores simple search statistics for one decision
    These are useful later when comparing plain minimax with alpha-beta pruning
    """
    nodes_visited: int = 0
    cutoffs: int = 0
    max_depth_reached: int = 0


class MinimaxAgent(BaseAgent):
    """
    Minimax agent that can optionally use alpha-beta pruning

    This class is designed to work for both Tic Tac Toe and Connect 4
    For Tic Tac Toe, full-depth search is feasible
    For Connect 4, you will usually want to set a depth limit
    """

    def __init__(
        self,
        name: str = "minimax",
        use_alpha_beta: bool = False,
        depth_limit: Optional[int] = None,
    ) -> None:
        super().__init__(name=name)
        self.use_alpha_beta = use_alpha_beta
        self.depth_limit = depth_limit
        self.last_search_stats = SearchStats()

    def select_action(self, game: TicTacToe | Connect4) -> int:
        legal_actions = game.available_actions()
        if not legal_actions:
            raise ValueError("No legal actions available.")

        root_player = game.current_player
        self.last_search_stats = SearchStats()

        best_value = -math.inf
        best_action = legal_actions[0]

        alpha = -math.inf
        beta = math.inf

        for action in legal_actions:
            next_state = game.next_state(action)

            if self.use_alpha_beta:
                value = self._min_value(
                    next_state,
                    root_player=root_player,
                    depth=1,
                    alpha=alpha,
                    beta=beta,
                )
                alpha = max(alpha, best_value)
            else:
                value = self._min_value(
                    next_state,
                    root_player=root_player,
                    depth=1,
                    alpha=None,
                    beta=None,
                )

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _max_value(
        self,
        game: TicTacToe | Connect4,
        root_player: int,
        depth: int,
        alpha: Optional[float],
        beta: Optional[float],
    ) -> float:
        self._update_stats(depth)

        if self._should_stop(game, depth):
            return self._evaluate(game, root_player, depth)

        value = -math.inf

        for action in game.available_actions():
            child = game.next_state(action)
            value = max(
                value,
                self._min_value(child, root_player, depth + 1, alpha, beta),
            )

            if self.use_alpha_beta and alpha is not None and beta is not None:
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.last_search_stats.cutoffs += 1
                    break

        return value

    def _min_value(
        self,
        game: TicTacToe | Connect4,
        root_player: int,
        depth: int,
        alpha: Optional[float],
        beta: Optional[float],
    ) -> float:
        self._update_stats(depth)

        if self._should_stop(game, depth):
            return self._evaluate(game, root_player, depth)

        value = math.inf

        for action in game.available_actions():
            child = game.next_state(action)
            value = min(
                value,
                self._max_value(child, root_player, depth + 1, alpha, beta),
            )

            if self.use_alpha_beta and alpha is not None and beta is not None:
                beta = min(beta, value)
                if alpha >= beta:
                    self.last_search_stats.cutoffs += 1
                    break

        return value

    def _should_stop(self, game: TicTacToe | Connect4, depth: int) -> bool:
        if game.is_terminal():
            return True

        if self.depth_limit is not None and depth >= self.depth_limit:
            return True

        return False

    def _evaluate(self, game: TicTacToe | Connect4, root_player: int, depth: int) -> float:
        """
        Evaluate a position from the root player's perspective

        For terminal states:
        - win: strong positive score
        - loss: strong negative score
        - draw: zero

        For non-terminal cutoffs:
        - Tic Tac Toe: lightweight positional evaluation
        - Connect 4: window-based heuristic
        """
        winner = game.winner()
        if winner is not None:
            if winner == root_player:
                return 10000 - depth
            return -10000 + depth

        if game.is_draw():
            return 0

        if isinstance(game, TicTacToe):
            return self._evaluate_tictactoe(game, root_player)

        if isinstance(game, Connect4):
            return self._evaluate_connect4(game, root_player)

        raise TypeError(f"Unsupported game type: {type(game).__name__}")

    def _evaluate_tictactoe(self, game: TicTacToe, root_player: int) -> float:
        """
        Simple positional evaluation for depth-limited Tic Tac Toe

        This is mostly a fallback because full search is cheap for Tic Tac Toe,
        but having an evaluation function keeps the agent general and complete
        """
        score = 0
        opponent = -root_player

        for a, b, c in game.WIN_LINES:
            window = [game.board[a], game.board[b], game.board[c]]
            score += self._score_tictactoe_window(window, root_player, opponent)

        return score

    def _score_tictactoe_window(self, window: list[int], player: int, opponent: int) -> int:
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(0)

        if player_count == 3:
            return 100
        if player_count == 2 and empty_count == 1:
            return 10
        if player_count == 1 and empty_count == 2:
            return 1

        if opponent_count == 2 and empty_count == 1:
            return -12
        if opponent_count == 1 and empty_count == 2:
            return -1

        return 0

    def _evaluate_connect4(self, game: Connect4, root_player: int) -> float:
        """
        Heuristic evaluation for depth-limited Connect 4

        Rewards:
        - centre control
        - promising 2/3-in-a-row windows
        Penalises:
        - opponent threats
        """
        board = game.board
        rows = game.rows
        cols = game.cols
        opponent = -root_player
        score = 0

        centre_col = cols // 2
        centre_count = sum(1 for row in range(rows) if board[row][centre_col] == root_player)
        score += 6 * centre_count

        windows = []

        for row in range(rows):
            for col in range(cols - 3):
                windows.append([board[row][col + i] for i in range(4)])

        for row in range(rows - 3):
            for col in range(cols):
                windows.append([board[row + i][col] for i in range(4)])

        for row in range(rows - 3):
            for col in range(cols - 3):
                windows.append([board[row + i][col + i] for i in range(4)])

        for row in range(3, rows):
            for col in range(cols - 3):
                windows.append([board[row - i][col + i] for i in range(4)])

        for window in windows:
            score += self._score_connect4_window(window, root_player, opponent)

        return score

    def _score_connect4_window(self, window: list[int], player: int, opponent: int) -> int:
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(0)

        if player_count == 4:
            return 1000
        if player_count == 3 and empty_count == 1:
            return 50
        if player_count == 2 and empty_count == 2:
            return 10

        if opponent_count == 3 and empty_count == 1:
            return -80
        if opponent_count == 2 and empty_count == 2:
            return -8

        return 0

    def _update_stats(self, depth: int) -> None:
        self.last_search_stats.nodes_visited += 1
        self.last_search_stats.max_depth_reached = max(
            self.last_search_stats.max_depth_reached,
            depth,
        )