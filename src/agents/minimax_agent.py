from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

from src.agents.base_agent import BaseAgent
from src.games.connect4 import Connect4
from src.games.tictactoe import TicTacToe


class SearchTimeoutError(Exception):
    """Raised when search exceeds its configured deadline """
    pass


@dataclass
class SearchStats:
    """Stores simple search statistics for one decision"""
    nodes_visited: int = 0
    cutoffs: int = 0
    max_depth_reached: int = 0


class MinimaxAgent(BaseAgent):
    """
    Minimax agent that can optionally use alpha-beta pruning

    Works for both Tic Tac Toe and Connect 4
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

        self.search_deadline: Optional[float] = None
        self.progress_callback: Optional[Callable[[SearchStats], None]] = None
        self._sync_interval_nodes = 5000

    def configure_probe(
        self,
        search_deadline: Optional[float] = None,
        progress_callback: Optional[Callable[[SearchStats], None]] = None,
    ) -> None:
        """
        Configure optional deadline/progress reporting for long-running searches
        """
        self.search_deadline = search_deadline
        self.progress_callback = progress_callback

    def clear_probe_configuration(self) -> None:
        """
        Clear any previously configured deadline/progress hooks
        """
        self.search_deadline = None
        self.progress_callback = None

    def select_action(self, game: TicTacToe | Connect4) -> int:
        legal_actions = game.available_actions()
        if not legal_actions:
            raise ValueError("No legal actions available.")

        root_player = game.current_player
        self.last_search_stats = SearchStats()
        self._sync_progress(force=True)
        self._check_timeout()

        best_value = -math.inf
        best_action = legal_actions[0]

        alpha = -math.inf
        beta = math.inf

        for action in legal_actions:
            self._check_timeout()

            next_state = game.next_state(action)

            if self.use_alpha_beta:
                value = self._min_value(
                    next_state,
                    root_player=root_player,
                    depth=1,
                    alpha=alpha,
                    beta=beta,
                )
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

            if self.use_alpha_beta:
                alpha = max(alpha, best_value)

        self._sync_progress(force=True)
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
        self._check_timeout()

        if self._should_stop(game, depth):
            return self._evaluate(game, root_player, depth)

        value = -math.inf

        for action in game.available_actions():
            self._check_timeout()

            child = game.next_state(action)
            value = max(
                value,
                self._min_value(child, root_player, depth + 1, alpha, beta),
            )

            if self.use_alpha_beta and alpha is not None and beta is not None:
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.last_search_stats.cutoffs += 1
                    self._sync_progress()
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
        self._check_timeout()

        if self._should_stop(game, depth):
            return self._evaluate(game, root_player, depth)

        value = math.inf

        for action in game.available_actions():
            self._check_timeout()

            child = game.next_state(action)
            value = min(
                value,
                self._max_value(child, root_player, depth + 1, alpha, beta),
            )

            if self.use_alpha_beta and alpha is not None and beta is not None:
                beta = min(beta, value)
                if alpha >= beta:
                    self.last_search_stats.cutoffs += 1
                    self._sync_progress()
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
        Evaluate position from the root player's perspective

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
        self._sync_progress()

    def _check_timeout(self) -> None:
        """
        Raise timeout if search deadline has been configured and exceeded
        """
        if self.search_deadline is not None and time.perf_counter() >= self.search_deadline:
            self._sync_progress(force=True)
            raise SearchTimeoutError("Search exceeded configured deadline.")

    def _sync_progress(self, force: bool = False) -> None:
        """
        Periodically publish current search stats to any configured callback
        """
        if self.progress_callback is None:
            return

        if force or self.last_search_stats.nodes_visited % self._sync_interval_nodes == 0:
            self.progress_callback(self.last_search_stats)