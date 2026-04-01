from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from src.agents.default_opponent import DefaultOpponent
from src.agents.dqn_agent import DQNAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.random_agent import RandomAgent
from src.evaluation.matchup_runner import run_matchup
from src.games.connect4 import Connect4
from src.games.tictactoe import TicTacToe
from src.training.train_dqn import train_dqn_agent_mixed_seats
from src.training.train_q_learning import train_q_learning_agent_mixed_seats
from src.utils.io_utils import save_dict_rows_to_csv


RESULTS_DIR = Path("results/final")
MODELS_DIR = Path("results/models")


def _ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _save_metadata(metadata: dict) -> None:
    metadata_path = RESULTS_DIR / "head_to_head_metadata_stresstest.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# Separate filenames so this run does not overwrite baseline or earlier stress models.
def _q_model_path(game_name: str, seed: int) -> Path:
    return MODELS_DIR / f"{game_name}_q_learning_STRESS_seed{seed}.json"


def _dqn_model_path(game_name: str, seed: int) -> Path:
    return MODELS_DIR / f"{game_name}_dqn_STRESS_h256_seed{seed}.pt"


# =========================
# TIC TAC TOE
# =========================

def _load_or_train_tictactoe_q_agent(seed: int) -> QLearningAgent:
    agent = QLearningAgent(
        name="tabular_q_learning",
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon=1.0,
        seed=seed,
    )
    model_path = _q_model_path("tictactoe", seed)

    if model_path.exists():
        agent.load(str(model_path))
    else:
        opponent = DefaultOpponent(name="default_opponent", seed=seed)
        train_q_learning_agent_mixed_seats(
            agent=agent,
            game_factory=TicTacToe,
            opponent=opponent,
            episodes=30000,
            epsilon_decay=0.9993,
            min_epsilon=0.05,
        )
        agent.save(str(model_path))

    agent.set_epsilon(0.0)
    return agent


def _load_or_train_tictactoe_dqn_agent(seed: int) -> DQNAgent:
    agent = DQNAgent(
        name="dqn",
        state_size=10,
        action_size=9,
        learning_rate=5e-4,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.9997,
        min_epsilon=0.05,
        batch_size=64,
        replay_buffer_size=20000,
        target_update_frequency=100,
        hidden_size=256,
        seed=seed,
    )
    model_path = _dqn_model_path("tictactoe", seed)

    if model_path.exists():
        agent.load(str(model_path))
    else:
        opponent = DefaultOpponent(name="default_opponent", seed=seed)
        train_dqn_agent_mixed_seats(
            agent=agent,
            game_factory=TicTacToe,
            opponent=opponent,
            episodes=50000,
        )
        agent.save(str(model_path))

    agent.set_epsilon(0.0)
    return agent


# =========================
# CONNECT 4
# =========================

def _load_or_train_connect4_q_agent(seed: int) -> QLearningAgent:
    agent = QLearningAgent(
        name="tabular_q_learning",
        learning_rate=0.15,
        discount_factor=0.99,
        epsilon=1.0,
        seed=seed,
    )
    model_path = _q_model_path("connect4", seed)

    if model_path.exists():
        agent.load(str(model_path))
    else:
        opponent = RandomAgent(name="random_opponent", seed=seed)
        train_q_learning_agent_mixed_seats(
            agent=agent,
            game_factory=Connect4,
            opponent=opponent,
            episodes=120000,
            epsilon_decay=0.9997,
            min_epsilon=0.05,
        )
        agent.save(str(model_path))

    agent.set_epsilon(0.0)
    return agent


def _load_or_train_connect4_dqn_agent(seed: int) -> DQNAgent:
    agent = DQNAgent(
        name="dqn",
        state_size=43,
        action_size=7,
        learning_rate=5e-4,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.9998,
        min_epsilon=0.05,
        batch_size=128,
        replay_buffer_size=50000,
        target_update_frequency=250,
        hidden_size=256,
        seed=seed,
    )
    model_path = _dqn_model_path("connect4", seed)

    if model_path.exists():
        agent.load(str(model_path))
    else:
        opponent = RandomAgent(name="random_opponent", seed=seed)
        train_dqn_agent_mixed_seats(
            agent=agent,
            game_factory=Connect4,
            opponent=opponent,
            episodes=150000,
        )
        agent.save(str(model_path))

    agent.set_epsilon(0.0)
    return agent


# =========================
# MATCHUPS
# =========================

def _build_matchup_row(
    game_name: str,
    agent_a_name: str,
    agent_b_name: str,
    stats,
) -> dict:
    return {
        "game": game_name,
        "agent_a": agent_a_name,
        "agent_b": agent_b_name,
        "games": stats.games,
        "agent_a_win_rate": stats.player1_win_rate,
        "agent_b_win_rate": stats.player2_win_rate,
        "draw_rate": stats.draw_rate,
    }


def _run_matchup_set(
    game_factory: Callable,
    game_name: str,
    matchups: list[tuple[str, object, str, object]],
    games_per_matchup: int,
) -> list[dict]:
    rows: list[dict] = []

    for agent_a_name, agent_a, agent_b_name, agent_b in matchups:
        stats = run_matchup(
            game_factory=game_factory,
            agent_a=agent_a,
            agent_b=agent_b,
            games=games_per_matchup,
            alternate_starting_player=True,
        )
        rows.append(
            _build_matchup_row(
                game_name=game_name,
                agent_a_name=agent_a_name,
                agent_b_name=agent_b_name,
                stats=stats,
            )
        )

    return rows


def run_tictactoe_head_to_head(seed: int = 42) -> list[dict]:
    minimax_agent = MinimaxAgent(name="minimax", use_alpha_beta=False)
    alphabeta_agent = MinimaxAgent(name="alpha_beta", use_alpha_beta=True)
    q_agent = _load_or_train_tictactoe_q_agent(seed)
    dqn_agent = _load_or_train_tictactoe_dqn_agent(seed)

    matchups = [
        ("minimax", minimax_agent, "alpha_beta", alphabeta_agent),
        ("minimax", minimax_agent, "q_learning", q_agent),
        ("minimax", minimax_agent, "dqn", dqn_agent),
        ("alpha_beta", alphabeta_agent, "q_learning", q_agent),
        ("alpha_beta", alphabeta_agent, "dqn", dqn_agent),
        ("q_learning", q_agent, "dqn", dqn_agent),
    ]

    return _run_matchup_set(
        game_factory=TicTacToe,
        game_name="tictactoe",
        matchups=matchups,
        games_per_matchup=200,
    )


def run_connect4_head_to_head(seed: int = 42) -> list[dict]:
    minimax_agent = MinimaxAgent(name="minimax", use_alpha_beta=False, depth_limit=3)
    alphabeta_agent = MinimaxAgent(name="alpha_beta", use_alpha_beta=True, depth_limit=3)
    q_agent = _load_or_train_connect4_q_agent(seed)
    dqn_agent = _load_or_train_connect4_dqn_agent(seed)

    matchups = [
        ("minimax", minimax_agent, "alpha_beta", alphabeta_agent),
        ("minimax", minimax_agent, "q_learning", q_agent),
        ("minimax", minimax_agent, "dqn", dqn_agent),
        ("alpha_beta", alphabeta_agent, "q_learning", q_agent),
        ("alpha_beta", alphabeta_agent, "dqn", dqn_agent),
        ("q_learning", q_agent, "dqn", dqn_agent),
    ]

    return _run_matchup_set(
        game_factory=Connect4,
        game_name="connect4",
        matchups=matchups,
        games_per_matchup=200,
    )


def main() -> None:
    _ensure_output_dirs()

    ttt_rows = run_tictactoe_head_to_head()
    c4_rows = run_connect4_head_to_head()

    save_dict_rows_to_csv(ttt_rows, RESULTS_DIR / "head_to_head_tictactoe_STRESS.csv")
    save_dict_rows_to_csv(c4_rows, RESULTS_DIR / "head_to_head_connect4_STRESS.csv")

    _save_metadata(
        {
            "type": "STRESS_TEST",
            "tictactoe_games": 200,
            "connect4_games": 200,
            "tictactoe_q_learning_episodes": 30000,
            "tictactoe_dqn_episodes": 50000,
            "connect4_q_learning_episodes": 120000,
            "connect4_dqn_episodes": 150000,
            "dqn_hidden_size": 256,
            "notes": [
                "Very large training budgets used.",
                "Separate model files used to preserve baseline and earlier stress-test results.",
                "This run tests whether increased DQN capacity improves performance.",
            ],
        }
    )

    print("\nSaved STRESS TEST results")
    print("- results/final/head_to_head_tictactoe_STRESS.csv")
    print("- results/final/head_to_head_connect4_STRESS.csv")
    print("- results/final/head_to_head_metadata_stresstest.json")


if __name__ == "__main__":
    main()