from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

from src.agents.minimax_agent import MinimaxAgent, SearchTimeoutError
from src.games.connect4 import Connect4
from src.utils.io_utils import save_dict_rows_to_csv


RESULTS_DIR = Path("results/final")
SEARCH_RESULTS_DIR = Path("results/search/connect4")


def make_opening_board() -> Connect4:
    return Connect4()


def make_midgame_board() -> Connect4:
    game = Connect4()

    # Fixed mid-game position chosen to be non-terminal
    # Board filled directly to keep probe reproducible
    game.board = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0],
        [0, 0, -1, 1, -1, 0, 0],
        [0, -1, 1, -1, 1, 0, 0],
        [1, -1, 1, 1, -1, -1, 0],
    ]
    game.current_player = 1
    return game


def board_label_to_factory(board_label: str):
    if board_label == "opening":
        return make_opening_board
    if board_label == "midgame":
        return make_midgame_board
    raise ValueError(f"Unknown board label: {board_label}")


def worker(
    use_alpha_beta: bool,
    board_label: str,
    timeout_seconds: int,
    shared_stats,
    output_queue: mp.Queue,
) -> None:
    game_factory = board_label_to_factory(board_label)
    game = game_factory()

    agent = MinimaxAgent(
        name="alpha_beta" if use_alpha_beta else "minimax",
        use_alpha_beta=use_alpha_beta,
        depth_limit=None,
    )

    def progress_callback(stats) -> None:
        shared_stats["nodes_visited"] = stats.nodes_visited
        shared_stats["cutoffs"] = stats.cutoffs
        shared_stats["max_depth_reached"] = stats.max_depth_reached
        shared_stats["last_update_seconds"] = time.perf_counter() - start_time

    shared_stats["nodes_visited"] = 0
    shared_stats["cutoffs"] = 0
    shared_stats["max_depth_reached"] = 0
    shared_stats["completed"] = False
    shared_stats["timed_out_inside_search"] = False
    shared_stats["last_update_seconds"] = 0.0

    start_time = time.perf_counter()
    deadline = start_time + timeout_seconds

    agent.configure_probe(
        search_deadline=deadline,
        progress_callback=progress_callback,
    )

    try:
        action = agent.select_action(game)
        elapsed = time.perf_counter() - start_time

        shared_stats["nodes_visited"] = agent.last_search_stats.nodes_visited
        shared_stats["cutoffs"] = agent.last_search_stats.cutoffs
        shared_stats["max_depth_reached"] = agent.last_search_stats.max_depth_reached
        shared_stats["completed"] = True

        output_queue.put(
            {
                "completed": True,
                "timed_out_inside_search": False,
                "selected_action": action,
                "elapsed_seconds": elapsed,
                "nodes_visited": agent.last_search_stats.nodes_visited,
                "cutoffs": agent.last_search_stats.cutoffs,
                "max_depth_reached": agent.last_search_stats.max_depth_reached,
                "notes": "",
            }
        )
    except SearchTimeoutError:
        elapsed = time.perf_counter() - start_time
        shared_stats["timed_out_inside_search"] = True

        output_queue.put(
            {
                "completed": False,
                "timed_out_inside_search": True,
                "selected_action": None,
                "elapsed_seconds": elapsed,
                "nodes_visited": shared_stats.get("nodes_visited", 0),
                "cutoffs": shared_stats.get("cutoffs", 0),
                "max_depth_reached": shared_stats.get("max_depth_reached", 0),
                "notes": "Search exceeded configured deadline inside recursive search.",
            }
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        output_queue.put(
            {
                "completed": False,
                "timed_out_inside_search": False,
                "selected_action": None,
                "elapsed_seconds": elapsed,
                "nodes_visited": shared_stats.get("nodes_visited", 0),
                "cutoffs": shared_stats.get("cutoffs", 0),
                "max_depth_reached": shared_stats.get("max_depth_reached", 0),
                "notes": f"Unexpected error: {exc}",
            }
        )
    finally:
        agent.clear_probe_configuration()


def run_probe(
    use_alpha_beta: bool,
    board_label: str,
    timeout_seconds: int = 1800,
) -> dict:
    algorithm = "alpha_beta" if use_alpha_beta else "minimax"

    manager = mp.Manager()
    shared_stats = manager.dict()
    output_queue: mp.Queue = mp.Queue()

    process = mp.Process(
        target=worker,
        args=(use_alpha_beta, board_label, timeout_seconds, shared_stats, output_queue),
    )

    start_time = time.perf_counter()
    process.start()
    process.join(timeout=timeout_seconds + 5)

    if process.is_alive():
        process.terminate()
        process.join()
        elapsed = time.perf_counter() - start_time

        return {
            "game": "connect4",
            "board_label": board_label,
            "algorithm": algorithm,
            "timeout_seconds": timeout_seconds,
            "completed": False,
            "timed_out_inside_search": False,
            "elapsed_seconds": elapsed,
            "selected_action": None,
            "nodes_visited": shared_stats.get("nodes_visited", 0),
            "nodes_per_second": (
                shared_stats.get("nodes_visited", 0) / elapsed if elapsed > 0 else 0.0
            ),
            "cutoffs": shared_stats.get("cutoffs", 0),
            "max_depth_reached": shared_stats.get("max_depth_reached", 0),
            "notes": "Process was still alive after timeout buffer and was terminated externally.",
        }

    elapsed = time.perf_counter() - start_time

    if output_queue.empty():
        return {
            "game": "connect4",
            "board_label": board_label,
            "algorithm": algorithm,
            "timeout_seconds": timeout_seconds,
            "completed": False,
            "timed_out_inside_search": False,
            "elapsed_seconds": elapsed,
            "selected_action": None,
            "nodes_visited": shared_stats.get("nodes_visited", 0),
            "nodes_per_second": (
                shared_stats.get("nodes_visited", 0) / elapsed if elapsed > 0 else 0.0
            ),
            "cutoffs": shared_stats.get("cutoffs", 0),
            "max_depth_reached": shared_stats.get("max_depth_reached", 0),
            "notes": "Worker exited without returning a result payload.",
        }

    result = output_queue.get()

    return {
        "game": "connect4",
        "board_label": board_label,
        "algorithm": algorithm,
        "timeout_seconds": timeout_seconds,
        "completed": result.get("completed", False),
        "timed_out_inside_search": result.get("timed_out_inside_search", False),
        "elapsed_seconds": elapsed,
        "selected_action": result.get("selected_action"),
        "nodes_visited": result.get("nodes_visited"),
        "nodes_per_second": (
            result.get("nodes_visited", 0) / elapsed if elapsed > 0 else 0.0
        ),
        "cutoffs": result.get("cutoffs"),
        "max_depth_reached": result.get("max_depth_reached"),
        "notes": result.get("notes", ""),
    }


def save_probe_metadata(timeout_seconds: int) -> None:
    metadata = {
        "experiment": "connect4_full_search_probe",
        "boards": [
            "opening",
            "midgame",
        ],
        "algorithms": [
            "minimax",
            "alpha_beta",
        ],
        "timeout_seconds": timeout_seconds,
        "notes": [
            "This experiment probes unrestricted Connect 4 search from selected board states.",
            "The goal is to provide evidence that full minimax search is infeasible on standard hardware.",
            "Progress statistics are captured live during recursive search via agent probe callbacks.",
        ],
    }

    metadata_path = RESULTS_DIR / "connect4_full_probe_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SEARCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timeout_seconds = 1800

    rows = []
    for board_label in ["opening", "midgame"]:
        rows.append(run_probe(use_alpha_beta=False, board_label=board_label, timeout_seconds=timeout_seconds))
        rows.append(run_probe(use_alpha_beta=True, board_label=board_label, timeout_seconds=timeout_seconds))

    for row in rows:
        print("\nProbe result")
        for key, value in row.items():
            print(f"{key}: {value}")

    save_dict_rows_to_csv(rows, SEARCH_RESULTS_DIR / "full_search_probe.csv")
    save_dict_rows_to_csv(rows, RESULTS_DIR / "connect4_full_search_probe.csv")
    save_probe_metadata(timeout_seconds)

    print("\nSaved probe files:")
    print("- results/search/connect4/full_search_probe.csv")
    print("- results/final/connect4_full_search_probe.csv")
    print("- results/final/connect4_full_probe_metadata.json")


if __name__ == "__main__":
    main()