from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path("results/final/plots")


def _save(fig, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_vs_opponent_tictactoe() -> None:
    # Search data - alternating seat evaluation vs default opponent
    search_df = pd.read_csv("results/final/search_summary_results.csv")
    ttt_search = search_df[search_df["game"] == "tictactoe"]

    # RL data - combine p1/p2 win rates (250 games each seat)
    q_df = pd.read_csv("results/q_learning/tictactoe/evaluation_summary.csv")
    dqn_df = pd.read_csv("results/dqn/tictactoe/evaluation_summary.csv")

    labels = []
    wins, draws, losses = [], [], []

    for _, row in ttt_search.iterrows():
        labels.append(row["algorithm"].replace("_", "-"))
        wins.append(row["player1_win_rate"])
        draws.append(row["draw_rate"])
        losses.append(1 - row["player1_win_rate"] - row["draw_rate"])

    for df, name in [(q_df, "Q-learning"), (dqn_df, "DQN")]:
        row = df.iloc[0]
        # combined win rate across both seats
        w = (row["p1_win_rate"] + row["p2_win_rate"]) / 2
        d = 1 - w  # no losses vs default opponent after training
        l = 0.0
        labels.append(name)
        wins.append(w)
        draws.append(d)
        losses.append(l)

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, wins, label="Win", color="steelblue")
    ax.bar(x, draws, bottom=wins, label="Draw", color="orange")
    ax.bar(x, losses, bottom=[w + d for w, d in zip(wins, draws)], label="Loss", color="tomato")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("Tic Tac Toe — All Algorithms vs Default Opponent")
    ax.legend()
    _save(fig, "tictactoe_vs_opponent.png")


def plot_vs_opponent_connect4() -> None:
    search_df = pd.read_csv("results/final/search_summary_results.csv")
    c4_search = search_df[search_df["game"] == "connect4"]

    q_df = pd.read_csv("results/q_learning/connect4/evaluation_summary.csv")
    dqn_df = pd.read_csv("results/dqn/connect4/evaluation_summary.csv")

    labels = []
    wins, draws, losses = [], [], []

    for _, row in c4_search.iterrows():
        depth = row["depth_limit"]
        alg = row["algorithm"].replace("_", "-")
        labels.append(f"{alg} (d={depth})")
        wins.append(row["player1_win_rate"])
        draws.append(row["draw_rate"])
        losses.append(1 - row["player1_win_rate"] - row["draw_rate"])

    for df, name in [(q_df, "Q-learning*"), (dqn_df, "DQN*")]:
        row = df.iloc[0]
        w = row["eval_win_rate"]
        d = row["eval_draws"] / row["eval_games"]
        l = row["eval_losses"] / row["eval_games"]
        labels.append(name)
        wins.append(w)
        draws.append(d)
        losses.append(l)

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, wins, label="Win", color="steelblue")
    ax.bar(x, draws, bottom=wins, label="Draw", color="orange")
    ax.bar(x, losses, bottom=[w + d for w, d in zip(wins, draws)], label="Loss", color="tomato")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("Connect 4 — All Algorithms vs Opponent\n(* RL agents evaluated against random opponent)")
    ax.legend()
    _save(fig, "connect4_vs_opponent.png")


def plot_head_to_head(game: str, title: str, filename: str) -> None:
    df = pd.read_csv(f"results/final/head_to_head_{game}.csv")

    labels = df["matchup"].tolist()
    a_vals = df["agent_a_win_rate"].tolist()
    b_vals = df["agent_b_win_rate"].tolist()
    d_vals = df["draw_rate"].tolist()

    x = range(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar([i - width for i in x], a_vals, width=width, label="Agent A win rate", color="steelblue")
    ax.bar(list(x), d_vals, width=width, label="Draw rate", color="orange")
    ax.bar([i + width for i in x], b_vals, width=width, label="Agent B win rate", color="tomato")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.legend()
    _save(fig, filename)


def plot_overall_comparison() -> None:
    # Build a summary from both games combining search and RL results
    search_df = pd.read_csv("results/final/search_summary_results.csv")
    q_ttt = pd.read_csv("results/q_learning/tictactoe/evaluation_summary.csv").iloc[0]
    dqn_ttt = pd.read_csv("results/dqn/tictactoe/evaluation_summary.csv").iloc[0]
    q_c4 = pd.read_csv("results/q_learning/connect4/evaluation_summary.csv").iloc[0]
    dqn_c4 = pd.read_csv("results/dqn/connect4/evaluation_summary.csv").iloc[0]

    rows = []
    for _, row in search_df.iterrows():
        depth = row["depth_limit"]
        alg = row["algorithm"]
        game = row["game"]
        label = f"{game}\n{alg} (d={depth})" if game == "connect4" else f"{game}\n{alg}"
        rows.append((label, row["player1_win_rate"], row["draw_rate"]))

    # Tictactoe RL — combined seat win rate
    ttt_q_win = (q_ttt["p1_win_rate"] + q_ttt["p2_win_rate"]) / 2
    ttt_dqn_win = (dqn_ttt["p1_win_rate"] + dqn_ttt["p2_win_rate"]) / 2
    rows.append(("tictactoe\nQ-learning", ttt_q_win, 1 - ttt_q_win))
    rows.append(("tictactoe\nDQN", ttt_dqn_win, 1 - ttt_dqn_win))

    # Connect4 RL — vs random opponent
    rows.append(("connect4\nQ-learning*", q_c4["eval_win_rate"], q_c4["eval_draws"] / q_c4["eval_games"]))
    rows.append(("connect4\nDQN*", dqn_c4["eval_win_rate"], dqn_c4["eval_draws"] / dqn_c4["eval_games"]))

    labels = [r[0] for r in rows]
    win_rates = [r[1] for r in rows]
    draw_rates = [r[2] for r in rows]
    loss_rates = [1 - w - d for w, d in zip(win_rates, draw_rates)]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, win_rates, label="Win", color="steelblue")
    ax.bar(x, draw_rates, bottom=win_rates, label="Draw", color="orange")
    ax.bar(x, loss_rates, bottom=[w + d for w, d in zip(win_rates, draw_rates)], label="Loss", color="tomato")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("Overall Algorithm Comparison Across Both Games\n(* Connect4 RL evaluated against random opponent)")
    ax.legend()
    _save(fig, "overall_comparison.png")


def main() -> None:
    print("Generating report figures...")

    plot_vs_opponent_tictactoe()
    plot_vs_opponent_connect4()
    plot_head_to_head("tictactoe", "Tic Tac Toe — Head-to-Head Results", "head_to_head_tictactoe.png")
    plot_head_to_head("connect4", "Connect 4 — Head-to-Head Results", "head_to_head_connect4.png")
    plot_overall_comparison()

    print("Done.")


if __name__ == "__main__":
    main()
