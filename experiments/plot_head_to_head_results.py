from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_head_to_head(csv_path: str, title: str, output_path: str) -> None:
    df = pd.read_csv(csv_path)

    labels = [f"{row['agent_a']} vs {row['agent_b']}" for _, row in df.iterrows()]
    a_vals = df["agent_a_win_rate"].tolist()
    b_vals = df["agent_b_win_rate"].tolist()
    d_vals = df["draw_rate"].tolist()

    x = range(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar([i - width for i in x], a_vals, width=width, label="Agent A win rate")
    plt.bar(x, d_vals, width=width, label="Draw rate")
    plt.bar([i + width for i in x], b_vals, width=width, label="Agent B win rate")

    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.ylabel("Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    output_dir = Path("results/final/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_head_to_head(
        csv_path="results/final/head_to_head_tictactoe.csv",
        title="Tic Tac Toe Head-to-Head Results",
        output_path=str(output_dir / "head_to_head_tictactoe.png"),
    )

    plot_head_to_head(
        csv_path="results/final/head_to_head_connect4.csv",
        title="Connect 4 Head-to-Head Results",
        output_path=str(output_dir / "head_to_head_connect4.png"),
    )

    print("\nSaved head-to-head plots:")
    print("- results/final/plots/head_to_head_tictactoe.png")
    print("- results/final/plots/head_to_head_connect4.png")


if __name__ == "__main__":
    main()