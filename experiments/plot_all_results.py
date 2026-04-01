from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_master_win_rates(master_csv: str, output_path: str) -> None:
    df = pd.read_csv(master_csv)

    labels = [f"{row['game']} | {row['algorithm']}" for _, row in df.iterrows()]
    values = df["win_rate"].tolist()

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Win rate")
    plt.title("Final Win Rate Comparison Across Main Experiments")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_search_efficiency(search_csv: str, output_path: str) -> None:
    df = pd.read_csv(search_csv)

    labels = [f"{row['game']} | {row['algorithm']} | d={row['depth_limit']}" for _, row in df.iterrows()]
    values = df["player1_avg_move_time"].tolist()

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average move time (seconds)")
    plt.title("Search Efficiency Comparison")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_search_nodes(search_csv: str, output_path: str) -> None:
    df = pd.read_csv(search_csv)

    labels = [f"{row['game']} | {row['algorithm']} | d={row['depth_limit']}" for _, row in df.iterrows()]
    values = df["player1_nodes"].tolist()

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Total nodes visited")
    plt.title("Search Node Expansion Comparison")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_rl_only(rl_csv: str, output_path: str) -> None:
    df = pd.read_csv(rl_csv)

    labels = [f"{row['game']} | {row['algorithm']}" for _, row in df.iterrows()]
    values = df["win_rate_mean"].tolist()

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean win rate")
    plt.title("RL Algorithm Comparison")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    output_dir = Path("results/final/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_master_win_rates(
        master_csv="results/final/master_summary.csv",
        output_path=str(output_dir / "master_win_rates.png"),
    )

    plot_search_efficiency(
        search_csv="results/final/search_summary_results.csv",
        output_path=str(output_dir / "search_efficiency.png"),
    )

    plot_search_nodes(
        search_csv="results/final/search_summary_results.csv",
        output_path=str(output_dir / "search_nodes.png"),
    )

    plot_rl_only(
        rl_csv="results/final/rl_summary_results.csv",
        output_path=str(output_dir / "rl_only_comparison.png"),
    )

    print("\nSaved final plots:")
    print("- results/final/plots/master_win_rates.png")
    print("- results/final/plots/search_efficiency.png")
    print("- results/final/plots/search_nodes.png")
    print("- results/final/plots/rl_only_comparison.png")


if __name__ == "__main__":
    main()