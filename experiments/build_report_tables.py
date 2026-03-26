from __future__ import annotations

import pandas as pd


def main() -> None:
    master = pd.read_csv("results/final/master_summary.csv")
    search = pd.read_csv("results/final/search_summary_results.csv")
    rl = pd.read_csv("results/final/rl_summary_results.csv")

    report_table = []

    for _, row in master.iterrows():
        report_table.append(
            {
                "Game": row["game"],
                "Algorithm": row["algorithm"],
                "Opponent": row["opponent"],
                "Depth Limit": row["depth_limit"],
                "Win Rate": row["win_rate"],
                "Draw Rate": row["draw_rate"],
                "Average Move Time": row["avg_move_time"],
                "Notes": row["notes"],
            }
        )

    report_df = pd.DataFrame(report_table)
    report_df.to_csv("results/final/report_main_table.csv", index=False)

    search_table = search.rename(
        columns={
            "game": "Game",
            "algorithm": "Algorithm",
            "depth_limit": "Depth Limit",
            "player1_win_rate": "Win Rate",
            "draw_rate": "Draw Rate",
            "player1_avg_move_time": "Average Move Time",
            "player1_nodes": "Nodes Visited",
            "player1_cutoffs": "Cutoffs",
            "player1_max_depth": "Max Depth",
        }
    )
    search_table.to_csv("results/final/report_search_table.csv", index=False)

    rl_table = rl.rename(
        columns={
            "game": "Game",
            "algorithm": "Algorithm",
            "opponent": "Opponent",
            "win_rate_mean": "Mean Win Rate",
            "win_rate_std": "Win Rate Std",
            "wins_mean": "Mean Wins",
            "losses_mean": "Mean Losses",
            "policy_size_mean": "Mean Policy Size",
        }
    )
    rl_table.to_csv("results/final/report_rl_table.csv", index=False)

    print("\nSaved report tables:")
    print("- results/final/report_main_table.csv")
    print("- results/final/report_search_table.csv")
    print("- results/final/report_rl_table.csv")


if __name__ == "__main__":
    main()