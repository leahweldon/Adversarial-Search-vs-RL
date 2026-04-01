from __future__ import annotations

from src.agents.dqn_agent import DQNAgent
from src.agents.minimax_agent import MinimaxAgent
from src.agents.q_learning_agent import QLearningAgent
from src.evaluation.matchup_runner import run_matchup
from src.games.connect4 import Connect4
from src.games.tictactoe import TicTacToe
from src.utils.io_utils import save_dict_rows_to_csv

GAMES_PER_MATCHUP = 100


def load_ttt_agents():
    q_agent = QLearningAgent(name="tabular_q_learning", seed=42)
    q_agent.load("results/models/tictactoe_q_learning_seed42.json")
    q_agent.set_epsilon(0.0)

    dqn_agent = DQNAgent(name="dqn", state_size=10, action_size=9, seed=42)
    dqn_agent.load("results/models/tictactoe_dqn_seed42.pt")
    dqn_agent.set_epsilon(0.0)

    return q_agent, dqn_agent


def load_c4_agents():
    q_agent = QLearningAgent(name="tabular_q_learning", seed=42)
    q_agent.load("results/models/connect4_q_learning_seed42.json")
    q_agent.set_epsilon(0.0)

    dqn_agent = DQNAgent(name="dqn", state_size=43, action_size=7, seed=42)
    dqn_agent.load("results/models/connect4_dqn_seed42.pt")
    dqn_agent.set_epsilon(0.0)

    return q_agent, dqn_agent


def matchup_row(game, agent_a_name, agent_b_name, stats):
    return {
        "game": game,
        "matchup": f"{agent_a_name} vs {agent_b_name}",
        "games": stats.games,
        "agent_a_win_rate": stats.player1_win_rate,
        "agent_b_win_rate": stats.player2_win_rate,
        "draw_rate": stats.draw_rate,
        "agent_a_avg_move_time": stats.player1_avg_move_time,
        "agent_b_avg_move_time": stats.player2_avg_move_time,
    }


def run_tictactoe_matchups():
    minimax = MinimaxAgent(name="minimax", use_alpha_beta=False)
    alphabeta = MinimaxAgent(name="alpha_beta", use_alpha_beta=True)
    q_agent, dqn_agent = load_ttt_agents()

    matchups = [
        ("minimax", minimax, "alpha_beta", alphabeta),
        ("minimax", minimax, "tabular_q_learning", q_agent),
        ("minimax", minimax, "dqn", dqn_agent),
        ("alpha_beta", alphabeta, "tabular_q_learning", q_agent),
        ("alpha_beta", alphabeta, "dqn", dqn_agent),
        ("tabular_q_learning", q_agent, "dqn", dqn_agent),
    ]

    rows = []
    for a_name, a, b_name, b in matchups:
        print(f"  {a_name} vs {b_name}...")
        stats = run_matchup(TicTacToe, a, b, GAMES_PER_MATCHUP, alternate_starting_player=True)
        rows.append(matchup_row("tictactoe", a_name, b_name, stats))

    return rows


def run_connect4_matchups():
    minimax_d3 = MinimaxAgent(name="minimax_depth3", use_alpha_beta=False, depth_limit=3)
    alphabeta_d3 = MinimaxAgent(name="alpha_beta_depth3", use_alpha_beta=True, depth_limit=3)
    alphabeta_d4 = MinimaxAgent(name="alpha_beta_depth4", use_alpha_beta=True, depth_limit=4)
    q_agent, dqn_agent = load_c4_agents()

    matchups = [
        ("minimax_depth3", minimax_d3, "alpha_beta_depth3", alphabeta_d3),
        ("minimax_depth3", minimax_d3, "alpha_beta_depth4", alphabeta_d4),
        ("minimax_depth3", minimax_d3, "tabular_q_learning", q_agent),
        ("minimax_depth3", minimax_d3, "dqn", dqn_agent),
        ("alpha_beta_depth3", alphabeta_d3, "tabular_q_learning", q_agent),
        ("alpha_beta_depth3", alphabeta_d3, "dqn", dqn_agent),
        ("tabular_q_learning", q_agent, "dqn", dqn_agent),
    ]

    rows = []
    for a_name, a, b_name, b in matchups:
        print(f"  {a_name} vs {b_name}...")
        stats = run_matchup(Connect4, a, b, GAMES_PER_MATCHUP, alternate_starting_player=True)
        rows.append(matchup_row("connect4", a_name, b_name, stats))

    return rows


def main():
    print("Running Tictactoe head-to-head...")
    ttt_rows = run_tictactoe_matchups()

    print("Running Connect4 head-to-head...")
    c4_rows = run_connect4_matchups()

    save_dict_rows_to_csv(ttt_rows, "results/final/head_to_head_tictactoe.csv")
    save_dict_rows_to_csv(c4_rows, "results/final/head_to_head_connect4.csv")
    save_dict_rows_to_csv(ttt_rows + c4_rows, "results/final/head_to_head_master.csv")

    print("\nDone. Results saved to results/final/")


if __name__ == "__main__":
    main()
