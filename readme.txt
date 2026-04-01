CS7IS2 Assignment 3 — Readme
Leah Weldon

All commands should be run from the project root directory.

----------------------------------------
REQUIREMENTS
----------------------------------------
Python 3.12

Install dependencies:
    pip install -r requirements.txt


----------------------------------------
SEARCH ALGORITHMS
----------------------------------------

Tic Tac Toe — Minimax and Alpha-Beta pruning:
    python -m experiments.run_minimax_tictactoe

Connect 4 — Minimax and Alpha-Beta pruning (depth-limited search):
    python -m experiments.run_minimax_connect4

Connect 4 — Full search feasibility probe (30-minute timeout):
    python -m experiments.run_connect4_full_probe


----------------------------------------
REINFORCEMENT LEARNING
----------------------------------------

Tic Tac Toe — Tabular Q-learning (training + evaluation):
    python -m experiments.run_q_learning_tictactoe

Connect 4 — Tabular Q-learning (training + evaluation):
    python -m experiments.run_q_learning_connect4

Tic Tac Toe — Deep Q-Network (DQN) (training + evaluation):
    python -m experiments.run_dqn_tictactoe

Connect 4 — Deep Q-Network (DQN) (training + evaluation):
    python -m experiments.run_dqn_connect4


----------------------------------------
HEAD-TO-HEAD EVALUATION
----------------------------------------

Run all algorithm pairings (requires trained models in results/models/):
    python -m experiments.run_head_to_head_final


----------------------------------------
PLOT GENERATION
----------------------------------------

Generate all figures used in the report:
    python -m experiments.plot_report_figures


----------------------------------------
NOTES
----------------------------------------

- All experiments use a fixed random seed (42) for reproducibility.
- Pre-trained models are included in results/models/ so reinforcement learning agents can be evaluated without retraining.
- Training reinforcement learning agents from scratch takes approximately 10–20 minutes per agent on CPU.
- The Connect 4 full search probe runs for up to 30 minutes per configuration and is included to demonstrate the intractability of exhaustive search.