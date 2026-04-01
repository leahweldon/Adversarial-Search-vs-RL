# CS7IS2 Assignment 3 — Adversarial Search and Reinforcement Learning

This project implements and compares adversarial search and reinforcement learning approaches for two-player games: Tic Tac Toe and Connect 4.

The goal is to evaluate how classical search methods (Minimax and Alpha-Beta pruning) compare with learning-based methods (Q-learning and Deep Q-Networks) as game complexity increases.

Developed for CS7IS2 Artificial Intelligence at Trinity College Dublin.

---

## Algorithms

| Algorithm | Tic Tac Toe | Connect 4 |
|----------|------------|-----------|
| Minimax | Full-depth search | Depth-limited (d=3) |
| Alpha-Beta Pruning | Full-depth search | Depth-limited (d=3, d=4) |
| Tabular Q-Learning | Trained vs default opponent | Trained vs random opponent |
| Deep Q-Network (DQN) | Trained vs default opponent | Trained vs random opponent |

---

## Key Findings

- Minimax and Alpha-Beta achieve optimal play in Tic Tac Toe (100% draw rate)
- Alpha-Beta reduces computation by approximately 96% compared to Minimax
- Full-depth Connect 4 search is infeasible (confirmed experimentally)
- Depth-limited search performs strongly in Connect 4 (up to 100% win rate vs default opponent)
- Reinforcement learning performs well against weaker opponents but struggles against search-based agents
- DQN generalises better than tabular Q-learning in small state spaces, but not in larger ones

---

## Project Structure

```
src/
├── agents/          # Minimax, Q-learning, DQN agents
├── games/           # Game environments (Tic Tac Toe, Connect 4)
├── training/        # RL training logic
├── evaluation/      # Matchups and metrics
└── utils/           # Plotting and file handling

experiments/         # Scripts to run all experiments
results/             # Models, CSV outputs, and figures
```

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Python 3.12 required.

---

## Running Experiments

All commands should be run from the project root.

### Search

```bash
python -m experiments.run_minimax_tictactoe
python -m experiments.run_minimax_connect4
python -m experiments.run_connect4_full_probe   # 30-minute feasibility test
```

### Reinforcement Learning

```bash
python -m experiments.run_q_learning_tictactoe
python -m experiments.run_q_learning_connect4
python -m experiments.run_dqn_tictactoe
python -m experiments.run_dqn_connect4
```

### Head-to-Head Evaluation

```bash
python -m experiments.run_head_to_head_final
```

### Plot Generation

```bash
python -m experiments.plot_report_figures
```

---

## Results

- All experiments use a fixed random seed (42) for reproducibility
- Pre-trained models are included in `results/models/`
- Full results are stored as CSV files and visualised in the report

---

## Notes

- Connect 4 full-depth Minimax was tested for 30 minutes and confirmed to be infeasible
- Depth-limited search with a heuristic evaluation function was used instead
- RL agents for Connect 4 are trained against a random opponent to improve learning stability
- Training each RL agent takes approximately 10–20 minutes on CPU

---

## Author

Leah Weldon  
MSc Computer Science  
Trinity College Dublin