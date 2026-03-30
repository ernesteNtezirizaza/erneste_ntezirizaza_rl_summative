# Real-Time Posture Monitoring RL Summative

This project implements a mission-based reinforcement learning system for office posture monitoring and correction.

It includes:

- A realistic custom Gymnasium environment for posture quality management.
- Visualization using Pygame.
- Three RL methods requested by your instruction (excluding A2C):
  - DQN (value-based)
  - REINFORCE (policy gradient)
  - PPO (policy gradient actor-critic style)
- Hyperparameter sweeps with 10 runs per algorithm.
- Plots and tables for report discussion.

## 1) Project Structure

```text
project_root/
├── environment/
│   ├── custom_env.py
│   ├── rendering.py
│   └── agent_diagram.md
├── training/
│   ├── dqn_training.py
│   ├── pg_training.py
│   ├── reinforce_agent.py
│   ├── analysis.py
│   └── common.py
├── models/
│   ├── dqn/
│   └── pg/
├── results/
│   ├── figures/
│   └── tables/
├── play_random.py
├── main.py
├── requirements.txt
└── README.md
```

## 2) Environment Definition (for report section)

### Mission

Keep posture healthy for a full office session while controlling fatigue and avoiding severe ergonomic degradation.

### Action Space (Discrete(8))

1. align_neck
2. relax_shoulders
3. lumbar_reset
4. monitor_prompt
5. micro_stretch_break
6. short_walk_break
7. breathing_reset
8. ignore_prompt

### Observation Space (Box with 8 features)

1. neck_quality
2. shoulder_quality
3. spine_quality
4. fatigue
5. desk_time_norm
6. break_pressure
7. recent_improvement
8. last_action_norm

### Reward Logic

- Positive reward for posture quality improvement.
- Additional reward for increasing ergonomic score.
- Fatigue penalty.
- Bonus for timely break actions when fatigue is high.
- Penalty when ignoring prompts under high break pressure.

### Start State

- Posture quality starts in moderate range (0.45 to 0.75).
- Low initial fatigue.

### Terminal Conditions

- Severe posture collapse (mean quality below threshold).
- Work session timeout (max_steps reached).

## 3) Step-by-Step Setup

### Step 1: Clone and enter repository

```bash
git https://github.com/ernesteNtezirizaza/erneste_ntezirizaza_rl_summative.git
cd erneste_ntezirizaza_rl_summative
```

### Step 2: Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify environment visualization with random actions

```bash
python play_random.py
```

This demonstrates the GUI and mission components before any training.

## 4) Training Workflows

### 4.1 Train DQN (10 hyperparameter runs)

```bash
python -m training.dqn_training --timesteps 80000 --seed 100
```

Outputs:

- models in `models/dqn/`
- run logs and summary in `results/tables/dqn_*`

### 4.2 Train PPO and REINFORCE (10 runs each)

```bash
python -m training.pg_training --algo both --timesteps 80000 --seed 300
```

You may run only one algorithm:

```bash
python -m training.pg_training --algo ppo --timesteps 80000 --seed 300
python -m training.pg_training --algo reinforce --seed 600
```

Outputs:

- models in `models/pg/`
- run logs and summary in `results/tables/`

## 5) Generate Report Figures and Tables

```bash
python -m training.analysis
```

Generated artifacts include:

- Cumulative reward curves
- DQN objective proxy curve (train/loss)
- Policy gradient entropy curve
- Convergence stability comparison
- Generalization summary CSV

## 6) Run Best Performing Agent (for demo video)

Auto-select best model by evaluation reward:

```bash
python main.py --episodes 1 --max-steps 500
```

Force a specific algorithm:

```bash
python main.py --algo dqn --episodes 1
python main.py --algo ppo --episodes 1
python main.py --algo reinforce --episodes 1
```

## 7) Hyperparameter Table Guidance (for final PDF)

Use the generated CSV files in `results/tables/`:

- `dqn_sweep_results.csv`
- `ppo_sweep_results.csv`
- `reinforce_sweep_results.csv`

Each already has 10 runs per algorithm with different hyperparameters.

For your report discussion, explain:

- learning_rate effects on stability and convergence speed
- gamma effects on short-term vs long-term strategy
- entropy settings and exploration behavior
- batch/buffer/steps effects on sample efficiency

## 8) Common Commands (Quick Reference)

```bash
# random visualization
python play_random.py

# DQN sweep
python -m training.dqn_training --timesteps 80000

# PPO + REINFORCE sweeps
python -m training.pg_training --algo both --timesteps 80000

# Generate figures/tables
python -m training.analysis

# Run best model in GUI
python main.py --episodes 1
```
