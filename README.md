# PostureMonitor RL — Real-Time Office Posture Coaching Agent

### `erneste_ntezirizaza_rl_summative`

> **Capstone Project:** Real-Time Posture Monitoring and Correction System for Office Workers: A Machine Learning Approach to Preventing Musculoskeletal Disorders.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [Experiments Summary](#experiments-summary)
- [Visual Diagrams & Key Insights](#visual-diagrams--key-insights)
- [Discussion & Analysis](#discussion--analysis)
- [Environment Details](#environment-details)
- [Reward Structure](#reward-structure)
- [JSON API Export](#json-api-export)
- [Algorithm Summary](#algorithm-summary-best-runs)
- [REINFORCE: Pure PyTorch Implementation](#reinforce-pure-pytorch-implementation)
- [DQN: Stable-Baselines3 Implementation](#dqn-stable-baselines3-implementation)
- [PPO: Stable-Baselines3 Implementation](#ppo-stable-baselines3-implementation)
- [Hyperparameter Configurations](#hyperparameter-configurations-30-runs-total)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This project implements a Reinforcement Learning (RL) agent that acts as a **real-time posture coaching system** for office workers. The agent observes biomechanical posture metrics (head tilt, neck angle, back curvature, etc.) and learns the optimal intervention strategy — balancing corrective alerts against alert fatigue — to minimise musculoskeletal disorder (MSD) risk.

Three RL algorithms are compared:

- **DQN** (Deep Q-Network) — Value-Based (Stable-Baselines3)
- **REINFORCE** — Pure Policy Gradient, Williams (1992) (Custom PyTorch implementation)
- **PPO** (Proximal Policy Optimisation) — Advanced Policy Gradient (Stable-Baselines3)

---

## Project Structure

```
erneste_ntezirizaza_rl_summative/
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── play.py
├── generate_plots.py
├── environment/
│   ├── __init__.py
│   ├── custom_env.py
│   └── rendering.py
├── training/
│   ├── __init__.py
│   ├── dqn_training.py
│   └── pg_training.py
├── static/
│   └── static_demo.py
├── logs/
│   ├── api_export.json
│   ├── dqn/
│   └── pg/
├── models/
│   ├── dqn/
│   └── pg/
└── plots/
  ├── 01_env_architecture.png
  ├── 02_agent_env_loop.png
  ├── 03_reward_structure.png
  ├── 04_training_curves.png
  ├── 05_hp_heatmap.png
  ├── 06_convergence.png
  ├── 07_entropy_curves.png
  ├── 08_dqn_objective.png
  ├── 09_generalisation.png
  └── 10_algorithm_comparison.png
```

---

## Setup Instructions

### 1. Prerequisites

- Python **3.10 – 3.12** (recommended: 3.11)
- `pip` (package manager)
- A display (for Pygame GUI) — or use `--no-render` for headless mode

### 2. Clone the Repository

```bash
git clone https://github.com/ernesteNtezirizaza/erneste_ntezirizaza_rl_summative.git
cd erneste_ntezirizaza_rl_summative
```

### 3. Create a Virtual Environment (recommended)

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note for Apple Silicon (M1/M2/M3):** PyTorch installs correctly via pip. If you encounter issues, visit [pytorch.org](https://pytorch.org) for platform-specific instructions.

---

## Running the Project

### A — Random Action Demo (No Model Required)

Demonstrates the environment visualisation with a random agent. No training needed.

```bash
python static/static_demo.py
```

**What you'll see:** The Pygame window opens showing the office worker silhouette, real-time posture metric bars, a live reward chart, and random actions being taken each step.

---

### B — Train All Models

Training runs 10 hyperparameter combinations per algorithm (30 runs total). This may take **30–90 minutes** depending on your hardware.

```bash
# Train DQN (10 runs)
python training/dqn_training.py

# Train REINFORCE + PPO (10 runs each)
python training/pg_training.py
```

Results are saved to:

- `models/dqn/best_dqn_model.zip` + `models/dqn/best_dqn_model.json` (hyperparameters)
- `models/pg/best_reinforce_model.pt` + `models/pg/best_reinforce_model.json` (hyperparameters)
- `models/pg/best_ppo_model.zip` + `models/pg/best_ppo_model.json` (hyperparameters)
- `logs/dqn/dqn_hyperparameter_results.csv`
- `logs/pg/reinforce_results.csv`
- `logs/pg/ppo_results.csv`

---

### C — Run the Agent Behavior (Rubric: play.py)

```bash
# Auto-select best available model
python play.py

# Force a specific model
python play.py --model ppo
python play.py --model dqn
python play.py --model reinforce

# Run multiple episodes
python play.py --episodes 5

# Headless mode (no Pygame window)
python play.py --no-render

# Export episode data as JSON API payload
python play.py --export-json
```

`play.py` delegates to `main.py`, which contains the full model-loading and
agent-environment interaction loop.

Rubric note: System Implementation and Agent Behavior is demonstrated by
running `play.py`, where the trained policy selects posture-coaching actions
that align with the environment goal.

---

### D — Generate All Diagrams & Plots

```bash
python generate_plots.py
```

Saves 10 figures to `plots/`:

| File                          | Description                           |
| ----------------------------- | ------------------------------------- |
| `01_env_architecture.png`     | Full environment architecture diagram |
| `02_agent_env_loop.png`       | Agent–environment interaction loop    |
| `03_reward_structure.png`     | Reward structure overview             |
| `04_training_curves.png`      | Training reward curves (all methods)  |
| `05_hp_heatmap.png`           | Hyperparameter sensitivity heatmaps   |
| `06_convergence.png`          | Convergence comparison                |
| `07_entropy_curves.png`       | Policy entropy curves (PG methods)    |
| `08_dqn_objective.png`        | DQN loss / Q-value / ε curves         |
| `09_generalisation.png`       | Generalisation test across 20 seeds   |
| `10_algorithm_comparison.png` | Final algorithm comparison summary    |

---

## Experiments Summary

| Algorithm | Best Artifact                       | Best Run | Best Mean Reward ± Std | Mean of Mean Rewards | Key Result                                             |
| --------- | ----------------------------------- | -------- | ---------------------- | -------------------- | ------------------------------------------------------ |
| DQN       | `models/dqn/best_dqn_model.zip`     | 9        | 279.80 ± 4.35          | 147.56               | Strong sample efficiency and steady learning           |
| REINFORCE | `models/pg/best_reinforce_model.pt` | 1        | 247.30 ± 0.00          | 137.31               | Highest variance, but now fully custom and transparent |
| PPO       | `models/pg/best_ppo_model.zip`      | 6        | 276.51 ± 9.80          | 270.78               | Best overall stability and final performance           |

### Key Insights From the Experiments

- PPO achieved the strongest average performance across runs, with a mean-of-means score of 270.78.
- DQN produced the best single run in the sweep: run 9 reached 279.80 mean reward.
- Pure REINFORCE also produced a strong best run (247.30 mean reward), but it remained the most variance-sensitive algorithm.
- The best model metadata is stored alongside each checkpoint as a `.json` file containing the selected hyperparameters.

---

## Visual Diagrams & Key Insights

### 1. Environment Architecture

![Updated Environment Architecture](plots/01_env_architecture.png?raw=1)

The architecture diagram shows the full RL loop: observations flow into the agent, the agent selects a posture-coaching action, the environment applies the effect, and the reward signal closes the loop.

### 2. Agent-Environment Interaction

![Agent-Environment Loop](plots/02_agent_env_loop.png)

This diagram makes the timestep interaction explicit and is useful for explaining how state, action, and reward are exchanged at every step.

### 3. Reward Structure

![Reward Structure](plots/03_reward_structure.png)

The reward plot highlights the asymmetry in the design: corrective actions are rewarded positively, while ignored alerts and fatigue peaks are penalised more strongly.

### 4. Final Algorithm Comparison

![Algorithm Comparison](plots/10_algorithm_comparison.png)

This summary figure reflects the CSV results: PPO is the strongest average performer, DQN delivered the best single run, and REINFORCE remains the most variance-sensitive.

## Discussion & Analysis

This section connects the visuals to concrete learning behavior (stability, exploration/exploitation balance, and convergence).

### 1. Cumulative Reward Curves (All Methods)

Figure: `plots/04_training_curves.png`

- All three methods show upward learning trends, confirming that the environment reward signal is learnable.
- PPO achieves the most consistent high plateau over the final training window, which aligns with its best mean-of-means score (270.78 across 10 runs).
- DQN improves quickly in early timesteps (strong sample efficiency), but the spread across runs is wider than PPO (best run 279.80, yet mean-of-means 147.56).
- REINFORCE reaches competitive peaks in some runs but exhibits the largest instability across the sweep, reflected by lower aggregate mean-of-means (137.31).

### 2. DQN Objective Curves

Figure: `plots/08_dqn_objective.png`

- TD loss decreases over training, indicating Bellman residual reduction and improving value consistency.
- Q-value estimates rise while exploration rate $\epsilon$ decays, showing a standard shift from exploration to exploitation.
- The best DQN run reaches 279.80 ± 4.35, but non-best runs vary significantly (for example run 2: 249.95 ± 69.60), indicating sensitivity to hyperparameters.

### 3. Policy Entropy Curves (PG Methods)

Figure: `plots/07_entropy_curves.png`

- Entropy starts high and declines over time, which is expected as policies become more confident.
- This demonstrates healthy exploration/exploitation transition: broad early action sampling, then sharper action preferences near convergence.
- For REINFORCE, per-episode entropy logs confirm this collapse toward deterministic behavior (run-1 entropy mean 0.0143 with final entropy near 0).

### 4. Convergence Comparison

Figure: `plots/06_convergence.png`

- PPO converges fastest and sustains high reward more reliably in the final phase.
- DQN converges more gradually but remains competitive in best-run performance.
- REINFORCE converges slower and with larger oscillations, consistent with higher-variance Monte-Carlo gradient estimates.
- The convergence ordering in the plot is consistent with aggregate performance: PPO > DQN > REINFORCE (by mean-of-means).

### 5. Hyperparameter Sensitivity

Figure: `plots/05_hp_heatmap.png`

- Performance is not uniform across $(\text{LR}, \gamma)$ settings; there are clear high-performing regions.
- DQN shows sharper sensitivity, where small changes in exploration schedule and buffer/batch settings can produce large return differences.
- PPO displays broader robust regions, which helps explain stronger average performance across 10 runs.

### 6. Generalization Across Unseen Seeds

Figure: `plots/09_generalisation.png`

- PPO maintains the highest and most stable mean reward over unseen seeds, supporting stronger policy robustness.
- DQN generalizes reasonably but with wider spread than PPO.
- REINFORCE exhibits the widest variability under seed shift, reinforcing the variance findings from training and entropy behavior.

### 7. Final Comparative Interpretation

Figure: `plots/10_algorithm_comparison.png`

- Best single run: DQN (279.80 ± 4.35).
- Best average across full sweep: PPO (270.78 mean-of-means).
- REINFORCE achieved its top score in multiple runs (247.30), but with weaker overall consistency.
- Practical conclusion for deployment: PPO is the strongest default policy due to stability + robust cross-run behavior; DQN is a strong alternative when aggressively tuned; REINFORCE is most valuable for algorithmic transparency and educational interpretability.

---

## Environment Details

### Observation Space (8 continuous values)

| Index | Feature          | Range        | Ideal Range  |
| ----- | ---------------- | ------------ | ------------ |
| 0     | Head Tilt        | [-30°, +30°] | [-15°, +15°] |
| 1     | Neck Angle       | [-45°, +10°] | [-15°, 0°]   |
| 2     | Shoulder Drop    | [0, 10]      | [0, 5]       |
| 3     | Back Curvature   | [0°, 45°]    | [0°, 20°]    |
| 4     | Screen Distance  | [30, 90 cm]  | [50, 70 cm]  |
| 5     | Sitting Duration | [0, 120 min] | [0, 45 min]  |
| 6     | Fatigue Level    | [0.0, 1.0]   | [0.0, 0.6]   |
| 7     | Alerts Ignored   | [0, 5]       | [0, 3]       |

### Action Space (6 discrete actions)

| Action | Description                        |
| ------ | ---------------------------------- |
| 0      | Do Nothing                         |
| 1      | Send Gentle Alert                  |
| 2      | Send Urgent Alert                  |
| 3      | Send Break Reminder                |
| 4      | Prompt Stretch Exercise            |
| 5      | Suggest Screen Distance Adjustment |

### Terminal Conditions

- Fatigue level reaches **1.0** (MSD risk threshold exceeded)
- Episode length reaches **200 steps** (truncation)

---

## Reward Structure

| Event                                      | Reward |
| ------------------------------------------ | ------ |
| Worker corrects posture after gentle alert | +2.0   |
| Worker corrects posture after urgent alert | +2.5   |
| Good posture, no action needed             | +1.0   |
| Break reminder given on time               | +2.0   |
| Sustained good posture (per 10 steps)      | +0.3   |
| Premature break reminder                   | -0.3   |
| Unnecessary alert (posture fine)           | -0.5   |
| Alert ignored (gentle)                     | -1.0   |
| Alert ignored (urgent)                     | -1.5   |
| Fatigue peak reached                       | -2.0   |

---

## JSON API Export

The `--export-json` flag serialises episode trajectories as a structured JSON payload (`logs/api_export.json`), demonstrating how this RL agent can be integrated into a **web or mobile application backend**:

```bash
python main.py --export-json
```

The output JSON contains per-step observations, actions, and rewards — ready to be consumed by a REST API frontend.

---

## Algorithm Summary (Best Runs)

| Algorithm | Best Run | Best Mean Reward ± Std | Mean of Mean Rewards | Notes                                                    |
| --------- | -------- | ---------------------- | -------------------- | -------------------------------------------------------- |
| DQN       | 9        | 279.80 ± 4.35          | 147.56               | Best single run in the DQN sweep                         |
| REINFORCE | 1        | 247.30 ± 0.00          | 137.31               | Strong best run, but higher variability across the sweep |
| PPO       | 6        | 276.51 ± 9.80          | 270.78               | Best average performer across all runs                   |

---

## REINFORCE: Pure PyTorch Implementation

REINFORCE is now implemented from scratch using pure PyTorch — **no Stable-Baselines3 wrapper, no actor-critic architecture, no shared value head.**

This is the genuine **Williams (1992) Monte-Carlo Policy Gradient** algorithm:

### Core Algorithm (Plain Policy Gradient)

```
Per Episode:
  1. Roll out complete episode trajectory τ = (s₀, a₀, r₁, s₁, a₁, r₂, ..., s_T)
  2. Compute discounted Monte-Carlo returns:
     G_t = Σ_{k=0}^{T-t} γᵏ · rₜ₊ₖ
  3. Optional baseline subtraction (b = mean(G_t)) to reduce variance
  4. Policy loss: L = -Σ_t G_t · log π(aₜ|sₜ) + λ_ent · H[π(·|s_t)]
  5. Update: θ ← θ + α∇_θ L  (via Adam optimizer with gradient clipping)
```

### Implementation Details

**`training/pg_training.py` — Key Components:**

#### `PolicyNetwork` (No Critic Head)

```python
class PolicyNetwork(nn.Module):
    """
    Stochastic policy π_θ(a|s) outputting softmax probabilities.
    Pure feedforward network — NO value head, NO critic architecture.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: list):
        # Simple MLP: obs → hidden layers → 6 action logits → softmax
```

#### `REINFORCETrainer` (Full Algorithm Implementation)

- **`collect_episode()`**: Rolls out complete trajectory, computes Monte-Carlo returns
- **`update()`**: Applies policy gradient with entropy regularisation
- **`train()`**: Main training loop with **per-episode logging** for visibility
- **`save()`**: PyTorch checkpoint (.pt) containing policy state + training history

### Training Outputs

- **Best model**: `models/pg/best_reinforce_model.pt` (PyTorch checkpoint)
- **Metadata**: `models/pg/best_reinforce_model.json` (hyperparameters + performance)
- **Results table**: `logs/pg/reinforce_results.csv` (all 10 runs)
- **Logs**: Timestep logging + training curves saved to `logs/pg/`

### Why Pure REINFORCE?

1. **Authentic algorithm** — Williams' original formulation, not approximated via actor-critic
2. **Lower variance baseline** — Explicit mean return subtraction (optional per hyperparameter)
3. **Gradient clipping** — Prevents unstable updates typical of high-variance policy gradients
4. **Full transparency** — All components visible and customisable within ~200 lines of code

## DQN: Stable-Baselines3 Implementation

DQN is implemented with Stable-Baselines3 as the value-based baseline for the project. The training script applies the algorithm directly to `PostureMonitorEnv` with an MLP policy and a compact evaluation/logging loop.

### Core Implementation

**`training/dqn_training.py` — Key Components:**

- **`HP_GRID`**: 10 hyperparameter combinations covering learning rate, gamma, replay buffer size, batch size, target-update `tau`, and exploration schedule
- **`make_env()`**: Wraps `PostureMonitorEnv` in `Monitor` for episode tracking
- **`DQN(...)`**: Uses `MlpPolicy` with `policy_kwargs=dict(net_arch=...)`
- **`EvalCallback`**: Evaluates every 5,000 timesteps on a separate seeded environment
- **`TimestepLoggingCallback`**: Prints step-level reward and episode return during training
- **`model.learn(...)`**: Trains for `60,000` timesteps per run on CPU
- **`model.save(...)`**: Stores each run as a `.zip` checkpoint, then copies the best run to `best_dqn_model.zip`

### Training Outputs

- **Best model**: `models/dqn/best_dqn_model.zip`
- **Metadata**: `models/dqn/best_dqn_model.json`
- **Results table**: `logs/dqn/dqn_hyperparameter_results.csv`
- **Logs**: Per-run evaluation logs under `logs/dqn/run_*/`

### Why DQN Here?

1. **Discrete action fit** — The posture coaching task has a small discrete action space, which suits DQN well
2. **Sample efficiency** — Replay buffer and target network updates help stabilise value learning
3. **Comparable baseline** — Provides a strong off-policy benchmark against policy-gradient methods

## PPO: Stable-Baselines3 Implementation

PPO is implemented with Stable-Baselines3 as the clipped policy-gradient baseline. It uses separate actor and critic network branches with matched architectures and the standard PPO training loop.

### Core Implementation

**`training/pg_training.py` — PPO Components:**

- **`PPO_GRID`**: 10 hyperparameter combinations covering learning rate, gamma, entropy coefficient, rollout length, batch size, epoch count, clip range, GAE lambda, and network size
- **`make_env()`**: Wraps `PostureMonitorEnv` in `Monitor` for evaluation consistency
- **`PPO(...)`**: Uses `MlpPolicy` with `policy_kwargs=dict(net_arch=dict(pi=..., vf=...))`
- **`EvalCallback`**: Evaluates every 5,000 timesteps on a separate seeded environment
- **`PPOTimestepLoggingCallback`**: Reports training progress at fixed timestep intervals
- **`model.learn(...)`**: Trains for `60,000` timesteps per run on CPU
- **`model.save(...)`**: Stores each run as a `.zip` checkpoint, then copies the best run to `best_ppo_model.zip`

### Training Outputs

- **Best model**: `models/pg/best_ppo_model.zip`
- **Metadata**: `models/pg/best_ppo_model.json`
- **Results table**: `logs/pg/ppo_results.csv`
- **Logs**: Per-run evaluation logs under `logs/pg/ppo_run_*/`

### Why PPO Here?

1. **Stable policy updates** — Clipped objective reduces destructive policy jumps
2. **Strong final performance** — In this project, PPO gave the best average score across runs
3. **Good variance control** — GAE, entropy bonus, and separate actor/critic heads improve stability

### main.py Model Loading Update

The model loader now correctly handles **three distinct model formats:**

| Algorithm | Format             | Extension | Loader         |
| --------- | ------------------ | --------- | -------------- |
| DQN       | Stable-Baselines3  | `.zip`    | `DQN.load()`   |
| REINFORCE | PyTorch checkpoint | `.pt`     | `torch.load()` |
| PPO       | Stable-Baselines3  | `.zip`    | `PPO.load()`   |

Architecture is automatically inferred from the checkpoint metadata.

## Hyperparameter Configurations (30 runs total)

The table below combines the full REINFORCE, DQN, and PPO grids used in the project.

| Algorithm | Run | LR   | γ    | Key Hyperparameters                                               | Network        |
| --------- | --- | ---- | ---- | ----------------------------------------------------------------- | -------------- |
| REINFORCE | 1   | 1e-3 | 0.99 | Ent=0.01; Baseline=Yes                                            | [64, 64]       |
| REINFORCE | 2   | 5e-4 | 0.99 | Ent=0.05; Baseline=Yes                                            | [128, 128]     |
| REINFORCE | 3   | 2e-4 | 0.95 | Ent=0.01; Baseline=No                                             | [64, 64]       |
| REINFORCE | 4   | 1e-3 | 0.98 | Ent=0.10; Baseline=Yes                                            | [256, 128]     |
| REINFORCE | 5   | 3e-4 | 0.99 | Ent=0.02; Baseline=Yes                                            | [128, 64]      |
| REINFORCE | 6   | 7e-4 | 0.90 | Ent=0.00; Baseline=No                                             | [64, 64]       |
| REINFORCE | 7   | 1e-4 | 0.99 | Ent=0.05; Baseline=Yes                                            | [256, 256]     |
| REINFORCE | 8   | 5e-3 | 0.97 | Ent=0.02; Baseline=Yes                                            | [128, 128, 64] |
| REINFORCE | 9   | 2e-4 | 0.99 | Ent=0.001; Baseline=No                                            | [512, 256]     |
| REINFORCE | 10  | 8e-4 | 0.98 | Ent=0.03; Baseline=Yes                                            | [256, 128, 64] |
| DQN       | 1   | 1e-3 | 0.99 | Buffer=10000; Batch=64; Tau=1.0; ε 1.00→0.05; Explo=0.20          | [64, 64]       |
| DQN       | 2   | 5e-4 | 0.99 | Buffer=20000; Batch=128; Tau=0.9; ε 1.00→0.05; Explo=0.30         | [128, 128]     |
| DQN       | 3   | 1e-4 | 0.95 | Buffer=10000; Batch=64; Tau=1.0; ε 1.00→0.02; Explo=0.40          | [64, 64]       |
| DQN       | 4   | 1e-3 | 0.98 | Buffer=50000; Batch=256; Tau=0.5; ε 1.00→0.01; Explo=0.20         | [256, 128]     |
| DQN       | 5   | 2e-4 | 0.99 | Buffer=30000; Batch=128; Tau=0.8; ε 0.80→0.05; Explo=0.50         | [128, 64]      |
| DQN       | 6   | 5e-3 | 0.90 | Buffer=10000; Batch=32; Tau=1.0; ε 1.00→0.10; Explo=0.10          | [64, 64]       |
| DQN       | 7   | 3e-4 | 0.99 | Buffer=20000; Batch=64; Tau=0.7; ε 1.00→0.05; Explo=0.30          | [256, 256]     |
| DQN       | 8   | 1e-3 | 0.97 | Buffer=10000; Batch=128; Tau=0.9; ε 1.00→0.02; Explo=0.20         | [128, 128, 64] |
| DQN       | 9   | 1e-4 | 0.99 | Buffer=50000; Batch=256; Tau=0.3; ε 1.00→0.01; Explo=0.60         | [512, 256]     |
| DQN       | 10  | 7e-4 | 0.98 | Buffer=30000; Batch=128; Tau=0.6; ε 0.90→0.05; Explo=0.35         | [256, 128, 64] |
| PPO       | 1   | 3e-4 | 0.99 | Ent=0.01; n_steps=512; Batch=64; Epochs=10; Clip=0.2; GAE=0.95    | [64, 64]       |
| PPO       | 2   | 1e-4 | 0.99 | Ent=0.05; n_steps=1024; Batch=128; Epochs=5; Clip=0.2; GAE=0.95   | [128, 128]     |
| PPO       | 3   | 5e-4 | 0.95 | Ent=0.01; n_steps=256; Batch=64; Epochs=10; Clip=0.1; GAE=0.90    | [64, 64]       |
| PPO       | 4   | 3e-4 | 0.98 | Ent=0.10; n_steps=2048; Batch=256; Epochs=20; Clip=0.3; GAE=0.98  | [256, 128]     |
| PPO       | 5   | 7e-4 | 0.99 | Ent=0.02; n_steps=512; Batch=128; Epochs=10; Clip=0.2; GAE=0.95   | [128, 64]      |
| PPO       | 6   | 1e-3 | 0.90 | Ent=0.00; n_steps=128; Batch=32; Epochs=5; Clip=0.2; GAE=0.80     | [64, 64]       |
| PPO       | 7   | 2e-4 | 0.99 | Ent=0.05; n_steps=1024; Batch=64; Epochs=15; Clip=0.15; GAE=0.95  | [256, 256]     |
| PPO       | 8   | 5e-4 | 0.97 | Ent=0.02; n_steps=512; Batch=128; Epochs=10; Clip=0.25; GAE=0.92  | [128, 128, 64] |
| PPO       | 9   | 1e-4 | 0.99 | Ent=0.001; n_steps=2048; Batch=256; Epochs=10; Clip=0.2; GAE=0.99 | [512, 256]     |
| PPO       | 10  | 4e-4 | 0.98 | Ent=0.03; n_steps=1024; Batch=128; Epochs=12; Clip=0.2; GAE=0.95  | [256, 128, 64] |

Full grids: `REINFORCE_GRID`, `HP_GRID`, and `PPO_GRID` in the training scripts.

---

## Troubleshooting

**Pygame display error on headless server:**

```bash
python main.py --no-render
```

**CUDA/GPU not available:**
All models default to `device="cpu"` — no GPU required.

**ModuleNotFoundError:**
Ensure you activated the virtual environment and ran `pip install -r requirements.txt`.

---

## License

Academic use only — Summative Assignment submission.
