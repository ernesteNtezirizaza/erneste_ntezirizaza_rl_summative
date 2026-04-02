"""
pg_training.py
==============
Trains a pure REINFORCE agent (implemented from scratch with PyTorch)
and a PPO agent (via Stable-Baselines3) on the PostureMonitorEnv.

REINFORCE Algorithm (Williams, 1992):
  - Collects full episode trajectories
  - Computes Monte-Carlo returns G_t = sum(gamma^k * r_{t+k})
  - Updates policy by gradient ascent: grad J = sum(G_t * grad log pi(a_t|s_t))
  - No value function / critic — pure policy gradient
  - Optional baseline (mean return subtraction) to reduce variance
  - NO A2C, NO Actor-Critic — vanilla REINFORCE only

Usage:
    python training/pg_training.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import PostureMonitorEnv

MODELS_DIR = "models/pg"
PLOTS_DIR  = "plots"
LOGS_DIR   = "logs/pg"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  PURE REINFORCE — Policy Network (no critic, no value head)
# ══════════════════════════════════════════════════════════════════

class PolicyNetwork(nn.Module):
    """
    Stochastic policy pi_theta(a|s) for the REINFORCE algorithm.
    A plain feedforward network mapping observations to a
    softmax probability distribution over actions.
    No value function or critic — this is vanilla REINFORCE.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: list):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)

    def get_distribution(self, obs: np.ndarray) -> Categorical:
        x     = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.forward(x).squeeze(0)
        return Categorical(probs)

    def select_action(self, obs: np.ndarray):
        dist     = self.get_distribution(obs)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


# ══════════════════════════════════════════════════════════════════
#  PURE REINFORCE — Trainer
# ══════════════════════════════════════════════════════════════════

class REINFORCETrainer:
    """
    Vanilla REINFORCE (Monte-Carlo Policy Gradient, Williams 1992).

    Algorithm per episode:
        1. Roll out full trajectory tau = (s0,a0,r1, s1,a1,r2, ..., s_T)
        2. Compute discounted returns: G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}
        3. Optionally subtract baseline b = mean(G_t) to reduce variance
        4. Policy loss = -sum_t  G_t * log pi(a_t | s_t)
        5. Add entropy bonus: -coef * H[pi(.|s_t)]  (encourages exploration)
        6. Backpropagate and update theta via Adam
    """

    def __init__(self, env, lr: float, gamma: float, hidden: list,
                 entropy_coef: float = 0.01, use_baseline: bool = True,
                 seed: int = 0):
        self.env          = env
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.use_baseline = use_baseline

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.policy    = PolicyNetwork(obs_dim, act_dim, hidden)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.episode_rewards   = []
        self.episode_lengths   = []
        self.policy_losses     = []
        self.entropy_history   = []

    # ── Roll out one complete episode ─────────────────────────────
    def collect_episode(self):
        obs, _    = self.env.reset()
        log_probs = []
        rewards   = []
        entropies = []
        done      = False

        while not done:
            action, log_prob = self.policy.select_action(obs)

            # Entropy for monitoring exploration breadth
            dist    = self.policy.get_distribution(obs)
            entropies.append(dist.entropy())

            obs, reward, terminated, truncated, _ = self.env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            done = terminated or truncated

        # ── Compute discounted Monte-Carlo returns G_t ─────────────
        G       = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.FloatTensor(returns)

        # ── Baseline: subtract mean return to reduce variance ───────
        if self.use_baseline:
            returns_t = returns_t - returns_t.mean()
            std = returns_t.std()
            if std > 1e-8:
                returns_t = returns_t / std

        return log_probs, returns_t, entropies, sum(rewards), len(rewards)

    # ── REINFORCE gradient update ─────────────────────────────────
    def update(self, log_probs, returns, entropies):
        # Core REINFORCE objective: maximise E[G_t * log pi(a_t|s_t)]
        policy_loss  = torch.stack(
            [-lp * G for lp, G in zip(log_probs, returns)]
        ).sum()

        # Entropy regularisation (keeps policy from collapsing prematurely)
        entropy_loss = -self.entropy_coef * torch.stack(entropies).mean()

        loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients — critical for REINFORCE stability
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item(), torch.stack(entropies).mean().item()

    # ── Full training loop ────────────────────────────────────────
    def train(self, num_episodes: int = 600, log_every: int = 50):
        for ep in range(1, num_episodes + 1):
            log_probs, returns, entropies, ep_reward, ep_len = self.collect_episode()
            loss, entropy = self.update(log_probs, returns, entropies)

            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_len)
            self.policy_losses.append(loss)
            self.entropy_history.append(entropy)

            # Log at every episode for visibility
            avg_r = np.mean(self.episode_rewards[-log_every:])
            if ep % log_every == 0:
                print(f"    Ep {ep:4d}/{num_episodes} | "
                      f"Avg Reward (last {log_every}): {avg_r:7.2f} | "
                      f"Episode Reward: {ep_reward:7.2f} | "
                      f"Entropy: {entropy:.4f} | "
                      f"Loss: {loss:.4f}")
            else:
                # Print every episode for detailed tracking
                print(f"    Ep {ep:4d}/{num_episodes} | "
                      f"Reward: {ep_reward:7.2f} | "
                      f"Length: {ep_len:3d} | "
                      f"Entropy: {entropy:.4f}")

        return np.mean(self.episode_rewards[-50:])

    # ── Greedy evaluation (deterministic) ─────────────────────────
    def evaluate(self, n_episodes: int = 20) -> tuple:
        rewards = []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            total  = 0.0
            done   = False
            while not done:
                x      = torch.FloatTensor(obs).unsqueeze(0)
                probs  = self.policy(x).squeeze(0)
                action = probs.argmax().item()   # argmax = deterministic
                obs, r, terminated, truncated, _ = self.env.step(action)
                total += r
                done   = terminated or truncated
            rewards.append(total)
        return float(np.mean(rewards)), float(np.std(rewards))

    # ── Save ──────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            "policy_state_dict":    self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode_rewards":      self.episode_rewards,
            "entropy_history":      self.entropy_history,
        }, path)
        print(f"    [REINFORCE] Saved → {path}")


# ══════════════════════════════════════════════════════════════════
#  REINFORCE — 10 HYPERPARAMETER COMBINATIONS
# ══════════════════════════════════════════════════════════════════

REINFORCE_GRID = [
    {"run": 1,  "lr": 1e-3,  "gamma": 0.99, "ent": 0.01,  "baseline": True,  "net": [64, 64]},
    {"run": 2,  "lr": 5e-4,  "gamma": 0.99, "ent": 0.05,  "baseline": True,  "net": [128, 128]},
    {"run": 3,  "lr": 2e-4,  "gamma": 0.95, "ent": 0.01,  "baseline": False, "net": [64, 64]},
    {"run": 4,  "lr": 1e-3,  "gamma": 0.98, "ent": 0.10,  "baseline": True,  "net": [256, 128]},
    {"run": 5,  "lr": 3e-4,  "gamma": 0.99, "ent": 0.02,  "baseline": True,  "net": [128, 64]},
    {"run": 6,  "lr": 7e-4,  "gamma": 0.90, "ent": 0.00,  "baseline": False, "net": [64, 64]},
    {"run": 7,  "lr": 1e-4,  "gamma": 0.99, "ent": 0.05,  "baseline": True,  "net": [256, 256]},
    {"run": 8,  "lr": 5e-3,  "gamma": 0.97, "ent": 0.02,  "baseline": True,  "net": [128, 128, 64]},
    {"run": 9,  "lr": 2e-4,  "gamma": 0.99, "ent": 0.001, "baseline": False, "net": [512, 256]},
    {"run": 10, "lr": 8e-4,  "gamma": 0.98, "ent": 0.03,  "baseline": True,  "net": [256, 128, 64]},
]

NUM_EPISODES_REINFORCE = 600


def save_reinforce_run_logs(run_id: int, hp: dict, trainer: REINFORCETrainer,
                            mean_r: float, std_r: float):
    """Save per-run REINFORCE logs under logs/pg, similar to DQN run logging."""
    history_df = pd.DataFrame({
        "episode": np.arange(1, len(trainer.episode_rewards) + 1),
        "episode_reward": trainer.episode_rewards,
        "episode_length": trainer.episode_lengths,
        "policy_loss": trainer.policy_losses,
        "entropy": trainer.entropy_history,
    })
    history_path = f"{LOGS_DIR}/reinforce_run_{run_id}_history.csv"
    history_df.to_csv(history_path, index=False)

    summary = {
        "run_id": run_id,
        "mean_reward": float(mean_r),
        "std_reward": float(std_r),
        "hyperparameters": hp,
        "history_path": history_path,
    }
    summary_path = f"{LOGS_DIR}/reinforce_run_{run_id}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"    [REINFORCE] Run logs saved → {history_path}")
    print(f"    [REINFORCE] Run summary saved → {summary_path}")


def train_reinforce(hp: dict):
    run_id = hp["run"]
    print(f"\n{'='*60}")
    print(f"REINFORCE Run {run_id}/10 | LR={hp['lr']} | gamma={hp['gamma']} | "
          f"Entropy={hp['ent']} | Baseline={'Yes' if hp['baseline'] else 'No'} | "
          f"Net={hp['net']}")
    print(f"{'='*60}")

    env = PostureMonitorEnv()

    trainer = REINFORCETrainer(
        env          = env,
        lr           = hp["lr"],
        gamma        = hp["gamma"],
        hidden       = hp["net"],
        entropy_coef = hp["ent"],
        use_baseline = hp["baseline"],
        seed         = run_id * 7,
    )

    trainer.train(num_episodes=NUM_EPISODES_REINFORCE)
    mean_r, std_r = trainer.evaluate(n_episodes=20)
    print(f"  Eval → Mean: {mean_r:.2f} | Std: {std_r:.2f}")

    save_reinforce_run_logs(run_id, hp, trainer, mean_r, std_r)
    trainer.save(f"{MODELS_DIR}/reinforce_run_{run_id}.pt")
    env.close()

    return mean_r, std_r, trainer.episode_rewards, trainer.entropy_history, hp


# ══════════════════════════════════════════════════════════════════
#  PPO — 10 HYPERPARAMETER COMBINATIONS (Stable-Baselines3)
# ══════════════════════════════════════════════════════════════════

PPO_GRID = [
    {"run": 1,  "lr": 3e-4, "gamma": 0.99, "ent": 0.01,  "n_steps": 512,  "batch": 64,  "n_epochs": 10, "clip": 0.2,  "gae": 0.95, "net": [64, 64]},
    {"run": 2,  "lr": 1e-4, "gamma": 0.99, "ent": 0.05,  "n_steps": 1024, "batch": 128, "n_epochs": 5,  "clip": 0.2,  "gae": 0.95, "net": [128, 128]},
    {"run": 3,  "lr": 5e-4, "gamma": 0.95, "ent": 0.01,  "n_steps": 256,  "batch": 64,  "n_epochs": 10, "clip": 0.1,  "gae": 0.90, "net": [64, 64]},
    {"run": 4,  "lr": 3e-4, "gamma": 0.98, "ent": 0.10,  "n_steps": 2048, "batch": 256, "n_epochs": 20, "clip": 0.3,  "gae": 0.98, "net": [256, 128]},
    {"run": 5,  "lr": 7e-4, "gamma": 0.99, "ent": 0.02,  "n_steps": 512,  "batch": 128, "n_epochs": 10, "clip": 0.2,  "gae": 0.95, "net": [128, 64]},
    {"run": 6,  "lr": 1e-3, "gamma": 0.90, "ent": 0.00,  "n_steps": 128,  "batch": 32,  "n_epochs": 5,  "clip": 0.2,  "gae": 0.80, "net": [64, 64]},
    {"run": 7,  "lr": 2e-4, "gamma": 0.99, "ent": 0.05,  "n_steps": 1024, "batch": 64,  "n_epochs": 15, "clip": 0.15, "gae": 0.95, "net": [256, 256]},
    {"run": 8,  "lr": 5e-4, "gamma": 0.97, "ent": 0.02,  "n_steps": 512,  "batch": 128, "n_epochs": 10, "clip": 0.25, "gae": 0.92, "net": [128, 128, 64]},
    {"run": 9,  "lr": 1e-4, "gamma": 0.99, "ent": 0.001, "n_steps": 2048, "batch": 256, "n_epochs": 10, "clip": 0.2,  "gae": 0.99, "net": [512, 256]},
    {"run": 10, "lr": 4e-4, "gamma": 0.98, "ent": 0.03,  "n_steps": 1024, "batch": 128, "n_epochs": 12, "clip": 0.2,  "gae": 0.95, "net": [256, 128, 64]},
]

TOTAL_TIMESTEPS_PPO = 60_000


# ──────────────────────────────────────────────────────────────────
# CUSTOM CALLBACK FOR PPO TIMESTEP LOGGING
# ──────────────────────────────────────────────────────────────────

class PPOTimestepLoggingCallback(BaseCallback):
    """
    Custom callback to log PPO training progress at each timestep.
    Prints timestep at regular intervals during training.
    """
    def __init__(self, log_every: int = 5000):
        super().__init__()
        self.log_every = log_every
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_every == 0:
            print(f"    [PPO Timestep {self.num_timesteps:6d}] Training progress...")
        return True


def make_env(seed=0):
    env = PostureMonitorEnv()
    env = Monitor(env)
    return env


def train_ppo(hp: dict):
    run_id = hp["run"]
    print(f"\n{'='*60}")
    print(f"PPO Run {run_id}/10 | LR={hp['lr']} | gamma={hp['gamma']} | "
          f"Clip={hp['clip']} | n_steps={hp['n_steps']} | n_epochs={hp['n_epochs']}")
    print(f"{'='*60}")

    env      = make_env(seed=run_id + 300)
    eval_env = make_env(seed=run_id + 400)

    policy_kwargs = dict(net_arch=dict(pi=hp["net"], vf=hp["net"]))

    model = PPO(
        policy          = "MlpPolicy",
        env             = env,
        learning_rate   = hp["lr"],
        gamma           = hp["gamma"],
        ent_coef        = hp["ent"],
        n_steps         = hp["n_steps"],
        batch_size      = hp["batch"],
        n_epochs        = hp["n_epochs"],
        clip_range      = hp["clip"],
        gae_lambda      = hp["gae"],
        policy_kwargs   = policy_kwargs,
        verbose         = 0,
        tensorboard_log = LOGS_DIR,
        device          = "cpu",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = None,  # Don't save intermediate best folders
        log_path             = f"{LOGS_DIR}/ppo_run_{run_id}",
        eval_freq            = 5000,
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 0,
    )
    
    timestep_cb = PPOTimestepLoggingCallback(log_every=5000)

    model.learn(total_timesteps=TOTAL_TIMESTEPS_PPO, callback=[eval_cb, timestep_cb], progress_bar=False)
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    print(f"  Eval → Mean: {mean_r:.2f} | Std: {std_r:.2f}")

    model.save(f"{MODELS_DIR}/ppo_run_{run_id}")
    env.close()
    eval_env.close()
    return mean_r, std_r, hp


# ══════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════

def smooth(y, w=15):
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


def plot_comparison(reinforce_results, ppo_results,
                    reinforce_reward_curves, reinforce_entropy_curves):
    C_REI = "#e07b39"
    C_PPO = "#5caa6f"
    C_BG  = "#f7f9fc"

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Policy Gradient — Pure REINFORCE vs PPO Hyperparameter Experiments",
                 fontsize=14, fontweight="bold")
    fig.patch.set_facecolor(C_BG)

    runs = list(range(1, 11))

    # ── REINFORCE reward bar ──────────────────────────────────────
    rmeans = [r["mean_reward"] for r in reinforce_results]
    rstds  = [r["std_reward"]  for r in reinforce_results]
    best_r = int(np.argmax(rmeans))
    bars   = axes[0][0].bar(runs, rmeans, yerr=rstds, color=C_REI,
                             capsize=4, edgecolor="white", alpha=0.85)
    bars[best_r].set_color("#f0a500")
    axes[0][0].set_title("Pure REINFORCE — Mean Reward per Hyperparameter Run")
    axes[0][0].set_xlabel("Run #"); axes[0][0].set_ylabel("Mean Reward")
    axes[0][0].set_xticks(runs); axes[0][0].grid(alpha=0.2, axis="y")
    axes[0][0].set_facecolor(C_BG)

    # ── PPO reward bar ────────────────────────────────────────────
    pmeans = [r["mean_reward"] for r in ppo_results]
    pstds  = [r["std_reward"]  for r in ppo_results]
    best_p = int(np.argmax(pmeans))
    bars2  = axes[0][1].bar(runs, pmeans, yerr=pstds, color=C_PPO,
                             capsize=4, edgecolor="white", alpha=0.85)
    bars2[best_p].set_color("#f0a500")
    axes[0][1].set_title("PPO — Mean Reward per Hyperparameter Run")
    axes[0][1].set_xlabel("Run #"); axes[0][1].set_ylabel("Mean Reward")
    axes[0][1].set_xticks(runs); axes[0][1].grid(alpha=0.2, axis="y")
    axes[0][1].set_facecolor(C_BG)

    # ── REINFORCE learning curves (top 3 runs) ────────────────────
    ax_lc = axes[1][0]
    ax_lc.set_facecolor(C_BG)
    sorted_runs = sorted(reinforce_results, key=lambda x: x["mean_reward"], reverse=True)[:3]
    colors_lc   = ["#e07b39", "#c45a1a", "#f5a070"]
    for i, row in enumerate(sorted_runs):
        rid   = row["run"] - 1
        curve = reinforce_reward_curves[rid]
        eps   = list(range(1, len(curve) + 1))
        hp    = REINFORCE_GRID[rid]
        ax_lc.plot(eps, smooth(curve, 20), color=colors_lc[i], linewidth=2,
                   label=f"Run {row['run']} (lr={hp['lr']}, γ={hp['gamma']}, baseline={'Y' if hp['baseline'] else 'N'})")
        ax_lc.plot(eps, curve, alpha=0.15, color=colors_lc[i], linewidth=0.6)
    ax_lc.set_title("REINFORCE — Training Reward Curves (Top 3 Runs)")
    ax_lc.set_xlabel("Episode"); ax_lc.set_ylabel("Episode Reward")
    ax_lc.legend(fontsize=8); ax_lc.grid(alpha=0.2)

    # ── Entropy coefficient vs reward scatter ─────────────────────
    ax_sc = axes[1][1]
    ax_sc.set_facecolor(C_BG)
    r_ents = [r["ent"] for r in reinforce_results]
    ax_sc.scatter(r_ents, rmeans, c=C_REI, s=90, edgecolors="white",
                  label="REINFORCE", zorder=3)
    for row in reinforce_results:
        ax_sc.annotate(f"R{row['run']}", (row["ent"], row["mean_reward"]),
                       fontsize=8, xytext=(3, 3), textcoords="offset points", color=C_REI)

    p_ents = [r["ent"] for r in ppo_results]
    ax_sc.scatter(p_ents, pmeans, c=C_PPO, s=90, edgecolors="white",
                  label="PPO", marker="D", zorder=3)
    for row in ppo_results:
        ax_sc.annotate(f"P{row['run']}", (row["ent"], row["mean_reward"]),
                       fontsize=8, xytext=(3, 3), textcoords="offset points", color=C_PPO)

    ax_sc.set_xlabel("Entropy Coefficient")
    ax_sc.set_ylabel("Mean Reward")
    ax_sc.set_title("Entropy Coefficient vs Mean Reward")
    ax_sc.legend(); ax_sc.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/pg_hyperparameter_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PG] Plot saved → {PLOTS_DIR}/pg_hyperparameter_results.png")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    # ── REINFORCE ──────────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  TRAINING PURE REINFORCE (PyTorch, from scratch)")
    print("  Algorithm: Monte-Carlo Policy Gradient (Williams 1992)")
    print("  NO A2C — NO Actor-Critic — purely REINFORCE")
    print("█" * 60)

    reinforce_results        = []
    reinforce_reward_curves  = []
    reinforce_entropy_curves = []
    best_reinforce_mean      = -np.inf
    best_reinforce_run_id    = 1
    best_reinforce_hp        = None

    for hp in REINFORCE_GRID:
        mean_r, std_r, ep_rewards, ent_hist, current_hp = train_reinforce(hp)
        row = {**hp, "mean_reward": round(mean_r, 3), "std_reward": round(std_r, 3)}
        reinforce_results.append(row)
        reinforce_reward_curves.append(ep_rewards)
        reinforce_entropy_curves.append(ent_hist)
        if mean_r > best_reinforce_mean:
            best_reinforce_mean   = mean_r
            best_reinforce_run_id = hp["run"]
            best_reinforce_hp     = current_hp

    # Copy best REINFORCE model and save with hyperparameters
    src_r = f"{MODELS_DIR}/reinforce_run_{best_reinforce_run_id}.pt"
    dst_r = f"{MODELS_DIR}/best_reinforce_model.pt"
    if os.path.exists(src_r):
        shutil.copy(src_r, dst_r)
        print(f"\n[REINFORCE] Best run {best_reinforce_run_id} → {dst_r}")
    
    # Save best REINFORCE model info with hyperparameters
    best_reinforce_info = {
        "run_id": best_reinforce_run_id,
        "mean_reward": best_reinforce_mean,
        "model_path": dst_r,
        "hyperparameters": {
            "learning_rate": float(best_reinforce_hp["lr"]),
            "gamma": float(best_reinforce_hp["gamma"]),
            "entropy_coefficient": float(best_reinforce_hp["ent"]),
            "use_baseline": bool(best_reinforce_hp["baseline"]),
            "network_architecture": best_reinforce_hp["net"],
        }
    }
    
    with open(f"{MODELS_DIR}/best_reinforce_model.json", "w") as f:
        json.dump(best_reinforce_info, f, indent=2)
    print(f"[REINFORCE] Best model info saved to {MODELS_DIR}/best_reinforce_model.json")

    # ── PPO ────────────────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  TRAINING PPO (Stable-Baselines3)")
    print("█" * 60)

    ppo_results   = []
    best_ppo_mean = -np.inf
    best_ppo_run  = 1
    best_ppo_hp   = None

    for hp in PPO_GRID:
        mean_r, std_r, current_hp = train_ppo(hp)
        row = {**hp, "mean_reward": round(mean_r, 3), "std_reward": round(std_r, 3)}
        ppo_results.append(row)
        if mean_r > best_ppo_mean:
            best_ppo_mean = mean_r
            best_ppo_run  = hp["run"]
            best_ppo_hp   = current_hp

    # Copy best PPO model and save with hyperparameters
    src_p = f"{MODELS_DIR}/ppo_run_{best_ppo_run}.zip"
    dst_p = f"{MODELS_DIR}/best_ppo_model.zip"
    if os.path.exists(src_p):
        shutil.copy(src_p, dst_p)
        print(f"\n[PPO] Best run {best_ppo_run} → {dst_p}")
    
    # Save best PPO model info with hyperparameters
    best_ppo_info = {
        "run_id": best_ppo_run,
        "mean_reward": best_ppo_mean,
        "model_path": dst_p,
        "hyperparameters": {
            "learning_rate": float(best_ppo_hp["lr"]),
            "gamma": float(best_ppo_hp["gamma"]),
            "entropy_coefficient": float(best_ppo_hp["ent"]),
            "n_steps": int(best_ppo_hp["n_steps"]),
            "batch_size": int(best_ppo_hp["batch"]),
            "n_epochs": int(best_ppo_hp["n_epochs"]),
            "clip_range": float(best_ppo_hp["clip"]),
            "gae_lambda": float(best_ppo_hp["gae"]),
            "network_architecture": best_ppo_hp["net"],
        }
    }
    
    with open(f"{MODELS_DIR}/best_ppo_model.json", "w") as f:
        json.dump(best_ppo_info, f, indent=2)
    print(f"[PPO] Best model info saved to {MODELS_DIR}/best_ppo_model.json")

    # ── Save CSVs ─────────────────────────────────────────────────
    pd.DataFrame(reinforce_results).to_csv(f"{LOGS_DIR}/reinforce_results.csv", index=False)
    pd.DataFrame(ppo_results).to_csv(f"{LOGS_DIR}/ppo_results.csv", index=False)

    # ── Save best-run metadata with hyperparameters ────────────────
    with open(f"{MODELS_DIR}/best_runs.json", "w") as f:
        json.dump({
            "reinforce": best_reinforce_info,
            "ppo":       best_ppo_info,
        }, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────
    plot_comparison(reinforce_results, ppo_results,
                    reinforce_reward_curves, reinforce_entropy_curves)

    # ── Print tables ──────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("REINFORCE HYPERPARAMETER TABLE (Pure Monte-Carlo Policy Gradient — No A2C)")
    print("=" * 95)
    df_r = pd.DataFrame(reinforce_results)
    print(df_r[["run", "lr", "gamma", "ent", "baseline", "net",
                "mean_reward", "std_reward"]].to_string(index=False))

    print("\n" + "=" * 95)
    print("PPO HYPERPARAMETER TABLE")
    print("=" * 95)
    df_p = pd.DataFrame(ppo_results)
    print(df_p[["run", "lr", "gamma", "ent", "n_steps", "batch",
                "n_epochs", "clip", "gae", "mean_reward", "std_reward"]].to_string(index=False))

    print(f"\n{'='*60}")
    print(f"  REINFORCE best → Run {best_reinforce_run_id}  |  {best_reinforce_mean:.2f}")
    print(f"  PPO       best → Run {best_ppo_run}            |  {best_ppo_mean:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
