"""
dqn_training.py
===============
Trains a DQN agent on the PostureMonitorEnv using Stable-Baselines3.
Runs 10 hyperparameter combinations and saves results + best model.

Usage:
    python training/dqn_training.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import PostureMonitorEnv

MODELS_DIR = "models/dqn"
PLOTS_DIR  = "plots"
LOGS_DIR   = "logs/dqn"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

TOTAL_TIMESTEPS = 60_000   # per run

# ──────────────────────────────────────────────────────────────────
# 10 HYPERPARAMETER COMBINATIONS
# ──────────────────────────────────────────────────────────────────
HP_GRID = [
    # run  lr       gamma  buffer  batch  tau    explo_frac  explo_eps_start  explo_eps_end  net_arch
    {"run": 1,  "lr": 1e-3,  "gamma": 0.99, "buffer": 10000, "batch": 64,  "tau": 1.0,  "ef": 0.2,  "es": 1.0,  "ee": 0.05, "net": [64, 64]},
    {"run": 2,  "lr": 5e-4,  "gamma": 0.99, "buffer": 20000, "batch": 128, "tau": 0.9,  "ef": 0.3,  "es": 1.0,  "ee": 0.05, "net": [128, 128]},
    {"run": 3,  "lr": 1e-4,  "gamma": 0.95, "buffer": 10000, "batch": 64,  "tau": 1.0,  "ef": 0.4,  "es": 1.0,  "ee": 0.02, "net": [64, 64]},
    {"run": 4,  "lr": 1e-3,  "gamma": 0.98, "buffer": 50000, "batch": 256, "tau": 0.5,  "ef": 0.2,  "es": 1.0,  "ee": 0.01, "net": [256, 128]},
    {"run": 5,  "lr": 2e-4,  "gamma": 0.99, "buffer": 30000, "batch": 128, "tau": 0.8,  "ef": 0.5,  "es": 0.8,  "ee": 0.05, "net": [128, 64]},
    {"run": 6,  "lr": 5e-3,  "gamma": 0.90, "buffer": 10000, "batch": 32,  "tau": 1.0,  "ef": 0.1,  "es": 1.0,  "ee": 0.1,  "net": [64, 64]},
    {"run": 7,  "lr": 3e-4,  "gamma": 0.99, "buffer": 20000, "batch": 64,  "tau": 0.7,  "ef": 0.3,  "es": 1.0,  "ee": 0.05, "net": [256, 256]},
    {"run": 8,  "lr": 1e-3,  "gamma": 0.97, "buffer": 10000, "batch": 128, "tau": 0.9,  "ef": 0.2,  "es": 1.0,  "ee": 0.02, "net": [128, 128, 64]},
    {"run": 9,  "lr": 1e-4,  "gamma": 0.99, "buffer": 50000, "batch": 256, "tau": 0.3,  "ef": 0.6,  "es": 1.0,  "ee": 0.01, "net": [512, 256]},
    {"run": 10, "lr": 7e-4,  "gamma": 0.98, "buffer": 30000, "batch": 128, "tau": 0.6,  "ef": 0.35, "es": 0.9,  "ee": 0.05, "net": [256, 128, 64]},
]


# ──────────────────────────────────────────────────────────────────
# CUSTOM CALLBACK FOR TIMESTEP LOGGING
# ──────────────────────────────────────────────────────────────────

class TimestepLoggingCallback(BaseCallback):
    """
    Custom callback to log training progress at each timestep.
    Prints timestep, cumulative reward, and loss at regular intervals.
    """
    def __init__(self, log_every: int = 1):
        super().__init__()
        self.log_every = log_every
        self.current_ep_reward = 0.0
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        step_reward = 0.0
        if rewards is not None:
            step_reward = float(np.array(rewards).reshape(-1)[0])
            self.current_ep_reward += step_reward

        if self.num_timesteps % self.log_every == 0:
            print(
                f"    [Timestep {self.num_timesteps:6d}] "
                f"Step Reward: {step_reward:+6.3f} | "
                f"Episode Reward: {self.current_ep_reward:+8.3f}"
            )

        if dones is not None and bool(np.array(dones).reshape(-1)[0]):
            self.episode_count += 1
            print(
                f"    [Episode {self.episode_count:3d} End] "
                f"Return: {self.current_ep_reward:+8.3f}"
            )
            self.current_ep_reward = 0.0

        return True


def make_env(seed=0):
    env = PostureMonitorEnv()
    env = Monitor(env)
    return env


def train_dqn(hp: dict):
    run_id  = hp["run"]
    print(f"\n{'='*60}")
    print(f"DQN Run {run_id}/10 | LR={hp['lr']} | Gamma={hp['gamma']} | Buffer={hp['buffer']}")
    print(f"{'='*60}")

    env      = make_env(seed=run_id)
    eval_env = make_env(seed=run_id + 100)

    policy_kwargs = dict(net_arch=hp["net"])

    model = DQN(
        policy             = "MlpPolicy",
        env                = env,
        learning_rate      = hp["lr"],
        gamma              = hp["gamma"],
        buffer_size        = hp["buffer"],
        batch_size         = hp["batch"],
        tau                = hp["tau"],
        exploration_fraction      = hp["ef"],
        exploration_initial_eps   = hp["es"],
        exploration_final_eps     = hp["ee"],
        policy_kwargs      = policy_kwargs,
        verbose            = 0,
        tensorboard_log    = LOGS_DIR,
        device             = "cpu",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = None,  # Don't save intermediate best folders
        log_path             = f"{LOGS_DIR}/run_{run_id}",
        eval_freq            = 5000,
        n_eval_episodes      = 10,
        deterministic        = True,
        verbose              = 0,
    )
    
    timestep_cb = TimestepLoggingCallback(log_every=5000)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_cb, timestep_cb], progress_bar=False)

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    print(f"  → Mean Reward: {mean_r:.2f} ± {std_r:.2f}")

    model.save(f"{MODELS_DIR}/dqn_run_{run_id}")
    env.close()
    eval_env.close()
    return mean_r, std_r, hp


def main():
    results = []
    best_mean   = -np.inf
    best_run_id = 1
    best_hp     = None

    for hp in HP_GRID:
        mean_r, std_r, current_hp = train_dqn(hp)
        row = {**hp, "mean_reward": round(mean_r, 3), "std_reward": round(std_r, 3)}
        results.append(row)
        if mean_r > best_mean:
            best_mean   = mean_r
            best_run_id = hp["run"]
            best_hp     = current_hp

    # ── Save results table ────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(f"{LOGS_DIR}/dqn_hyperparameter_results.csv", index=False)
    print(f"\n[DQN] Best run: {best_run_id} with mean reward {best_mean:.2f}")

    # ── Copy best model with hyperparameters ───────────────────────
    import shutil
    src = f"{MODELS_DIR}/dqn_run_{best_run_id}.zip"
    dst = f"{MODELS_DIR}/best_dqn_model.zip"
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"[DQN] Best model saved to {dst}")
    
    # ── Save best model info with hyperparameters ──────────────────
    best_model_info = {
        "run_id": best_run_id,
        "mean_reward": best_mean,
        "model_path": dst,
        "hyperparameters": {
            "learning_rate": float(best_hp["lr"]),
            "gamma": float(best_hp["gamma"]),
            "buffer_size": int(best_hp["buffer"]),
            "batch_size": int(best_hp["batch"]),
            "tau": float(best_hp["tau"]),
            "exploration_fraction": float(best_hp["ef"]),
            "exploration_initial_eps": float(best_hp["es"]),
            "exploration_final_eps": float(best_hp["ee"]),
            "network_architecture": best_hp["net"],
        }
    }
    
    with open(f"{MODELS_DIR}/best_dqn_model.json", "w") as f:
        json.dump(best_model_info, f, indent=2)
    print(f"[DQN] Best model info saved to {MODELS_DIR}/best_dqn_model.json")

    # ── Plot reward vs run ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("DQN Hyperparameter Experiment Results", fontsize=14, fontweight="bold")

    runs   = [r["run"]         for r in results]
    means  = [r["mean_reward"] for r in results]
    stds   = [r["std_reward"]  for r in results]
    lrs    = [r["lr"]          for r in results]
    gammas = [r["gamma"]       for r in results]

    bars = axes[0].bar(runs, means, yerr=stds, color="#4a90d9", capsize=4,
                       edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Run #")
    axes[0].set_ylabel("Mean Reward (20 episodes)")
    axes[0].set_title("DQN Mean Reward per Hyperparameter Run")
    axes[0].set_xticks(runs)
    bars[best_run_id - 1].set_color("#f0a500")
    axes[0].axhline(max(means), color="orange", linestyle="--", linewidth=1, label="Best")
    axes[0].legend()

    sc = axes[1].scatter(lrs, means, c=gammas, cmap="viridis", s=100, edgecolors="white")
    for i, r in enumerate(results):
        axes[1].annotate(f"R{r['run']}", (r["lr"], r["mean_reward"]), fontsize=8,
                         xytext=(3, 3), textcoords="offset points")
    plt.colorbar(sc, ax=axes[1], label="Gamma (γ)")
    axes[1].set_xlabel("Learning Rate")
    axes[1].set_ylabel("Mean Reward")
    axes[1].set_title("LR vs Mean Reward (colour = γ)")
    axes[1].set_xscale("log")

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/dqn_hyperparameter_results.png", dpi=150)
    plt.close()
    print(f"[DQN] Plot saved to {PLOTS_DIR}/dqn_hyperparameter_results.png")

    # ── Print table ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("DQN HYPERPARAMETER TABLE")
    print("=" * 80)
    print(df[["run","lr","gamma","buffer","batch","tau","ef","ee","mean_reward","std_reward"]].to_string(index=False))


if __name__ == "__main__":
    main()
