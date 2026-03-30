from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from training.common import (
    MODELS_DIR,
    TABLES_DIR,
    MetricCaptureCallback,
    RunResult,
    ensure_dirs,
    make_env,
    save_run_registry,
    write_results_csv,
)


def dqn_hyperparameter_grid() -> List[Dict]:
    return [
        {"learning_rate": 1e-3, "gamma": 0.95, "buffer_size": 20_000, "batch_size": 64, "exploration_fraction": 0.20, "target_update_interval": 500},
        {"learning_rate": 7e-4, "gamma": 0.97, "buffer_size": 30_000, "batch_size": 64, "exploration_fraction": 0.25, "target_update_interval": 500},
        {"learning_rate": 5e-4, "gamma": 0.99, "buffer_size": 50_000, "batch_size": 64, "exploration_fraction": 0.15, "target_update_interval": 750},
        {"learning_rate": 3e-4, "gamma": 0.99, "buffer_size": 80_000, "batch_size": 128, "exploration_fraction": 0.12, "target_update_interval": 1_000},
        {"learning_rate": 2e-4, "gamma": 0.98, "buffer_size": 60_000, "batch_size": 128, "exploration_fraction": 0.10, "target_update_interval": 750},
        {"learning_rate": 1e-4, "gamma": 0.995, "buffer_size": 100_000, "batch_size": 128, "exploration_fraction": 0.08, "target_update_interval": 1_500},
        {"learning_rate": 8e-4, "gamma": 0.96, "buffer_size": 40_000, "batch_size": 64, "exploration_fraction": 0.18, "target_update_interval": 500},
        {"learning_rate": 6e-4, "gamma": 0.985, "buffer_size": 70_000, "batch_size": 256, "exploration_fraction": 0.10, "target_update_interval": 2_000},
        {"learning_rate": 2.5e-4, "gamma": 0.992, "buffer_size": 120_000, "batch_size": 128, "exploration_fraction": 0.06, "target_update_interval": 2_000},
        {"learning_rate": 4e-4, "gamma": 0.975, "buffer_size": 90_000, "batch_size": 64, "exploration_fraction": 0.14, "target_update_interval": 1_000},
    ]


def run_dqn_sweep(total_timesteps: int = 80_000, base_seed: int = 100) -> None:
    ensure_dirs()
    grid = dqn_hyperparameter_grid()

    run_results: List[RunResult] = []
    registry = []

    for run_id, params in enumerate(grid, start=1):
        seed = base_seed + run_id
        monitor_path = TABLES_DIR / f"monitor_dqn_run_{run_id}.csv"
        model_path = MODELS_DIR / "dqn" / f"dqn_run_{run_id}"
        metrics_path = TABLES_DIR / f"metrics_dqn_run_{run_id}.csv"

        env = make_env(seed=seed, monitor_path=monitor_path)
        callback = MetricCaptureCallback(
            save_path=metrics_path,
            keys=["train/loss", "rollout/ep_rew_mean", "rollout/exploration_rate"],
        )

        model = DQN(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            seed=seed,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            buffer_size=params["buffer_size"],
            batch_size=params["batch_size"],
            exploration_fraction=params["exploration_fraction"],
            target_update_interval=params["target_update_interval"],
            tensorboard_log=str(TABLES_DIR / "tb_dqn"),
        )

        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
        model.save(str(model_path))

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

        print(
            f"[DQN][run={run_id}] mean_eval_reward={mean_reward:.3f} "
            f"std_eval_reward={std_reward:.3f}"
        )

        run_results.append(
            RunResult(
                algorithm="DQN",
                run_id=run_id,
                mean_eval_reward=float(mean_reward),
                std_eval_reward=float(std_reward),
                model_path=str(model_path),
                monitor_path=str(monitor_path),
                hyperparameters=params,
            )
        )

        registry.append(
            {
                "algorithm": "DQN",
                "run_id": run_id,
                "seed": seed,
                "model_path": str(model_path),
                "monitor_path": str(monitor_path),
                "metrics_path": str(metrics_path),
                "hyperparameters": params,
                "mean_eval_reward": float(mean_reward),
                "std_eval_reward": float(std_reward),
            }
        )
        env.close()

    result_csv = TABLES_DIR / "dqn_sweep_results.csv"
    write_results_csv(result_csv, run_results)
    save_run_registry(TABLES_DIR / "dqn_registry.json", registry)

    df = pd.read_csv(result_csv)
    best = df.sort_values("mean_eval_reward", ascending=False).iloc[0]

    best_metadata = {
        "algorithm": "DQN",
        "run_id": int(best["run_id"]),
        "model_path": best["model_path"],
        "mean_eval_reward": float(best["mean_eval_reward"]),
        "std_eval_reward": float(best["std_eval_reward"]),
    }
    (TABLES_DIR / "best_dqn.json").write_text(json.dumps(best_metadata, indent=2), encoding="utf-8")

    print("Saved DQN sweep results and best model metadata.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN hyperparameter sweep for posture environment")
    parser.add_argument("--timesteps", type=int, default=80_000, help="Training timesteps per run")
    parser.add_argument("--seed", type=int, default=100, help="Base seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dqn_sweep(total_timesteps=args.timesteps, base_seed=args.seed)
