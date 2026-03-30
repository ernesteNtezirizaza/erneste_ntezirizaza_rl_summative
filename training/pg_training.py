from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from stable_baselines3 import PPO
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
from training.reinforce_agent import ReinforceConfig, run_reinforce_experiment


def ppo_hyperparameter_grid() -> List[Dict]:
    return [
        {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 1024, "batch_size": 64, "ent_coef": 0.00, "clip_range": 0.20},
        {"learning_rate": 2e-4, "gamma": 0.99, "n_steps": 1024, "batch_size": 128, "ent_coef": 0.01, "clip_range": 0.20},
        {"learning_rate": 1e-4, "gamma": 0.995, "n_steps": 2048, "batch_size": 128, "ent_coef": 0.01, "clip_range": 0.15},
        {"learning_rate": 4e-4, "gamma": 0.98, "n_steps": 512, "batch_size": 64, "ent_coef": 0.02, "clip_range": 0.25},
        {"learning_rate": 5e-4, "gamma": 0.97, "n_steps": 512, "batch_size": 128, "ent_coef": 0.03, "clip_range": 0.30},
        {"learning_rate": 2.5e-4, "gamma": 0.995, "n_steps": 2048, "batch_size": 256, "ent_coef": 0.00, "clip_range": 0.12},
        {"learning_rate": 7e-4, "gamma": 0.96, "n_steps": 256, "batch_size": 64, "ent_coef": 0.04, "clip_range": 0.30},
        {"learning_rate": 3.5e-4, "gamma": 0.985, "n_steps": 1024, "batch_size": 256, "ent_coef": 0.005, "clip_range": 0.20},
        {"learning_rate": 1.5e-4, "gamma": 0.999, "n_steps": 2048, "batch_size": 128, "ent_coef": 0.008, "clip_range": 0.12},
        {"learning_rate": 6e-4, "gamma": 0.98, "n_steps": 768, "batch_size": 96, "ent_coef": 0.02, "clip_range": 0.22},
    ]


def reinforce_hyperparameter_grid() -> List[Dict]:
    return [
        {"learning_rate": 1e-3, "gamma": 0.95, "hidden_dim": 128, "entropy_coef": 0.010, "episodes": 260},
        {"learning_rate": 7e-4, "gamma": 0.97, "hidden_dim": 128, "entropy_coef": 0.008, "episodes": 260},
        {"learning_rate": 5e-4, "gamma": 0.99, "hidden_dim": 256, "entropy_coef": 0.006, "episodes": 300},
        {"learning_rate": 3e-4, "gamma": 0.995, "hidden_dim": 256, "entropy_coef": 0.004, "episodes": 320},
        {"learning_rate": 2e-4, "gamma": 0.98, "hidden_dim": 128, "entropy_coef": 0.012, "episodes": 280},
        {"learning_rate": 9e-4, "gamma": 0.96, "hidden_dim": 64, "entropy_coef": 0.020, "episodes": 250},
        {"learning_rate": 4e-4, "gamma": 0.985, "hidden_dim": 192, "entropy_coef": 0.006, "episodes": 300},
        {"learning_rate": 6e-4, "gamma": 0.975, "hidden_dim": 128, "entropy_coef": 0.010, "episodes": 280},
        {"learning_rate": 2.5e-4, "gamma": 0.999, "hidden_dim": 256, "entropy_coef": 0.002, "episodes": 340},
        {"learning_rate": 1.2e-3, "gamma": 0.94, "hidden_dim": 64, "entropy_coef": 0.025, "episodes": 240},
    ]


def run_ppo_sweep(total_timesteps: int = 80_000, base_seed: int = 300) -> None:
    ensure_dirs()
    grid = ppo_hyperparameter_grid()

    results: List[RunResult] = []
    registry = []

    for run_id, params in enumerate(grid, start=1):
        seed = base_seed + run_id
        monitor_path = TABLES_DIR / f"monitor_ppo_run_{run_id}.csv"
        model_path = MODELS_DIR / "pg" / f"ppo_run_{run_id}"
        metrics_path = TABLES_DIR / f"metrics_ppo_run_{run_id}.csv"

        env = make_env(seed=seed, monitor_path=monitor_path)
        callback = MetricCaptureCallback(
            save_path=metrics_path,
            keys=["train/entropy_loss", "train/value_loss", "rollout/ep_rew_mean"],
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            seed=seed,
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            ent_coef=params["ent_coef"],
            clip_range=params["clip_range"],
            tensorboard_log=str(TABLES_DIR / "tb_ppo"),
        )

        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
        model.save(str(model_path))

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

        print(
            f"[PPO][run={run_id}] mean_eval_reward={mean_reward:.3f} "
            f"std_eval_reward={std_reward:.3f}"
        )

        results.append(
            RunResult(
                algorithm="PPO",
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
                "algorithm": "PPO",
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

    result_csv = TABLES_DIR / "ppo_sweep_results.csv"
    write_results_csv(result_csv, results)
    save_run_registry(TABLES_DIR / "ppo_registry.json", registry)

    df = pd.read_csv(result_csv)
    best = df.sort_values("mean_eval_reward", ascending=False).iloc[0]
    best_metadata = {
        "algorithm": "PPO",
        "run_id": int(best["run_id"]),
        "model_path": best["model_path"],
        "mean_eval_reward": float(best["mean_eval_reward"]),
        "std_eval_reward": float(best["std_eval_reward"]),
    }
    (TABLES_DIR / "best_ppo.json").write_text(json.dumps(best_metadata, indent=2), encoding="utf-8")

    print("Saved PPO sweep results and best model metadata.")


def run_reinforce_sweep(base_seed: int = 600) -> None:
    ensure_dirs()
    grid = reinforce_hyperparameter_grid()

    records = []
    registry = []

    for run_id, params in enumerate(grid, start=1):
        seed = base_seed + run_id
        model_path = MODELS_DIR / "pg" / f"reinforce_run_{run_id}.pt"
        log_path = TABLES_DIR / f"metrics_reinforce_run_{run_id}.csv"

        cfg = ReinforceConfig(
            learning_rate=float(params["learning_rate"]),
            gamma=float(params["gamma"]),
            hidden_dim=int(params["hidden_dim"]),
            entropy_coef=float(params["entropy_coef"]),
            max_steps=480,
            episodes=int(params["episodes"]),
        )

        logs, mean_reward, std_reward = run_reinforce_experiment(cfg, seed=seed, model_path=model_path)

        pd.DataFrame(logs).to_csv(log_path, index=False)

        print(
            f"[REINFORCE][run={run_id}] mean_eval_reward={mean_reward:.3f} "
            f"std_eval_reward={std_reward:.3f}"
        )

        records.append(
            {
                "algorithm": "REINFORCE",
                "run_id": run_id,
                "mean_eval_reward": float(mean_reward),
                "std_eval_reward": float(std_reward),
                "model_path": str(model_path),
                "monitor_path": str(log_path),
                "hyperparameters": json.dumps(params),
            }
        )

        registry.append(
            {
                "algorithm": "REINFORCE",
                "run_id": run_id,
                "seed": seed,
                "model_path": str(model_path),
                "metrics_path": str(log_path),
                "hyperparameters": params,
                "mean_eval_reward": float(mean_reward),
                "std_eval_reward": float(std_reward),
            }
        )

    result_csv = TABLES_DIR / "reinforce_sweep_results.csv"
    pd.DataFrame(records).to_csv(result_csv, index=False)
    save_run_registry(TABLES_DIR / "reinforce_registry.json", registry)

    df = pd.read_csv(result_csv)
    best = df.sort_values("mean_eval_reward", ascending=False).iloc[0]
    best_metadata = {
        "algorithm": "REINFORCE",
        "run_id": int(best["run_id"]),
        "model_path": best["model_path"],
        "mean_eval_reward": float(best["mean_eval_reward"]),
        "std_eval_reward": float(best["std_eval_reward"]),
    }
    (TABLES_DIR / "best_reinforce.json").write_text(json.dumps(best_metadata, indent=2), encoding="utf-8")

    print("Saved REINFORCE sweep results and best model metadata.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Policy gradient training sweeps")
    parser.add_argument("--algo", type=str, choices=["ppo", "reinforce", "both"], default="both")
    parser.add_argument("--timesteps", type=int, default=80_000, help="Timesteps for PPO")
    parser.add_argument("--seed", type=int, default=300, help="Base seed for PPO")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.algo in ("ppo", "both"):
        run_ppo_sweep(total_timesteps=args.timesteps, base_seed=args.seed)

    if args.algo in ("reinforce", "both"):
        run_reinforce_sweep(base_seed=args.seed + 300)
