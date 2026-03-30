from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import OfficePostureEnv


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"


def ensure_dirs() -> None:
    for path in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR / "dqn", MODELS_DIR / "pg"]:
        path.mkdir(parents=True, exist_ok=True)


def make_env(seed: int, monitor_path: Path, max_steps: int = 480) -> gym.Env:
    env = OfficePostureEnv(max_steps=max_steps)
    env = Monitor(env, filename=str(monitor_path))
    env.reset(seed=seed)
    return env


@dataclass
class RunResult:
    algorithm: str
    run_id: int
    mean_eval_reward: float
    std_eval_reward: float
    model_path: str
    monitor_path: str
    hyperparameters: Dict[str, float]


def write_results_csv(csv_path: Path, rows: Iterable[RunResult]) -> None:
    rows = list(rows)
    if not rows:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "algorithm",
        "run_id",
        "mean_eval_reward",
        "std_eval_reward",
        "model_path",
        "monitor_path",
        "hyperparameters",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_dict = asdict(row)
            row_dict["hyperparameters"] = json.dumps(row_dict["hyperparameters"])
            writer.writerow(row_dict)


def save_run_registry(path: Path, registry: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


class MetricCaptureCallback(BaseCallback):
    """Capture selected SB3 training metrics and save them as CSV."""

    def __init__(self, save_path: Path, keys: List[str], verbose: int = 0) -> None:
        super().__init__(verbose)
        self.save_path = save_path
        self.keys = keys
        self.rows: List[Dict[str, float]] = []

    def _on_step(self) -> bool:
        current_values = self.logger.name_to_value
        row = {"timesteps": float(self.num_timesteps)}
        found_any = False
        for key in self.keys:
            if key in current_values:
                row[key] = float(current_values[key])
                found_any = True
        if found_any:
            self.rows.append(row)
        return True

    def _on_training_end(self) -> None:
        if not self.rows:
            return

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure stable columns even if keys appear sparsely.
        all_fields = {"timesteps"}
        for row in self.rows:
            all_fields.update(row.keys())
        fieldnames = ["timesteps"] + sorted(k for k in all_fields if k != "timesteps")

        with self.save_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
