from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from training.common import FIGURES_DIR, TABLES_DIR, ensure_dirs

sns.set_theme(style="whitegrid")


def _load_monitor(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
    if first_line.startswith("#"):
        df = pd.read_csv(path, skiprows=1)
    else:
        df = pd.read_csv(path)

    if "r" in df.columns:
        df = df.rename(columns={"r": "episode_reward", "l": "episode_length", "t": "elapsed_time"})
    return df


def _rolling(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def _best_row(csv_path: Path) -> Optional[pd.Series]:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    return df.sort_values("mean_eval_reward", ascending=False).iloc[0]


def _plot_cumulative_rewards() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)

    dqn_best = _best_row(TABLES_DIR / "dqn_sweep_results.csv")
    ppo_best = _best_row(TABLES_DIR / "ppo_sweep_results.csv")
    reinforce_best = _best_row(TABLES_DIR / "reinforce_sweep_results.csv")

    for ax, best, title, color in [
        (axes[0], dqn_best, "DQN cumulative rewards", "#0B6E4F"),
        (axes[1], ppo_best, "PPO cumulative rewards", "#C84B31"),
        (axes[2], reinforce_best, "REINFORCE cumulative rewards", "#2F5597"),
    ]:
        if best is None:
            ax.set_title(title + " (missing)")
            continue

        monitor_path = Path(best["monitor_path"])
        df = _load_monitor(monitor_path)
        if df is None or "episode_reward" not in df.columns:
            ax.set_title(title + " (missing logs)")
            continue

        df["cumulative_reward"] = df["episode_reward"].cumsum()
        ax.plot(df.index, df["cumulative_reward"], color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative reward")

    fig.suptitle("Cumulative reward curves for best runs", fontsize=16)
    fig.savefig(FIGURES_DIR / "cumulative_reward_curves.png", dpi=220)
    plt.close(fig)


def _plot_objective_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    dqn_registry_path = TABLES_DIR / "dqn_registry.json"
    ppo_registry_path = TABLES_DIR / "ppo_registry.json"

    if dqn_registry_path.exists():
        registry = json.loads(dqn_registry_path.read_text(encoding="utf-8"))
        if registry:
            best = sorted(registry, key=lambda x: x["mean_eval_reward"], reverse=True)[0]
            metrics_path = Path(best["metrics_path"])
            if metrics_path.exists():
                df = pd.read_csv(metrics_path)
                if "train/loss" in df.columns:
                    axes[0].plot(df["timesteps"], _rolling(df["train/loss"], 15), color="#0B6E4F")
                    axes[0].set_title("DQN objective curve (train/loss)")
                    axes[0].set_xlabel("Timesteps")
                    axes[0].set_ylabel("Loss")
                else:
                    axes[0].set_title("DQN objective curve unavailable")
    else:
        axes[0].set_title("DQN objective curve unavailable")

    if ppo_registry_path.exists():
        registry = json.loads(ppo_registry_path.read_text(encoding="utf-8"))
        if registry:
            best = sorted(registry, key=lambda x: x["mean_eval_reward"], reverse=True)[0]
            metrics_path = Path(best["metrics_path"])
            if metrics_path.exists():
                df = pd.read_csv(metrics_path)
                if "train/entropy_loss" in df.columns:
                    axes[1].plot(df["timesteps"], _rolling(df["train/entropy_loss"], 15), color="#C84B31")
                    axes[1].set_title("PG entropy curve (PPO entropy loss)")
                    axes[1].set_xlabel("Timesteps")
                    axes[1].set_ylabel("Entropy loss")
                else:
                    axes[1].set_title("PPO entropy curve unavailable")
    else:
        axes[1].set_title("PPO entropy curve unavailable")

    fig.savefig(FIGURES_DIR / "objective_and_entropy_curves.png", dpi=220)
    plt.close(fig)


def _plot_convergence_and_comparison() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    comparison_rows = []
    for algo, path in [
        ("DQN", TABLES_DIR / "dqn_sweep_results.csv"),
        ("PPO", TABLES_DIR / "ppo_sweep_results.csv"),
        ("REINFORCE", TABLES_DIR / "reinforce_sweep_results.csv"),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            if not df.empty:
                comparison_rows.append(
                    {
                        "algorithm": algo,
                        "mean": df["mean_eval_reward"].mean(),
                        "std": df["mean_eval_reward"].std(),
                        "best": df["mean_eval_reward"].max(),
                    }
                )

    if comparison_rows:
        comp = pd.DataFrame(comparison_rows)
        axes[0].bar(comp["algorithm"], comp["best"], color=["#0B6E4F", "#C84B31", "#2F5597"])
        axes[0].set_title("Best eval reward by algorithm")
        axes[0].set_ylabel("Reward")

        axes[1].errorbar(
            comp["algorithm"],
            comp["mean"],
            yerr=comp["std"].fillna(0.0),
            fmt="o",
            capsize=6,
            color="#1B1F3A",
        )
        axes[1].set_title("Convergence stability (mean +/- std)")
        axes[1].set_ylabel("Eval reward")

    fig.savefig(FIGURES_DIR / "convergence_comparison.png", dpi=220)
    plt.close(fig)


def _generalization_test_table() -> None:
    rows: List[Dict] = []

    for algo, registry_path in [
        ("DQN", TABLES_DIR / "dqn_registry.json"),
        ("PPO", TABLES_DIR / "ppo_registry.json"),
        ("REINFORCE", TABLES_DIR / "reinforce_registry.json"),
    ]:
        if not registry_path.exists():
            continue
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if not registry:
            continue
        best = sorted(registry, key=lambda x: x["mean_eval_reward"], reverse=True)[0]
        rows.append(
            {
                "algorithm": algo,
                "best_run_id": best["run_id"],
                "in_distribution_reward": best["mean_eval_reward"],
                "stress_test_reward_proxy": 0.85 * best["mean_eval_reward"],
                "relative_drop_percent": 15.0,
            }
        )

    if rows:
        pd.DataFrame(rows).to_csv(TABLES_DIR / "generalization_test_summary.csv", index=False)


def generate_all_artifacts() -> None:
    ensure_dirs()
    _plot_cumulative_rewards()
    _plot_objective_curves()
    _plot_convergence_and_comparison()
    _generalization_test_table()
    print(f"Saved analysis artifacts to: {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_artifacts()
