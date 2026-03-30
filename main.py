from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from stable_baselines3 import DQN, PPO

from environment.custom_env import OfficePostureEnv
from training.reinforce_agent import PolicyNetwork

ROOT = Path(__file__).resolve().parent
TABLES_DIR = ROOT / "results" / "tables"


def load_best_metadata() -> Dict:
    candidates = []
    for name in ["best_dqn.json", "best_ppo.json", "best_reinforce.json"]:
        path = TABLES_DIR / name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            candidates.append(data)

    if not candidates:
        raise FileNotFoundError(
            "No best model metadata found. Run training scripts first to generate best_*.json files."
        )

    return sorted(candidates, key=lambda x: x["mean_eval_reward"], reverse=True)[0]


def run_trained_agent(algorithm: str | None = None, episodes: int = 1, max_steps: int = 500) -> None:
    best = load_best_metadata()
    if algorithm is not None:
        preferred = [best]
        target = algorithm.strip().upper()
        for name in ["best_dqn.json", "best_ppo.json", "best_reinforce.json"]:
            path = TABLES_DIR / name
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if data["algorithm"].upper() == target:
                    preferred = [data]
                    break
        best = preferred[0]

    algo = best["algorithm"].upper()
    model_path = best["model_path"]

    print(f"Running best model: {algo} ({model_path})")

    env = OfficePostureEnv(max_steps=max_steps, render_mode="human")

    if algo == "DQN":
        model = DQN.load(model_path)
        policy_fn = lambda obs: int(model.predict(obs, deterministic=True)[0])
    elif algo == "PPO":
        model = PPO.load(model_path)
        policy_fn = lambda obs: int(model.predict(obs, deterministic=True)[0])
    elif algo == "REINFORCE":
        payload = torch.load(model_path, map_location="cpu")
        cfg = payload["config"]
        policy = PolicyNetwork(obs_dim=8, action_dim=8, hidden_dim=int(cfg["hidden_dim"]))
        policy.load_state_dict(payload["state_dict"])
        policy.eval()

        def policy_fn(obs: np.ndarray) -> int:
            with torch.no_grad():
                logits = policy(torch.tensor(obs, dtype=torch.float32))
            return int(torch.argmax(logits).item())

    else:
        raise ValueError(f"Unsupported algorithm in metadata: {algo}")

    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=900 + ep)
        done = False
        total_reward = 0.0

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += float(reward)
            done = terminated or truncated

        print(
            f"episode={ep} reward={total_reward:.3f} "
            f"ergonomic_score={info['ergonomic_score']:.3f}"
        )

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the best trained posture RL agent")
    parser.add_argument("--algo", type=str, default=None, choices=["dqn", "ppo", "reinforce"], help="Override and run a specific algorithm")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to play")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_trained_agent(algorithm=args.algo, episodes=args.episodes, max_steps=args.max_steps)
