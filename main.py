"""
main.py
=======
Entry point for running the best performing RL agent
on the PostureMonitorEnv with full GUI visualisation.

Usage:
    python main.py                    # auto-selects best available model
    python main.py --model dqn        # force DQN
    python main.py --model ppo        # force PPO
    python main.py --model reinforce  # force REINFORCE
    python main.py --episodes 5       # run 5 episodes
    python main.py --no-render        # headless mode (terminal only)
    python main.py --export-json      # export episode data as JSON API payload
"""

import sys
import os
import json
import argparse
import time
import numpy as np
import types
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env import PostureMonitorEnv

ACTION_NAMES = [
    "Do Nothing",
    "Gentle Alert",
    "Urgent Alert",
    "Break Reminder",
    "Stretch Prompt",
    "Screen Adjust",
]

# Action IDs
ACTION_NOTHING = 0
ACTION_GENTLE_ALERT = 1
ACTION_URGENT_ALERT = 2
ACTION_BREAK_REMINDER = 3
ACTION_STRETCH_PROMPT = 4
ACTION_SCREEN_ADJUST = 5

# Observation indices (kept local to avoid importing environment internals)
HEAD_TILT = 0
NECK_ANGLE = 1
SHOULDER_DROP = 2
BACK_CURVATURE = 3
SCREEN_DISTANCE = 4
SITTING_DURATION = 5
FATIGUE_LEVEL = 6


def _is_posture_bad_obs(obs: np.ndarray) -> bool:
    """Lightweight posture check used by the runtime action guard."""
    return (
        abs(obs[HEAD_TILT]) > 15
        or obs[NECK_ANGLE] < -15
        or obs[SHOULDER_DROP] > 5
        or obs[BACK_CURVATURE] > 20
        or obs[SCREEN_DISTANCE] < 50
        or obs[SCREEN_DISTANCE] > 70
        or obs[SITTING_DURATION] > 45
    )


def _apply_action_guard(obs: np.ndarray, proposed_action: int,
                        prev_action: int | None, repeat_count: int) -> tuple[int, int, bool]:
    """
    Prevent long stretches of identical interventions during demonstration runs.
    This keeps play-time behavior readable and closer to realistic coaching.
    """
    if prev_action == proposed_action:
        repeat_count += 1
    else:
        repeat_count = 1

    action = proposed_action
    overridden = False

    if proposed_action == ACTION_STRETCH_PROMPT:
        guard_limit = 20
    elif proposed_action == ACTION_URGENT_ALERT:
        guard_limit = 12
    else:
        guard_limit = 14

    if repeat_count >= guard_limit and proposed_action != ACTION_NOTHING:
        posture_bad = _is_posture_bad_obs(obs)
        fatigue = float(obs[FATIGUE_LEVEL])
        sitting = float(obs[SITTING_DURATION])
        screen = float(obs[SCREEN_DISTANCE])

        if sitting >= 60:
            action = ACTION_BREAK_REMINDER
        elif screen < 50 or screen > 70:
            action = ACTION_SCREEN_ADJUST
        elif not posture_bad and fatigue < 0.45:
            action = ACTION_NOTHING
        elif proposed_action == ACTION_STRETCH_PROMPT:
            # Keep stretch behavior unless context strongly suggests a switch.
            action = ACTION_STRETCH_PROMPT
        elif proposed_action == ACTION_URGENT_ALERT:
            action = ACTION_GENTLE_ALERT
        else:
            action = ACTION_NOTHING

        overridden = action != proposed_action
        if overridden:
            repeat_count = 1

    return int(action), int(repeat_count), overridden


def _install_numpy_core_compat():
    """Provide legacy numpy._core module aliases needed by older pickles."""
    if 'numpy._core' in sys.modules:
        return

    core_pkg = types.ModuleType('numpy._core')
    core_pkg.__path__ = []

    core_modules = {
        'numeric',
        'multiarray',
        'umath',
        'numerictypes',
        'fromnumeric',
        'shape_base',
        'arrayprint',
        'records',
        'defchararray',
        'einsumfunc',
        'getlimits',
        '_multiarray_umath',
        'overrides',
    }

    for module_name in core_modules:
        try:
            module = importlib.import_module(f'numpy.core.{module_name}')
            sys.modules[f'numpy._core.{module_name}'] = module
            setattr(core_pkg, module_name, module)
        except Exception:
            continue

    sys.modules['numpy._core'] = core_pkg


def load_best_model(model_choice: str):
    """
    Load the best available trained model.

    REINFORCE is implemented from scratch in PyTorch (.pt file).
    DQN and PPO are loaded from Stable-Baselines3 (.zip files).
    """
    _install_numpy_core_compat()

    from stable_baselines3 import DQN, PPO

    # ── Try PPO (SB3 .zip) ────────────────────────────────────────
    if model_choice in ("auto", "ppo"):
        path = "models/pg/best_ppo_model.zip"
        if os.path.exists(path):
            try:
                model = PPO.load(path, device="cpu")
                print(f"[main] Loaded PPO model from {path}")
                return "PPO", model
            except Exception as e:
                print(f"[main] Could not load PPO: {e}")

    # ── Try DQN (SB3 .zip) ────────────────────────────────────────
    if model_choice in ("auto", "dqn"):
        path = "models/dqn/best_dqn_model.zip"
        if os.path.exists(path):
            try:
                model = DQN.load(path, device="cpu")
                print(f"[main] Loaded DQN model from {path}")
                return "DQN", model
            except Exception as e:
                print(f"[main] Could not load DQN: {e}")

    # ── Try REINFORCE (pure PyTorch .pt) ─────────────────────────
    if model_choice in ("auto", "reinforce"):
        path = "models/pg/best_reinforce_model.pt"
        if os.path.exists(path):
            try:
                import torch
                import torch.nn as nn
                from torch.distributions import Categorical

                # Rebuild the PolicyNetwork inline so main.py stays self-contained
                class PolicyNetwork(nn.Module):
                    def __init__(self, obs_dim, act_dim, hidden):
                        super().__init__()
                        layers, in_dim = [], obs_dim
                        for h in hidden:
                            layers += [nn.Linear(in_dim, h), nn.ReLU()]
                            in_dim = h
                        layers.append(nn.Linear(in_dim, act_dim))
                        self.net = nn.Sequential(*layers)
                    def forward(self, x):
                        return torch.softmax(self.net(x), dim=-1)

                # Wrap into an object with a predict() interface
                class REINFORCEModel:
                    def __init__(self, ckpt_path, obs_dim=8, act_dim=6):
                        ckpt   = torch.load(ckpt_path, map_location="cpu")
                        # Infer hidden size from state dict keys
                        hidden = [64, 64]   # default; matches most run configs
                        self.policy = PolicyNetwork(obs_dim, act_dim, hidden)
                        try:
                            self.policy.load_state_dict(ckpt["policy_state_dict"])
                        except RuntimeError:
                            # Try larger network if shape mismatch
                            for h in [[128,128],[256,128],[128,64],[256,256]]:
                                try:
                                    self.policy = PolicyNetwork(obs_dim, act_dim, h)
                                    self.policy.load_state_dict(ckpt["policy_state_dict"])
                                    break
                                except RuntimeError:
                                    continue
                        self.policy.eval()

                    def predict(self, obs, deterministic=True):
                        x      = torch.FloatTensor(obs).unsqueeze(0)
                        probs  = self.policy(x).squeeze(0)
                        action = probs.argmax().item() if deterministic else \
                                 Categorical(probs).sample().item()
                        return np.array([action]), None

                model = REINFORCEModel(path)
                print(f"[main] Loaded pure REINFORCE model from {path}")
                return "REINFORCE", model
            except Exception as e:
                print(f"[main] Could not load REINFORCE: {e}")

    print("[main] No trained model found. Using random policy (run training first).")
    return "Random", None


def run_episode(env, model, renderer, episode_num, export_json=False):
    """Run one episode and return episode statistics."""
    obs, info = env.reset(seed=episode_num * 42)
    total_reward = 0.0
    step         = 0
    episode_data = []
    prev_action  = None
    repeat_count = 0

    print(f"\n{'─'*60}")
    print(f"  Episode {episode_num} | Worker compliance: {info.get('worker_compliance', '?'):.2f}")
    print(f"{'─'*60}")

    while True:
        # ── Action selection ──────────────────────────────────────
        if model is not None:
            raw_action, _ = model.predict(obs, deterministic=True)
            raw_action = int(raw_action)
            action, repeat_count, guarded = _apply_action_guard(
                obs, raw_action, prev_action, repeat_count
            )
        else:
            action = env.action_space.sample()
            guarded = False
            repeat_count = 1 if action == prev_action else 0

        prev_action = action

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step         += 1

        # ── Render ────────────────────────────────────────────────
        if renderer is not None:
            renderer.render(
                state        = obs,
                step         = step,
                total_reward = total_reward,
                action       = action,
                last_reward  = reward,
            )
            time.sleep(0.08)

        # ── Terminal verbose ──────────────────────────────────────
        guard_tag = " [guard]" if guarded else ""
        print(
            f"  Step {step:3d} | {ACTION_NAMES[action]:16s} | "
            f"Reward: {reward:+5.2f} | Total: {total_reward:7.2f} | "
            f"Fatigue: {obs[6]:.2f} | Bad: {info['posture_bad']}{guard_tag}"
        )

        if export_json:
            episode_data.append({
                "step":        step,
                "action":      action,
                "action_name": ACTION_NAMES[action],
                "reward":      round(float(reward), 4),
                "observation": obs.tolist(),
                "info":        {k: (bool(v) if isinstance(v, (bool, np.bool_)) else
                                    float(v) if isinstance(v, (float, np.floating)) else v)
                                for k, v in info.items()},
            })

        if terminated or truncated:
            break

    print(f"\n  Episode {episode_num} complete | Total Reward: {total_reward:.2f} | Steps: {step}")
    return {"episode": episode_num, "total_reward": total_reward, "steps": step,
            "trajectory": episode_data}


def export_as_json_api(all_episodes: list, model_name: str):
    """
    Serialize episode data as a JSON API payload.
    Demonstrates how this RL agent can serve as a backend API
    to a frontend web/mobile application.
    """
    payload = {
        "api_version":  "1.0",
        "model":        model_name,
        "description":  "PostureMonitor RL — Real-Time Coaching Agent",
        "actions":      ACTION_NAMES,
        "episodes":     all_episodes,
        "summary": {
            "num_episodes":   len(all_episodes),
            "mean_reward":    float(np.mean([e["total_reward"] for e in all_episodes])),
            "mean_steps":     float(np.mean([e["steps"]        for e in all_episodes])),
        },
    }
    out_path = "logs/api_export.json"
    os.makedirs("logs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[JSON API] Exported to {out_path}")
    print(f"[JSON API] This payload can be consumed by a web/mobile frontend to")
    print(f"           display posture coaching recommendations in real time.")
    return payload


def main():
    parser = argparse.ArgumentParser(description="PostureMonitor RL — Run Best Agent")
    parser.add_argument("--model",       default="auto",
                        choices=["auto", "dqn", "ppo", "reinforce"],
                        help="Which model to load")
    parser.add_argument("--episodes",    type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--no-render",   action="store_true",
                        help="Run headless (no pygame window)")
    parser.add_argument("--export-json", action="store_true",
                        help="Export episode trajectories as JSON API payload")
    args = parser.parse_args()

    print("=" * 60)
    print("  PostureMonitor RL — Real-Time Office Posture Coaching")
    print("  Capstone: Preventing Musculoskeletal Disorders (MSDs)")
    print("=" * 60)
    print(f"  Model:      {args.model.upper()}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Rendering:  {'Disabled' if args.no_render else 'Pygame GUI'}")
    print("=" * 60)

    model_name, model = load_best_model(args.model)

    # ── Environment & renderer ────────────────────────────────────
    env = PostureMonitorEnv(render_mode=None)

    renderer = None
    if not args.no_render:
        try:
            from environment.rendering import PostureRenderer
            renderer = PostureRenderer()
            print("[main] Pygame renderer initialised.")
        except Exception as e:
            print(f"[main] Renderer unavailable ({e}). Running headless.")

    # ── Run episodes ──────────────────────────────────────────────
    all_results = []
    for ep in range(1, args.episodes + 1):
        result = run_episode(env, model, renderer, ep, export_json=args.export_json)
        all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────
    rewards = [r["total_reward"] for r in all_results]
    steps   = [r["steps"]        for r in all_results]
    print("\n" + "=" * 60)
    print("  AGENT PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"  Model:           {model_name}")
    print(f"  Episodes run:    {args.episodes}")
    print(f"  Mean reward:     {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean steps:      {np.mean(steps):.1f}")
    print(f"  Best episode:    #{np.argmax(rewards)+1} ({max(rewards):.2f})")
    print("=" * 60)

    if args.export_json:
        export_as_json_api(all_results, model_name)

    if renderer:
        renderer.close()
    env.close()


if __name__ == "__main__":
    main()
