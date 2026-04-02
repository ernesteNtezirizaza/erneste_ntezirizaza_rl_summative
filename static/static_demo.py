"""
static_demo.py
==============
Demonstrates the PostureMonitorEnv with a random-action agent.
No model or training involved — purely shows the environment
visualisation components running in real time.

Usage:
    python static/static_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from environment.custom_env import PostureMonitorEnv
from environment.rendering  import PostureRenderer

DEMO_STEPS   = 300
STEP_DELAY   = 0.12   # seconds between steps (for visibility)


def run_random_demo():
    print("=" * 60)
    print("PostureMonitorEnv — Random-Action Demo (No Model)")
    print("=" * 60)
    print("Demonstrating environment components with random actions.\n")

    env      = PostureMonitorEnv(render_mode=None)
    renderer = PostureRenderer()

    obs, info = env.reset(seed=42)
    total_reward = 0.0

    action_names = [
        "Do Nothing", "Gentle Alert", "Urgent Alert",
        "Break Reminder", "Stretch Prompt", "Screen Adjust",
    ]

    for step in range(DEMO_STEPS):
        action = env.action_space.sample()   # RANDOM — no model
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        renderer.render(
            state        = obs,
            step         = step + 1,
            total_reward = total_reward,
            action       = action,
            last_reward  = reward,
        )

        print(
            f"Step {step+1:3d} | Action: {action_names[action]:16s} | "
            f"Reward: {reward:+5.2f} | Total: {total_reward:7.2f} | "
            f"Fatigue: {obs[6]:.2f} | Posture Bad: {info['posture_bad']}"
        )

        time.sleep(STEP_DELAY)

        if terminated or truncated:
            print("\n[Episode ended — resetting environment]")
            obs, info = env.reset()
            total_reward = 0.0

    renderer.close()
    env.close()
    print("\nDemo complete.")


if __name__ == "__main__":
    run_random_demo()
