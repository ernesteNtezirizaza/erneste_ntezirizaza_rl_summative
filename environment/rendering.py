from __future__ import annotations

from typing import Optional

import pygame

from environment.custom_env import OfficePostureEnv


class PostureDashboard:
    """Pygame visualization wrapper for running environment episodes."""

    def __init__(self, env: OfficePostureEnv) -> None:
        self.env = env

    def run_random_episode(self, steps: int = 300, seed: int = 42) -> None:
        obs, info = self.env.reset(seed=seed)
        print("Starting random agent demo...")
        print(f"Initial ergonomic score: {info['ergonomic_score']:.3f}")

        for step in range(1, steps + 1):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    return

            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()

            if step % 20 == 0:
                print(
                    f"step={step:03d} action={action} reward={reward:+.3f} "
                    f"quality={info['mean_quality']:.3f} fatigue={info['fatigue']:.3f}"
                )

            if terminated or truncated:
                print(f"Episode finished at step {step}")
                break

        self.env.close()


def run_random_demo(max_steps: int = 320, seed: int = 42) -> None:
    env = OfficePostureEnv(render_mode="human", max_steps=max_steps)
    dashboard = PostureDashboard(env)
    dashboard.run_random_episode(steps=max_steps, seed=seed)


if __name__ == "__main__":
    run_random_demo()
