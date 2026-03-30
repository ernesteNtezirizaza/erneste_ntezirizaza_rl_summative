from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class ErgonomicThresholds:
    critical_posture: float = 0.18
    high_fatigue: float = 0.75
    break_need: float = 0.55


class OfficePostureEnv(gym.Env):
    """Mission-based posture monitoring environment for office workers.

    Observation vector (8 floats):
    0 neck_quality            [0, 1] higher is better
    1 shoulder_quality        [0, 1]
    2 spine_quality           [0, 1]
    3 fatigue                 [0, 1] lower is better
    4 desk_time_norm          [0, 1]
    5 break_pressure          [0, 1]
    6 recent_improvement      [-1, 1]
    7 last_action_norm        [0, 1]

    Action space (8 discrete actions):
    0 align_neck
    1 relax_shoulders
    2 lumbar_reset
    3 monitor_prompt
    4 micro_stretch_break
    5 short_walk_break
    6 breathing_reset
    7 ignore_prompt
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        max_steps: int = 480,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.thresholds = ErgonomicThresholds()

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._quality = np.zeros(3, dtype=np.float32)
        self._fatigue = 0.0
        self._recent_improvement = 0.0
        self._break_pressure = 0.0
        self._last_action = 0

        self._window = None
        self._clock = None

    def _get_obs(self) -> np.ndarray:
        desk_time_norm = min(1.0, self._step_count / float(self.max_steps))
        obs = np.array(
            [
                self._quality[0],
                self._quality[1],
                self._quality[2],
                self._fatigue,
                desk_time_norm,
                self._break_pressure,
                self._recent_improvement,
                self._last_action / 7.0,
            ],
            dtype=np.float32,
        )
        return obs

    def _ergonomic_score(self) -> float:
        quality_score = float(np.mean(self._quality))
        fatigue_penalty = 0.35 * self._fatigue
        return float(np.clip(quality_score - fatigue_penalty, 0.0, 1.0))

    def _action_effect(self, action: int) -> Tuple[np.ndarray, float]:
        quality_delta = np.zeros(3, dtype=np.float32)
        fatigue_delta = 0.0

        if action == 0:
            quality_delta = np.array([0.08, 0.01, 0.0], dtype=np.float32)
        elif action == 1:
            quality_delta = np.array([0.01, 0.08, 0.01], dtype=np.float32)
        elif action == 2:
            quality_delta = np.array([0.0, 0.02, 0.09], dtype=np.float32)
        elif action == 3:
            quality_delta = np.array([0.03, 0.03, 0.03], dtype=np.float32)
        elif action == 4:
            quality_delta = np.array([0.05, 0.05, 0.06], dtype=np.float32)
            fatigue_delta = -0.09
        elif action == 5:
            quality_delta = np.array([0.04, 0.05, 0.04], dtype=np.float32)
            fatigue_delta = -0.12
        elif action == 6:
            quality_delta = np.array([0.03, 0.02, 0.03], dtype=np.float32)
            fatigue_delta = -0.07
        elif action == 7:
            quality_delta = np.array([-0.02, -0.02, -0.03], dtype=np.float32)
            fatigue_delta = 0.02

        return quality_delta, fatigue_delta

    def _drift(self) -> Tuple[np.ndarray, float]:
        # Natural posture deterioration becomes stronger as fatigue rises.
        base_drift = 0.012 + 0.016 * self._fatigue
        random_drift = self._rng.normal(loc=0.0, scale=0.004, size=3).astype(np.float32)
        quality_drift = np.full(3, -base_drift, dtype=np.float32) + random_drift
        fatigue_drift = 0.005 + 0.006 * (1.0 - float(np.mean(self._quality)))
        return quality_drift, fatigue_drift

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._quality = self._rng.uniform(0.45, 0.75, size=3).astype(np.float32)
        self._fatigue = float(self._rng.uniform(0.05, 0.18))
        self._recent_improvement = 0.0
        self._break_pressure = 0.0
        self._last_action = 3

        obs = self._get_obs()
        info = {
            "ergonomic_score": self._ergonomic_score(),
            "mission": "Maintain healthy posture during office shift",
        }
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"

        prev_mean_quality = float(np.mean(self._quality))
        prev_score = self._ergonomic_score()

        drift_q, drift_f = self._drift()
        act_q, act_f = self._action_effect(action)

        self._quality = np.clip(self._quality + drift_q + act_q, 0.0, 1.0)
        self._fatigue = float(np.clip(self._fatigue + drift_f + act_f, 0.0, 1.0))

        self._step_count += 1
        self._last_action = int(action)

        mean_quality = float(np.mean(self._quality))
        improvement = mean_quality - prev_mean_quality
        self._recent_improvement = float(np.clip(0.9 * self._recent_improvement + improvement, -1.0, 1.0))

        desk_time_norm = min(1.0, self._step_count / float(self.max_steps))
        self._break_pressure = float(np.clip(0.6 * self._fatigue + 0.4 * desk_time_norm, 0.0, 1.0))

        reward = 4.0 * improvement
        reward += 1.4 * (self._ergonomic_score() - prev_score)
        reward -= 0.08 * self._fatigue

        if action in (4, 5) and self._fatigue > self.thresholds.break_need:
            reward += 0.25
        if action == 7 and self._break_pressure > self.thresholds.break_need:
            reward -= 0.35

        terminated = False
        truncated = False

        if mean_quality < self.thresholds.critical_posture:
            terminated = True
            reward -= 2.0
        elif self._step_count >= self.max_steps:
            terminated = True
            reward += 1.5 * self._ergonomic_score()

        obs = self._get_obs()
        info = {
            "ergonomic_score": self._ergonomic_score(),
            "mean_quality": mean_quality,
            "fatigue": self._fatigue,
            "break_pressure": self._break_pressure,
            "step": self._step_count,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        import pygame

        width, height = 900, 560
        if self._window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self._window = pygame.display.set_mode((width, height))
            else:
                self._window = pygame.Surface((width, height))
            self._clock = pygame.time.Clock()

        canvas = pygame.Surface((width, height))
        canvas.fill((240, 244, 246))

        def gauge(x: int, y: int, value: float, label: str, invert: bool = False) -> None:
            bar_w, bar_h = 220, 22
            pygame.draw.rect(canvas, (210, 214, 218), pygame.Rect(x, y, bar_w, bar_h), border_radius=8)
            draw_value = (1.0 - value) if invert else value
            fill = int(bar_w * np.clip(draw_value, 0.0, 1.0))
            color = (
                int(220 * (1 - draw_value)),
                int(170 * draw_value + 50),
                80,
            )
            pygame.draw.rect(canvas, color, pygame.Rect(x, y, fill, bar_h), border_radius=8)

            font = pygame.font.SysFont("Segoe UI", 20)
            txt = font.render(f"{label}: {value:.2f}", True, (20, 30, 40))
            canvas.blit(txt, (x, y - 28))

        gauge(50, 80, float(self._quality[0]), "Neck Quality")
        gauge(50, 160, float(self._quality[1]), "Shoulder Quality")
        gauge(50, 240, float(self._quality[2]), "Spine Quality")
        gauge(50, 320, float(self._fatigue), "Fatigue", invert=True)
        gauge(50, 400, float(self._break_pressure), "Break Pressure", invert=True)

        center_x, center_y = 620, 290
        quality = float(np.mean(self._quality))
        bend = int((1.0 - quality) * 50)

        skin = (46, 62, 80)
        pygame.draw.circle(canvas, skin, (center_x, center_y - 120), 25, width=3)
        pygame.draw.line(canvas, skin, (center_x, center_y - 95), (center_x + bend, center_y - 20), width=6)
        pygame.draw.line(canvas, skin, (center_x + bend, center_y - 20), (center_x + bend - 35, center_y + 80), width=6)
        pygame.draw.line(canvas, skin, (center_x + bend, center_y - 20), (center_x + bend + 35, center_y + 80), width=6)
        pygame.draw.line(canvas, skin, (center_x + bend - 5, center_y + 5), (center_x + bend - 40, center_y + 45), width=6)
        pygame.draw.line(canvas, skin, (center_x + bend - 5, center_y + 5), (center_x + bend + 40, center_y + 45), width=6)

        score_font = pygame.font.SysFont("Segoe UI", 28, bold=True)
        score = self._ergonomic_score()
        score_txt = score_font.render(f"Ergonomic Score: {score:.2f}", True, (12, 40, 70))
        canvas.blit(score_txt, (500, 45))

        action_names = [
            "align_neck",
            "relax_shoulders",
            "lumbar_reset",
            "monitor_prompt",
            "micro_stretch_break",
            "short_walk_break",
            "breathing_reset",
            "ignore_prompt",
        ]
        action_txt = pygame.font.SysFont("Segoe UI", 24).render(
            f"Last Action: {action_names[self._last_action]}", True, (30, 30, 30)
        )
        canvas.blit(action_txt, (470, 500))

        if self.render_mode == "human":
            self._window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self._clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self) -> None:
        if self._window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self._window = None
            self._clock = None
