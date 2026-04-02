"""
PostureMonitorEnv - Custom Gymnasium Environment
Real-Time Posture Monitoring and Correction System for Office Workers
Capstone Project RL Environment
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
# Posture state indices
HEAD_TILT       = 0   # degrees from neutral (-30 to +30)
NECK_ANGLE      = 1   # degrees (0 = straight, negative = forward)
SHOULDER_DROP   = 2   # asymmetry score (0-10)
BACK_CURVATURE  = 3   # lumbar angle (0 = upright, 30 = slouched)
SCREEN_DISTANCE = 4   # cm (40-90 cm ideal range: 50-70)
SITTING_DURATION= 5   # minutes (0-120)
FATIGUE_LEVEL   = 6   # 0-1 float (increases with time & bad posture)
ALERT_IGNORED   = 7   # count of consecutive ignored alerts (0-5)

NUM_OBS = 8

# Actions
ACTION_NOTHING         = 0
ACTION_GENTLE_ALERT    = 1
ACTION_URGENT_ALERT    = 2
ACTION_BREAK_REMINDER  = 3
ACTION_STRETCH_PROMPT  = 4
ACTION_SCREEN_ADJUST   = 5
NUM_ACTIONS = 6

# Posture thresholds
HEAD_TILT_THRESHOLD   = 15.0   # degrees — warn beyond this
NECK_THRESHOLD        = -15.0  # degrees — warn if below
SHOULDER_THRESHOLD    = 5.0    # asymmetry units
BACK_THRESHOLD        = 20.0   # degrees slouch
SCREEN_MIN            = 50.0   # cm
SCREEN_MAX            = 70.0   # cm
SIT_THRESHOLD         = 45.0   # minutes before break needed

MAX_STEPS             = 200
FATIGUE_INCREMENT     = 0.005
FATIGUE_BAD_POSTURE   = 0.015
MAX_IGNORED_ALERTS    = 5


class PostureMonitorEnv(gym.Env):
    """
    PostureMonitorEnv
    =================
    Simulates an office worker's posture over a work session.
    The RL agent acts as a smart posture coaching system:
    - It observes biomechanical posture metrics
    - Decides when and how to intervene (alert, remind, adjust)
    - Learns to balance correction effectiveness vs. alert fatigue

    Observation Space (8 continuous values):
        [head_tilt, neck_angle, shoulder_drop, back_curvature,
         screen_distance, sitting_duration, fatigue_level, alert_ignored_count]

    Action Space (6 discrete actions):
        0 - Do Nothing
        1 - Send Gentle Alert
        2 - Send Urgent Alert
        3 - Send Break Reminder
        4 - Prompt Stretch Exercise
        5 - Suggest Screen Distance Adjustment

    Reward Structure:
        +2.0  : Worker corrects posture after intervention
        +1.0  : Posture already good, no intervention needed
        -1.0  : Alert ignored (alert fatigue)
        -0.5  : Unnecessary alert when posture is fine
        -2.0  : Worker reaches dangerous fatigue level
        +0.3  : Sustained good posture bonus (every 10 steps)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # ── Observation Space ──────────────────────────────────────
        low  = np.array([-30, -45,  0,  0, 30,   0, 0, 0], dtype=np.float32)
        high = np.array([ 30,  10, 10, 45, 90, 120, 1, 5], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ── Action Space ───────────────────────────────────────────
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Internal state
        self._state    = None
        self._step_count = 0
        self._good_posture_streak = 0
        self._total_reward = 0.0
        self._history  = []           # for rendering / logging
        self._correction_prob = 0.7   # probability worker corrects on gentle alert
        self._worker_compliance = random.uniform(0.5, 1.0)  # per-episode trait

        # Renderer (lazy-loaded)
        self._renderer = None

    # ──────────────────────────────────────────────────────────────
    # RESET
    # ──────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        self._state = np.array([
            rng.uniform(-10, 10),    # head tilt  — near neutral at session start
            rng.uniform(-10,  5),    # neck angle
            rng.uniform(  0,  3),    # shoulder drop
            rng.uniform(  0, 10),    # back curvature — mostly upright
            rng.uniform( 55, 65),    # screen distance — within ideal range
            0.0,                      # sitting duration (fresh session)
            0.0,                      # fatigue
            0.0,                      # alert ignored count
        ], dtype=np.float32)

        self._step_count          = 0
        self._good_posture_streak = 0
        self._total_reward        = 0.0
        self._history             = []
        self._worker_compliance   = rng.uniform(0.5, 1.0)

        info = {"worker_compliance": float(self._worker_compliance)}
        return self._state.copy(), info

    # ──────────────────────────────────────────────────────────────
    # STEP
    # ──────────────────────────────────────────────────────────────
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        s = self._state
        reward       = 0.0
        terminated   = False
        truncated    = False

        posture_bad  = self._is_posture_bad(s)
        fatigue_high = s[FATIGUE_LEVEL] > 0.75

        # ── Apply action effects ───────────────────────────────────
        if action == ACTION_NOTHING:
            if not posture_bad:
                reward += 1.0   # correct decision: posture fine, no action needed
                self._good_posture_streak += 1
            else:
                reward -= 0.5   # missed a correction opportunity

        elif action == ACTION_GENTLE_ALERT:
            if posture_bad:
                if self.np_random.random() < self._worker_compliance * 0.8:
                    self._correct_posture_gradually(s)
                    reward += 2.0
                    s[ALERT_IGNORED] = max(0, s[ALERT_IGNORED] - 1)
                else:
                    s[ALERT_IGNORED] = min(MAX_IGNORED_ALERTS, s[ALERT_IGNORED] + 1)
                    reward -= 1.0
            else:
                reward -= 0.5   # unnecessary alert

        elif action == ACTION_URGENT_ALERT:
            if posture_bad or fatigue_high:
                if self.np_random.random() < self._worker_compliance * 0.9:
                    self._correct_posture_fully(s)
                    reward += 2.5
                    s[ALERT_IGNORED] = 0
                else:
                    s[ALERT_IGNORED] = min(MAX_IGNORED_ALERTS, s[ALERT_IGNORED] + 2)
                    reward -= 1.5
            else:
                reward -= 1.0   # crying wolf

        elif action == ACTION_BREAK_REMINDER:
            if s[SITTING_DURATION] >= SIT_THRESHOLD:
                s[SITTING_DURATION] = 0.0     # worker takes break
                s[FATIGUE_LEVEL]    = max(0, s[FATIGUE_LEVEL] - 0.2)
                reward += 2.0
            else:
                reward -= 0.3   # premature break reminder

        elif action == ACTION_STRETCH_PROMPT:
            if fatigue_high or s[SITTING_DURATION] > 30:
                s[FATIGUE_LEVEL]    = max(0, s[FATIGUE_LEVEL] - 0.15)
                s[SHOULDER_DROP]    = max(0, s[SHOULDER_DROP] - 2)
                s[BACK_CURVATURE]   = max(0, s[BACK_CURVATURE] - 3)
                reward += 1.5
            else:
                reward -= 0.2

        elif action == ACTION_SCREEN_ADJUST:
            dist = s[SCREEN_DISTANCE]
            if dist < SCREEN_MIN or dist > SCREEN_MAX:
                s[SCREEN_DISTANCE] = np.clip(dist + np.sign(60 - dist) * 5, 30, 90)
                reward += 1.0
            else:
                reward -= 0.3   # unnecessary adjustment

        # ── Environment dynamics (natural posture degradation) ────
        self._degrade_posture(s)

        # ── Sustained good posture bonus ──────────────────────────
        if self._good_posture_streak > 0 and self._good_posture_streak % 10 == 0:
            reward += 0.3

        # ── Danger zone penalty ───────────────────────────────────
        if s[FATIGUE_LEVEL] >= 1.0:
            reward   -= 2.0
            terminated = True   # worker has reached MSD risk threshold

        if s[ALERT_IGNORED] >= MAX_IGNORED_ALERTS:
            reward -= 1.0       # alert saturation — agent over-alerted

        # ── Step bookkeeping ──────────────────────────────────────
        self._step_count += 1
        self._total_reward += reward
        self._history.append({
            "step":   self._step_count,
            "action": action,
            "state":  s.copy(),
            "reward": reward,
        })

        if self._step_count >= MAX_STEPS:
            truncated = True

        obs  = np.clip(s, self.observation_space.low, self.observation_space.high)
        info = {
            "posture_bad":   bool(posture_bad),
            "total_reward":  float(self._total_reward),
            "step":          self._step_count,
            "fatigue":       float(s[FATIGUE_LEVEL]),
        }

        if self.render_mode == "human":
            self.render()

        return obs.astype(np.float32), float(reward), terminated, truncated, info

    # ──────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────
    def _is_posture_bad(self, s):
        return (
            abs(s[HEAD_TILT])     > HEAD_TILT_THRESHOLD  or
            s[NECK_ANGLE]         < NECK_THRESHOLD        or
            s[SHOULDER_DROP]      > SHOULDER_THRESHOLD    or
            s[BACK_CURVATURE]     > BACK_THRESHOLD        or
            s[SCREEN_DISTANCE]    < SCREEN_MIN            or
            s[SCREEN_DISTANCE]    > SCREEN_MAX            or
            s[SITTING_DURATION]   > SIT_THRESHOLD
        )

    def _correct_posture_gradually(self, s):
        """Simulate worker partially correcting posture."""
        s[HEAD_TILT]      *= 0.5
        s[NECK_ANGLE]      = min(s[NECK_ANGLE] + 5, 0)
        s[SHOULDER_DROP]   = max(0, s[SHOULDER_DROP] - 1)
        s[BACK_CURVATURE]  = max(0, s[BACK_CURVATURE] - 5)

    def _correct_posture_fully(self, s):
        """Simulate worker fully correcting posture after urgent alert."""
        s[HEAD_TILT]      = self.np_random.uniform(-5, 5)
        s[NECK_ANGLE]     = self.np_random.uniform(-5, 0)
        s[SHOULDER_DROP]  = self.np_random.uniform(0, 2)
        s[BACK_CURVATURE] = self.np_random.uniform(0, 8)

    def _degrade_posture(self, s):
        """Natural posture degradation over time."""
        rng = self.np_random
        s[SITTING_DURATION] = min(120, s[SITTING_DURATION] + 1.0)
        s[FATIGUE_LEVEL]    += FATIGUE_INCREMENT

        if self._is_posture_bad(s):
            s[FATIGUE_LEVEL] += FATIGUE_BAD_POSTURE
            self._good_posture_streak = 0

        # Gradual drift toward poor posture
        s[HEAD_TILT]      += rng.uniform(-1.5, 2.0)
        s[NECK_ANGLE]     += rng.uniform(-2.0, 0.5)
        s[SHOULDER_DROP]  += rng.uniform(-0.2, 0.5)
        s[BACK_CURVATURE] += rng.uniform(-0.5, 1.5)
        s[SCREEN_DISTANCE]+= rng.uniform(-1.0, 1.0)

        # Clip
        s[HEAD_TILT]       = np.clip(s[HEAD_TILT],      -30,  30)
        s[NECK_ANGLE]      = np.clip(s[NECK_ANGLE],     -45,  10)
        s[SHOULDER_DROP]   = np.clip(s[SHOULDER_DROP],    0,  10)
        s[BACK_CURVATURE]  = np.clip(s[BACK_CURVATURE],   0,  45)
        s[SCREEN_DISTANCE] = np.clip(s[SCREEN_DISTANCE], 30,  90)
        s[FATIGUE_LEVEL]   = np.clip(s[FATIGUE_LEVEL],    0,   1)

    # ──────────────────────────────────────────────────────────────
    # RENDER
    # ──────────────────────────────────────────────────────────────
    def render(self):
        if self.render_mode == "human":
            from environment.rendering import PostureRenderer
            if self._renderer is None:
                self._renderer = PostureRenderer()
            self._renderer.render(self._state, self._step_count, self._total_reward)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
