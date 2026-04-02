"""
rendering.py – Pygame-based 2D visualisation for PostureMonitorEnv
Draws an office worker silhouette with real-time posture indicators,
metric gauges, action banners, and reward tracking.
"""

import math
import sys

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

import numpy as np

# ── Colour palette ──────────────────────────────────────────────
BG          = (15,  20,  35)
PANEL       = (25,  32,  55)
ACCENT      = (0,  210, 180)
WARN        = (255, 180,  30)
DANGER      = (255,  60,  60)
GOOD        = (60,  210,  90)
TEXT_MAIN   = (220, 230, 255)
TEXT_DIM    = (100, 120, 160)
WHITE       = (255, 255, 255)
SKIN        = (255, 213, 170)
SHIRT       = ( 80, 130, 200)
CHAIR       = (100,  80,  60)
DESK        = (160, 130,  90)
SCREEN_COL  = ( 40,  60, 100)
MONITOR_COL = (180, 210, 255)

ACTION_LABELS = [
    "⏸  Do Nothing",
    "🔔  Gentle Alert",
    "🚨  Urgent Alert",
    "☕  Break Reminder",
    "🧘  Stretch Prompt",
    "📏  Screen Adjust",
]

WIDTH, HEIGHT = 1100, 680
FPS = 10


class PostureRenderer:
    """Pygame renderer for the posture monitoring RL environment."""

    def __init__(self):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for rendering. pip install pygame")
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("PostureMonitor RL – Office Worker Agent")
        self.clock  = pygame.time.Clock()

        # Fonts
        self.font_lg  = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_md  = pygame.font.SysFont("monospace", 16)
        self.font_sm  = pygame.font.SysFont("monospace", 13)
        self.font_xl  = pygame.font.SysFont("monospace", 28, bold=True)

        self._last_action      = 0
        self._action_timer     = 0
        self._reward_history   = []

    # ──────────────────────────────────────────────────────────────
    def render(self, state, step, total_reward, action=None, last_reward=0.0):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        if action is not None:
            self._last_action  = action
            self._action_timer = 40

        self._reward_history.append(total_reward)
        if len(self._reward_history) > 150:
            self._reward_history.pop(0)

        self.screen.fill(BG)

        self._draw_title()
        self._draw_worker(state)
        self._draw_metrics_panel(state)
        self._draw_action_banner()
        self._draw_gauges(state)
        self._draw_reward_chart()
        self._draw_step_info(step, total_reward, last_reward)

        pygame.display.flip()
        self.clock.tick(FPS)

        if self._action_timer > 0:
            self._action_timer -= 1

    # ── Title bar ─────────────────────────────────────────────────
    def _draw_title(self):
        title = self.font_xl.render("PostureMonitor RL  –  Office Worker Coaching Agent", True, ACCENT)
        self.screen.blit(title, (20, 12))
        pygame.draw.line(self.screen, ACCENT, (20, 48), (WIDTH - 20, 48), 1)

    # ── Worker silhouette ─────────────────────────────────────────
    def _draw_worker(self, state):
        """Draw a simple but expressive office worker with posture-reactive geometry."""
        ox, oy = 340, 380   # origin (hip area)

        back_curv   = state[3]   # 0-45
        neck_angle  = state[1]   # -45 to 10
        head_tilt   = state[0]   # -30 to 30
        shoulder_dr = state[2]   # 0-10

        slouch_offset = int(back_curv * 0.6)   # pixels forward lean

        # ── Desk ──────────────────────────────────────────────────
        pygame.draw.rect(self.screen, DESK, (140, 310, 360, 18), border_radius=4)

        # ── Monitor ───────────────────────────────────────────────
        screen_dist = state[4]  # 30-90 cm → map to pixel offset
        mon_x = int(200 - (screen_dist - 60) * 1.5)
        pygame.draw.rect(self.screen, SCREEN_COL,  (mon_x, 220, 110, 88), border_radius=6)
        pygame.draw.rect(self.screen, MONITOR_COL, (mon_x + 6, 226, 98, 70), border_radius=3)
        # stand
        pygame.draw.line(self.screen, TEXT_DIM, (mon_x + 55, 308), (mon_x + 55, 312), 4)
        pygame.draw.rect(self.screen, TEXT_DIM, (mon_x + 35, 311, 42, 5), border_radius=2)

        # ── Chair ─────────────────────────────────────────────────
        pygame.draw.rect(self.screen, CHAIR, (290, 390, 130, 16), border_radius=4)   # seat
        pygame.draw.rect(self.screen, CHAIR, (380, 330, 18, 75),  border_radius=4)   # backrest
        pygame.draw.line(self.screen, CHAIR, (295, 406), (280, 460), 4)  # legs
        pygame.draw.line(self.screen, CHAIR, (415, 406), (430, 460), 4)

        # ── Torso ─────────────────────────────────────────────────
        torso_top_x = ox + slouch_offset
        torso_top_y = oy - 120
        torso_color = DANGER if back_curv > 25 else WARN if back_curv > 15 else SHIRT
        pygame.draw.polygon(self.screen, torso_color, [
            (ox - 25, oy),
            (ox + 25, oy),
            (torso_top_x + 20, torso_top_y),
            (torso_top_x - 20, torso_top_y),
        ])

        # ── Shoulders ─────────────────────────────────────────────
        l_sh_y = torso_top_y + int(shoulder_dr * 2)
        r_sh_y = torso_top_y
        pygame.draw.circle(self.screen, torso_color, (torso_top_x - 35, l_sh_y), 14)
        pygame.draw.circle(self.screen, torso_color, (torso_top_x + 35, r_sh_y), 14)

        # Arms on keyboard
        pygame.draw.line(self.screen, SKIN, (torso_top_x - 35, l_sh_y), (torso_top_x - 20, oy - 20), 10)
        pygame.draw.line(self.screen, SKIN, (torso_top_x + 35, r_sh_y), (torso_top_x + 10, oy - 20), 10)

        # ── Neck ──────────────────────────────────────────────────
        neck_fwd = int(abs(min(neck_angle, 0)) * 0.4)
        neck_top_x = torso_top_x + neck_fwd
        neck_top_y = torso_top_y - 28
        neck_color = DANGER if neck_angle < -20 else WARN if neck_angle < -10 else SKIN
        pygame.draw.line(self.screen, neck_color, (torso_top_x, torso_top_y), (neck_top_x, neck_top_y), 10)

        # ── Head ──────────────────────────────────────────────────
        head_cx = neck_top_x + int(head_tilt * 0.3)
        head_cy = neck_top_y - 30
        head_color = DANGER if abs(head_tilt) > 20 else WARN if abs(head_tilt) > 10 else SKIN
        pygame.draw.circle(self.screen, head_color, (head_cx, head_cy), 28)
        # Face
        pygame.draw.circle(self.screen, (80, 60, 40), (head_cx - 8, head_cy - 4), 4)
        pygame.draw.circle(self.screen, (80, 60, 40), (head_cx + 8, head_cy - 4), 4)
        mouth_curve = int(5 - back_curv * 0.15)
        pygame.draw.arc(self.screen, (80, 60, 40),
                        (head_cx - 10, head_cy + 6, 20, 12),
                        math.pi if mouth_curve < 0 else 0,
                        math.pi if mouth_curve >= 0 else 2 * math.pi, 2)

        # ── Posture quality label ──────────────────────────────────
        quality, q_col = self._posture_quality(state)
        ql = self.font_md.render(f"Posture: {quality}", True, q_col)
        self.screen.blit(ql, (230, 480))

        # ── Legend arrows ─────────────────────────────────────────
        self._draw_annotation(torso_top_x + 60, torso_top_y, f"Back {back_curv:.0f}°", torso_color)
        self._draw_annotation(head_cx + 40,     head_cy - 10, f"Head {head_tilt:+.0f}°", head_color)
        self._draw_annotation(neck_top_x + 45,  neck_top_y,   f"Neck {neck_angle:.0f}°", neck_color)

    def _draw_annotation(self, x, y, text, color):
        pygame.draw.line(self.screen, color, (x - 5, y), (x + 18, y), 1)
        surf = self.font_sm.render(text, True, color)
        self.screen.blit(surf, (x + 20, y - 8))

    def _posture_quality(self, state):
        bad_flags = (
            abs(state[0]) > 15,
            state[1] < -15,
            state[2] > 5,
            state[3] > 20,
            state[4] < 50 or state[4] > 70,
            state[5] > 45,
        )
        n = sum(bad_flags)
        if n == 0:    return "EXCELLENT", GOOD
        elif n <= 2:  return "FAIR",      WARN
        else:         return "POOR",      DANGER

    # ── Metrics panel ─────────────────────────────────────────────
    def _draw_metrics_panel(self, state):
        panel_x = 680
        pygame.draw.rect(self.screen, PANEL, (panel_x, 58, 400, 370), border_radius=10)
        pygame.draw.rect(self.screen, ACCENT, (panel_x, 58, 400, 370), width=1, border_radius=10)

        title = self.font_lg.render("Posture Metrics", True, ACCENT)
        self.screen.blit(title, (panel_x + 14, 70))

        metrics = [
            ("Head Tilt",       state[0], -30, 30,  -15, 15,   "°"),
            ("Neck Angle",      state[1], -45, 10,  -15,  5,   "°"),
            ("Shoulder Drop",   state[2],   0, 10,    0,  5,   "u"),
            ("Back Curvature",  state[3],   0, 45,    0, 20,   "°"),
            ("Screen Dist.",    state[4],  30, 90,   50, 70,  "cm"),
            ("Sitting Time",    state[5],   0,120,    0, 45, "min"),
            ("Fatigue",         state[6],   0,  1,    0,  0.6, ""),
            ("Alerts Ignored",  state[7],   0,  5,    0,  3,   ""),
        ]

        for i, (label, val, lo, hi, ok_lo, ok_hi, unit) in enumerate(metrics):
            y = 104 + i * 38
            # Label
            lbl = self.font_sm.render(label, True, TEXT_DIM)
            self.screen.blit(lbl, (panel_x + 14, y))
            # Value
            in_range = ok_lo <= val <= ok_hi if ok_hi != ok_lo else True
            col = GOOD if in_range else DANGER
            val_txt = self.font_md.render(f"{val:.1f}{unit}", True, col)
            self.screen.blit(val_txt, (panel_x + 160, y))
            # Bar
            bx, bw, bh = panel_x + 240, 140, 12
            norm = (val - lo) / (hi - lo)
            fill = int(norm * bw)
            pygame.draw.rect(self.screen, (40, 50, 80), (bx, y + 2, bw, bh), border_radius=3)
            pygame.draw.rect(self.screen, col, (bx, y + 2, max(0, fill), bh), border_radius=3)
            # OK zone markers
            ok_lo_x = bx + int((ok_lo - lo) / (hi - lo) * bw)
            ok_hi_x = bx + int((ok_hi - lo) / (hi - lo) * bw)
            pygame.draw.line(self.screen, GOOD, (ok_lo_x, y), (ok_lo_x, y + bh + 4), 1)
            pygame.draw.line(self.screen, GOOD, (ok_hi_x, y), (ok_hi_x, y + bh + 4), 1)

    # ── Action banner ─────────────────────────────────────────────
    def _draw_action_banner(self):
        alpha = min(255, self._action_timer * 8)
        if alpha <= 0:
            return
        label = ACTION_LABELS[self._last_action]
        colors = [TEXT_DIM, WARN, DANGER, ACCENT, GOOD, MONITOR_COL]
        col = colors[self._last_action]
        surf = self.font_lg.render(f"Action → {label}", True, col)
        self.screen.blit(surf, (20, 56))

    # ── Gauges (fatigue + compliance) ────────────────────────────
    def _draw_gauges(self, state):
        self._draw_arc_gauge(90,  580, state[6], "Fatigue",   DANGER)
        self._draw_arc_gauge(200, 580, min(state[5]/120, 1), "Sit Time", WARN)

    def _draw_arc_gauge(self, cx, cy, value, label, color):
        r  = 38
        start_a = math.pi
        end_a   = math.pi + value * math.pi
        # Background arc
        steps = 40
        for i in range(steps):
            a = math.pi + i / steps * math.pi
            x1 = int(cx + r * math.cos(a))
            y1 = int(cy + r * math.sin(a))
            a2 = math.pi + (i + 1) / steps * math.pi
            x2 = int(cx + r * math.cos(a2))
            y2 = int(cy + r * math.sin(a2))
            pygame.draw.line(self.screen, (40, 50, 80), (x1, y1), (x2, y2), 6)
        # Value arc
        fill_steps = int(value * steps)
        for i in range(fill_steps):
            a  = math.pi + i / steps * math.pi
            x1 = int(cx + r * math.cos(a))
            y1 = int(cy + r * math.sin(a))
            a2 = math.pi + (i + 1) / steps * math.pi
            x2 = int(cx + r * math.cos(a2))
            y2 = int(cy + r * math.sin(a2))
            pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), 6)
        # Text
        pct = self.font_md.render(f"{value*100:.0f}%", True, color)
        self.screen.blit(pct, (cx - 18, cy - 14))
        lbl = self.font_sm.render(label, True, TEXT_DIM)
        self.screen.blit(lbl, (cx - 24, cy + 8))

    # ── Reward history chart ──────────────────────────────────────
    def _draw_reward_chart(self):
        cx, cy, cw, ch = 680, 445, 400, 110
        pygame.draw.rect(self.screen, PANEL, (cx, cy, cw, ch), border_radius=8)
        pygame.draw.rect(self.screen, ACCENT, (cx, cy, cw, ch), width=1, border_radius=8)
        title = self.font_sm.render("Cumulative Reward", True, ACCENT)
        self.screen.blit(title, (cx + 8, cy + 6))

        h = self._reward_history
        if len(h) < 2:
            return
        mn, mx = min(h), max(h) + 1e-6
        pts = []
        for i, v in enumerate(h):
            px = cx + 8 + int(i / max(len(h) - 1, 1) * (cw - 16))
            py = cy + ch - 20 - int((v - mn) / (mx - mn) * (ch - 30))
            pts.append((px, py))
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, ACCENT, False, pts, 2)
        # Zero line
        zero_y = cy + ch - 20 - int((0 - mn) / (mx - mn) * (ch - 30))
        pygame.draw.line(self.screen, TEXT_DIM, (cx + 8, zero_y), (cx + cw - 8, zero_y), 1)

    # ── Step / reward info ────────────────────────────────────────
    def _draw_step_info(self, step, total_reward, last_reward):
        info_y = 570
        step_t = self.font_md.render(f"Step: {step:4d}", True, TEXT_MAIN)
        tot_t  = self.font_md.render(f"Total Reward: {total_reward:7.2f}", True, TEXT_MAIN)
        lr_col = GOOD if last_reward >= 0 else DANGER
        lr_t   = self.font_md.render(f"Last Δ: {last_reward:+.2f}", True, lr_col)
        self.screen.blit(step_t, (20, info_y))
        self.screen.blit(tot_t,  (20, info_y + 22))
        self.screen.blit(lr_t,   (20, info_y + 44))

    # ── Cleanup ───────────────────────────────────────────────────
    def close(self):
        pygame.quit()
