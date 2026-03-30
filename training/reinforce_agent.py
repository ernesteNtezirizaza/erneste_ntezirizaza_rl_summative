from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from environment.custom_env import OfficePostureEnv


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ReinforceConfig:
    learning_rate: float
    gamma: float
    hidden_dim: int
    entropy_coef: float
    max_steps: int
    episodes: int


class ReinforceTrainer:
    def __init__(self, cfg: ReinforceConfig, seed: int = 0, device: str = "cpu") -> None:
        self.cfg = cfg
        self.seed = seed
        self.device = torch.device(device)

        self.env = OfficePostureEnv(max_steps=cfg.max_steps, seed=seed)
        obs_dim = int(self.env.observation_space.shape[0])
        action_dim = int(self.env.action_space.n)

        self.policy = PolicyNetwork(obs_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)

        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

    def select_action(self, obs: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        logits = self.policy(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return int(action.item()), log_prob, entropy

    def _compute_returns(self, rewards: List[float]) -> torch.Tensor:
        returns = []
        discounted = 0.0
        for r in reversed(rewards):
            discounted = r + self.cfg.gamma * discounted
            returns.insert(0, discounted)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if returns_t.numel() > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        return returns_t

    def train(self) -> Dict[str, List[float]]:
        reward_history: List[float] = []
        entropy_history: List[float] = []
        loss_history: List[float] = []

        for episode in range(1, self.cfg.episodes + 1):
            obs, _ = self.env.reset(seed=self.seed + episode)
            done = False

            rewards: List[float] = []
            log_probs: List[torch.Tensor] = []
            entropies: List[torch.Tensor] = []

            while not done:
                action, log_prob, entropy = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                rewards.append(float(reward))
                log_probs.append(log_prob)
                entropies.append(entropy)

                obs = next_obs
                done = terminated or truncated

            returns = self._compute_returns(rewards)
            log_probs_t = torch.stack(log_probs)
            entropy_t = torch.stack(entropies)

            policy_loss = -(log_probs_t * returns).sum()
            entropy_loss = -self.cfg.entropy_coef * entropy_t.mean()
            total_loss = policy_loss + entropy_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            episode_reward = float(np.sum(rewards))
            reward_history.append(episode_reward)
            entropy_history.append(float(entropy_t.mean().item()))
            loss_history.append(float(total_loss.item()))

            if episode % 25 == 0:
                rolling = np.mean(reward_history[-25:])
                print(
                    f"[REINFORCE] episode={episode} "
                    f"rolling_reward={rolling:.3f} loss={total_loss.item():.3f}"
                )

        return {
            "episode_reward": reward_history,
            "entropy": entropy_history,
            "loss": loss_history,
        }

    def evaluate(self, episodes: int = 10) -> Tuple[float, float]:
        returns = []
        for ep in range(episodes):
            obs, _ = self.env.reset(seed=10_000 + self.seed + ep)
            done = False
            total = 0.0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    logits = self.policy(obs_t)
                    action = int(torch.argmax(logits).item())
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total += float(reward)
                done = terminated or truncated
            returns.append(total)
        return float(np.mean(returns)), float(np.std(returns))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.policy.state_dict(),
            "config": self.cfg.__dict__,
        }
        torch.save(payload, path)


def run_reinforce_experiment(config: ReinforceConfig, seed: int, model_path: Path):
    trainer = ReinforceTrainer(config, seed=seed)
    logs = trainer.train()
    mean_reward, std_reward = trainer.evaluate(episodes=10)
    trainer.save(model_path)
    return logs, mean_reward, std_reward
