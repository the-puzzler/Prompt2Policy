from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prompt2policy.algorithms.interface import AlgorithmAdapter, AlgorithmConfig


@dataclass
class PPOAdapter(AlgorithmAdapter):
    model: object | None = None

    def train(self, env, eval_env, config: AlgorithmConfig) -> dict[str, float]:
        del eval_env

        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError("stable-baselines3 is required for PPOAdapter") from exc

        if self.model is None:
            self.model = PPO(
                policy="MultiInputPolicy",
                env=env,
                learning_rate=config.learning_rate,
                n_steps=config.n_steps,
                batch_size=config.batch_size,
                gamma=config.gamma,
                seed=config.seed,
                device=config.device,
                verbose=config.verbose,
            )

        self.model.learn(total_timesteps=config.total_timesteps)
        return {
            "timesteps": float(config.total_timesteps),
        }

    def predict(self, observation, deterministic: bool = True):
        if self.model is None:
            raise RuntimeError("PPO model not initialized")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save uninitialized PPO model")
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env=None) -> "PPOAdapter":
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError("stable-baselines3 is required for PPOAdapter") from exc

        model = PPO.load(path, env=env)
        return cls(model=model)

    def evaluate(self, env, episodes: int) -> dict[str, float]:
        rewards = []
        successes = []

        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            success = False
            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = bool(terminated or truncated)
                success = bool(info.get("is_success", False)) or success

            rewards.append(episode_reward)
            successes.append(1.0 if success else 0.0)

        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "success_rate": float(np.mean(successes)) if successes else 0.0,
        }
