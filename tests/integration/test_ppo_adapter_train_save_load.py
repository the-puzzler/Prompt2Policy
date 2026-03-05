from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prompt2policy.algorithms.interface import AlgorithmConfig
from prompt2policy.algorithms.ppo_adapter import PPOAdapter


class TinyDictEnv:
    def __init__(self):
        from gymnasium import spaces

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8),
                "proprio": spaces.Box(-1, 1, shape=(14,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self._step = 0

    def reset(self, *, seed=None, options=None):
        del seed, options
        self._step = 0
        return self._obs(), {}

    def step(self, action):
        del action
        self._step += 1
        reward = 1.0
        terminated = self._step >= 4
        truncated = False
        info = {"is_success": terminated}
        return self._obs(), reward, terminated, truncated, info

    def _obs(self):
        return {
            "image": np.zeros((84, 84, 3), dtype=np.uint8),
            "proprio": np.zeros((14,), dtype=np.float32),
        }

    def close(self):
        return None


@pytest.mark.integration
def test_ppo_adapter_train_and_reload(tmp_path: Path):
    pytest.importorskip("stable_baselines3")

    env = TinyDictEnv()
    adapter = PPOAdapter()
    cfg = AlgorithmConfig(total_timesteps=64, n_steps=8, batch_size=8)
    adapter.train(env, env, cfg)

    checkpoint = tmp_path / "ppo_test_model"
    adapter.save(str(checkpoint))

    loaded = PPOAdapter.load(str(checkpoint), env=env)
    obs, _ = env.reset()
    action = loaded.predict(obs)
    assert action.shape == (7,)
