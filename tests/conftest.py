from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class DummyDictEnv:
    def __init__(self):
        import gymnasium as gym
        from gymnasium import spaces

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8),
                "proprio": spaces.Box(-1, 1, shape=(14,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1, 1, shape=(7,), dtype=np.float32)
        self._step = 0

    def reset(self, *, seed=None, options=None):
        del seed, options
        self._step = 0
        obs = {
            "image": np.zeros((84, 84, 3), dtype=np.uint8),
            "proprio": np.zeros((14,), dtype=np.float32),
        }
        return obs, {}

    def step(self, action):
        del action
        self._step += 1
        obs = {
            "image": np.zeros((84, 84, 3), dtype=np.uint8),
            "proprio": np.zeros((14,), dtype=np.float32),
        }
        reward = 1.0
        terminated = self._step >= 5
        truncated = False
        info = {"is_success": terminated}
        return obs, reward, terminated, truncated, info

    def close(self):
        return None


class DummyAlgo:
    def __init__(self):
        self.saved_path = None

    def train(self, env, eval_env, config):
        del env, eval_env, config
        return {"timesteps": 10.0}

    def evaluate(self, env, episodes):
        del env
        return {"mean_reward": 10.0, "std_reward": 0.0, "success_rate": 1.0, "episodes": episodes}

    def save(self, path):
        self.saved_path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(f"{path}.zip").write_text("dummy")

    def predict(self, observation, deterministic=True):
        del observation, deterministic
        return np.zeros((7,), dtype=np.float32)


@pytest.fixture
def temp_experiment_paths(tmp_path: Path):
    robot_cfg = ROOT / "configs" / "robot_fr3.yaml"
    seed_task = ROOT / "tasks" / "seed" / "fr3_reaching.yaml"

    experiment = {
        "name": "test_run",
        "seed": 0,
        "robot_config_path": str(robot_cfg),
        "seed_task_path": str(seed_task),
        "runtime": {
            "backend": "native",
            "output_dir": str(tmp_path / "runs"),
            "camera_name": "third_person",
            "image_width": 84,
            "image_height": 84,
            "physics_timestep": 0.002,
            "control_timestep": 0.02,
        },
        "training": {
            "algorithm": "ppo",
            "total_timesteps": 8,
            "learning_rate": 3e-4,
            "n_steps": 8,
            "batch_size": 8,
            "gamma": 0.99,
        },
        "curriculum": {
            "max_stages": 2,
            "promotion_window_size": 1,
            "fallback_difficulty_increment": 0.1,
            "llm_provider": "mock",
            "force_invalid_llm_output": False,
        },
    }

    experiment_path = tmp_path / "experiment.yaml"
    with experiment_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(experiment, handle, sort_keys=False)

    return {
        "experiment_path": experiment_path,
        "output_dir": Path(experiment["runtime"]["output_dir"]),
    }
