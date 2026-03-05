from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prompt2policy.training.curriculum_runner import CurriculumRunner


class DummyEnv:
    def reset(self, *, seed=None, options=None):
        del seed, options
        obs = {
            "image": np.zeros((84, 84, 3), dtype=np.uint8),
            "proprio": np.zeros((14,), dtype=np.float32),
        }
        return obs, {}

    def step(self, action):
        del action
        obs = {
            "image": np.zeros((84, 84, 3), dtype=np.uint8),
            "proprio": np.zeros((14,), dtype=np.float32),
        }
        return obs, 1.0, True, False, {"is_success": True}

    def close(self):
        return None


class DummyAlgo:
    def train(self, env, eval_env, config):
        del env, eval_env, config
        return {"timesteps": 10.0}

    def evaluate(self, env, episodes):
        del env, episodes
        return {"mean_reward": 10.0, "std_reward": 0.0, "success_rate": 1.0}

    def save(self, path):
        Path(f"{path}.zip").write_text("dummy")


@pytest.mark.integration
def test_curriculum_runner_promotes_stage(monkeypatch, temp_experiment_paths):
    workspace_root = Path(__file__).resolve().parents[2]

    monkeypatch.setattr(
        "prompt2policy.training.curriculum_runner.create_env",
        lambda **kwargs: DummyEnv(),
    )
    monkeypatch.setattr(
        "prompt2policy.training.curriculum_runner.create_algorithm",
        lambda _: DummyAlgo(),
    )

    runner = CurriculumRunner(
        workspace_root=workspace_root,
        experiment_config_path=temp_experiment_paths["experiment_path"],
    )
    summary = runner.run()

    output_dir = temp_experiment_paths["output_dir"]
    assert summary["stages_completed"] == 2
    assert (output_dir / "stage_000" / "metrics.json").exists()
    assert (output_dir / "stage_001" / "metrics.json").exists()
    assert any((output_dir / "generated_tasks").glob("*.yaml"))
