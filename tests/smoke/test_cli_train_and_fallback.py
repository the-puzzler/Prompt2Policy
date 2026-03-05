from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from prompt2policy.cli.__main__ import main
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


@pytest.mark.smoke
def test_cli_one_stage_train_generates_artifacts(monkeypatch, tmp_path):
    workspace_root = Path(__file__).resolve().parents[2]
    source_config = workspace_root / "configs" / "experiment.yaml"

    experiment = yaml.safe_load(source_config.read_text())
    experiment["runtime"]["output_dir"] = str(tmp_path / "runs")
    experiment["curriculum"]["max_stages"] = 1

    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment, sort_keys=False))

    monkeypatch.setattr(
        "prompt2policy.training.curriculum_runner.create_env",
        lambda **kwargs: DummyEnv(),
    )
    monkeypatch.setattr(
        "prompt2policy.training.curriculum_runner.create_algorithm",
        lambda _: DummyAlgo(),
    )

    monkeypatch.setattr(sys, "argv", ["prompt2policy", "train", "--experiment", str(experiment_path)])
    exit_code = main()
    assert exit_code == 0

    output_dir = Path(experiment["runtime"]["output_dir"])
    assert (output_dir / "stage_000" / "metrics.json").exists()
    assert (output_dir / "stage_000" / "policy_checkpoint.zip").exists()


@pytest.mark.smoke
def test_fallback_task_generation_when_llm_output_invalid(monkeypatch, temp_experiment_paths):
    workspace_root = Path(__file__).resolve().parents[2]

    cfg_path = temp_experiment_paths["experiment_path"]
    config = yaml.safe_load(cfg_path.read_text())
    config["curriculum"]["force_invalid_llm_output"] = True
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))

    monkeypatch.setattr(
        "prompt2policy.training.curriculum_runner.create_env",
        lambda **kwargs: DummyEnv(),
    )
    monkeypatch.setattr(
        "prompt2policy.training.curriculum_runner.create_algorithm",
        lambda _: DummyAlgo(),
    )

    summary = CurriculumRunner(
        workspace_root=workspace_root,
        experiment_config_path=cfg_path,
    ).run()

    output_dir = temp_experiment_paths["output_dir"]
    generated_tasks = list((output_dir / "generated_tasks").glob("*.yaml"))
    assert summary["stages_completed"] == 2
    assert generated_tasks
    assert any("fallback" in path.stem for path in generated_tasks)
