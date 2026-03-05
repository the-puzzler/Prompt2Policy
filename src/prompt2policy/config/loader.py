from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from prompt2policy.config.models import ExperimentConfig
from prompt2policy.curriculum.specs import TaskSpec
from prompt2policy.robots.specs import RobotSpec


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_handle:
        data = yaml.safe_load(file_handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must contain a mapping at root")
    return data


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(payload, file_handle, sort_keys=False)


def load_experiment_config(path: Path) -> ExperimentConfig:
    data = load_yaml(path)
    return ExperimentConfig.model_validate(data)


def load_robot_spec(path: Path) -> RobotSpec:
    data = load_yaml(path)
    return RobotSpec.model_validate(data)


def load_task_spec(path: Path) -> TaskSpec:
    data = load_yaml(path)
    return TaskSpec.model_validate(data)


def save_task_spec(path: Path, task_spec: TaskSpec) -> None:
    save_yaml(path, task_spec.model_dump(mode="json"))
