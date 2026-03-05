from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["native", "mcp"] = "native"
    output_dir: str = "runs/default"
    camera_name: str = "third_person"
    image_width: int = 84
    image_height: int = 84
    physics_timestep: float = 0.002
    control_timestep: float = 0.02


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: str = "ppo"
    total_timesteps: int = 30_000
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 64
    gamma: float = 0.99


class CurriculumConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_stages: int = 1
    promotion_window_size: int = 3
    fallback_difficulty_increment: float = 0.1
    llm_provider: str = "mock"
    force_invalid_llm_output: bool = False


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "prompt2policy_fr3"
    seed: int = 0
    robot_config_path: str = "configs/robot_fr3.yaml"
    seed_task_path: str = "tasks/seed/fr3_reaching.yaml"
    runtime: RuntimeConfig = RuntimeConfig()
    training: TrainingConfig = TrainingConfig()
    curriculum: CurriculumConfig = CurriculumConfig()
