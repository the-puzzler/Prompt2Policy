from __future__ import annotations

import logging
from pathlib import Path

from prompt2policy.curriculum.specs import TaskSpec
from prompt2policy.envs.hybrid_env import HybridRobotEnv
from prompt2policy.envs.mcp_backend import MCPEnvBackend
from prompt2policy.envs.native_backend import NativeMuJoCoBackend
from prompt2policy.rewards.model import RewardModel
from prompt2policy.robots.specs import RobotSpec


LOGGER = logging.getLogger("prompt2policy.env_factory")


def create_env(
    *,
    workspace_root: Path,
    backend_name: str,
    scene_path: Path,
    robot_spec: RobotSpec,
    task_spec: TaskSpec,
    reward_model: RewardModel,
    image_width: int,
    image_height: int,
    camera_name: str,
    control_timestep: float,
    physics_timestep: float,
    model_id: str,
) -> HybridRobotEnv:
    LOGGER.info(
        "Creating env | backend=%s | model_id=%s | scene_path=%s",
        backend_name,
        model_id,
        scene_path,
    )
    if backend_name == "native":
        backend = NativeMuJoCoBackend(
            scene_path=scene_path,
            robot_spec=robot_spec,
            world_spec=task_spec.world_spec,
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
        )
    elif backend_name == "mcp":
        backend = MCPEnvBackend(
            workspace_root=workspace_root,
            scene_path=scene_path,
            robot_spec=robot_spec,
            model_id=model_id,
            control_timestep=control_timestep,
        )
    else:
        raise ValueError(f"Unsupported backend '{backend_name}'. Expected 'native' or 'mcp'.")

    LOGGER.debug(
        "Env backend ready | camera=%s | image=%sx%s | joints=%d",
        camera_name,
        image_width,
        image_height,
        len(robot_spec.controlled_joints),
    )
    return HybridRobotEnv(
        backend=backend,
        robot_spec=robot_spec,
        task_spec=task_spec,
        reward_model=reward_model,
        image_width=image_width,
        image_height=image_height,
        camera_name=camera_name,
    )
