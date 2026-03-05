from __future__ import annotations

from pathlib import Path

from prompt2policy.curriculum.specs import TaskSpec
from prompt2policy.envs.factory import create_env
from prompt2policy.envs.hybrid_env import HybridRobotEnv
from prompt2policy.rewards.model import RewardModel, compile_reward_model
from prompt2policy.robots.specs import RobotSpec
from prompt2policy.world.builder import WorldBuilder


def build_scene_and_reward(
    *,
    workspace_root: Path,
    output_scene_path: Path,
    robot_spec: RobotSpec,
    task_spec: TaskSpec,
) -> tuple[Path, RewardModel]:
    world_builder = WorldBuilder(workspace_root=workspace_root)
    scene_path = world_builder.compose_scene(
        world_spec=task_spec.world_spec,
        output_path=output_scene_path,
        default_camera=robot_spec.camera_defaults[0],
    )
    reward_model = compile_reward_model(task_spec.reward_spec)
    return scene_path, reward_model


def build_train_and_eval_envs(
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
    visualize: bool,
    create_env_fn=create_env,
) -> tuple[HybridRobotEnv, HybridRobotEnv]:
    env = create_env_fn(
        workspace_root=workspace_root,
        backend_name=backend_name,
        scene_path=scene_path,
        robot_spec=robot_spec,
        task_spec=task_spec,
        reward_model=reward_model,
        image_width=image_width,
        image_height=image_height,
        camera_name=camera_name,
        control_timestep=control_timestep,
        physics_timestep=physics_timestep,
        model_id=model_id,
        visualize=visualize,
    )

    eval_env = create_env_fn(
        workspace_root=workspace_root,
        backend_name=backend_name,
        scene_path=scene_path,
        robot_spec=robot_spec,
        task_spec=task_spec,
        reward_model=reward_model,
        image_width=image_width,
        image_height=image_height,
        camera_name=camera_name,
        control_timestep=control_timestep,
        physics_timestep=physics_timestep,
        model_id=f"{model_id}_eval",
        visualize=False,
    )
    return env, eval_env


def build_single_env(
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
    create_env_fn=create_env,
) -> HybridRobotEnv:
    return create_env_fn(
        workspace_root=workspace_root,
        backend_name=backend_name,
        scene_path=scene_path,
        robot_spec=robot_spec,
        task_spec=task_spec,
        reward_model=reward_model,
        image_width=image_width,
        image_height=image_height,
        camera_name=camera_name,
        control_timestep=control_timestep,
        physics_timestep=physics_timestep,
        model_id=model_id,
        visualize=False,
    )
