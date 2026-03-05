from __future__ import annotations

from pathlib import Path

from prompt2policy.algorithms.ppo_adapter import PPOAdapter
from prompt2policy.config.loader import load_robot_spec, load_task_spec
from prompt2policy.envs.factory import create_env
from prompt2policy.rewards.model import compile_reward_model
from prompt2policy.world.builder import WorldBuilder


def evaluate_checkpoint(
    *,
    workspace_root: Path,
    checkpoint_path: Path,
    robot_config_path: Path,
    task_path: Path,
    backend_name: str = "native",
    episodes: int = 5,
) -> dict[str, float]:
    robot_spec = load_robot_spec(robot_config_path)
    task_spec = load_task_spec(task_path)

    reward_model = compile_reward_model(task_spec.reward_spec)
    world_builder = WorldBuilder(workspace_root=workspace_root)

    output_scene = workspace_root / ".tmp_eval" / "scene.xml"
    scene_path = world_builder.compose_scene(
        world_spec=task_spec.world_spec,
        output_path=output_scene,
        default_camera=robot_spec.camera_defaults[0],
    )

    env = create_env(
        workspace_root=workspace_root,
        backend_name=backend_name,
        scene_path=scene_path,
        robot_spec=robot_spec,
        task_spec=task_spec,
        reward_model=reward_model,
        image_width=84,
        image_height=84,
        camera_name="third_person",
        control_timestep=0.02,
        physics_timestep=0.002,
        model_id="eval_model",
    )

    adapter = PPOAdapter.load(str(checkpoint_path), env=env)
    metrics = adapter.evaluate(env, episodes=episodes)
    env.close()
    return metrics
