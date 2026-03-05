from __future__ import annotations

from pathlib import Path

from prompt2policy.algorithms.ppo_adapter import PPOAdapter
from prompt2policy.config.loader import load_robot_spec, load_task_spec
from prompt2policy.envs.factory import create_env
from prompt2policy.training.runtime_factory import build_scene_and_reward, build_single_env


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

    output_scene = workspace_root / ".tmp_eval" / "scene.xml"
    scene_path, reward_model = build_scene_and_reward(
        workspace_root=workspace_root,
        output_scene_path=output_scene,
        robot_spec=robot_spec,
        task_spec=task_spec,
    )

    env = build_single_env(
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
        create_env_fn=create_env,
    )

    adapter = PPOAdapter.load(str(checkpoint_path), env=env)
    metrics = adapter.evaluate(env, episodes=episodes)
    env.close()
    return metrics
