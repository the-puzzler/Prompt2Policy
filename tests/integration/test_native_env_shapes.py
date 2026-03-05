from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prompt2policy.config.loader import load_robot_spec, load_task_spec
from prompt2policy.envs.factory import create_env
from prompt2policy.rewards.model import compile_reward_model
from prompt2policy.world.builder import WorldBuilder


@pytest.mark.integration
def test_native_env_observation_shapes():
    pytest.importorskip("mujoco")

    workspace_root = Path(__file__).resolve().parents[2]
    robot_spec = load_robot_spec(workspace_root / "configs" / "robot_fr3.yaml")
    task_spec = load_task_spec(workspace_root / "tasks" / "seed" / "fr3_reaching.yaml")

    world_builder = WorldBuilder(workspace_root=workspace_root)
    scene_path = world_builder.compose_scene(
        world_spec=task_spec.world_spec,
        output_path=workspace_root / ".tmp_test" / "native_scene.xml",
        default_camera=robot_spec.camera_defaults[0],
    )

    reward_model = compile_reward_model(task_spec.reward_spec)

    try:
        env = create_env(
            workspace_root=workspace_root,
            backend_name="native",
            scene_path=scene_path,
            robot_spec=robot_spec,
            task_spec=task_spec,
            reward_model=reward_model,
            image_width=84,
            image_height=84,
            camera_name="third_person",
            control_timestep=0.02,
            physics_timestep=0.002,
            model_id="native_test",
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Native env unavailable in current runtime: {exc}")

    obs, _ = env.reset()
    assert obs["image"].shape == (84, 84, 3)
    assert obs["image"].dtype == np.uint8
    assert obs["proprio"].shape == (14,)

    action = np.zeros((7,), dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs["image"].shape == (84, 84, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "task_id" in info

    env.close()
