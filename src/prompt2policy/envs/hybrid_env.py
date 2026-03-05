from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - handled at runtime
    gym = None
    spaces = None

from prompt2policy.curriculum.specs import TaskSpec
from prompt2policy.envs.backend import EnvBackend
from prompt2policy.observations.assembler import ObservationAssembler
from prompt2policy.rewards.model import RewardModel
from prompt2policy.robots.specs import RobotSpec


class HybridRobotEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        backend: EnvBackend,
        robot_spec: RobotSpec,
        task_spec: TaskSpec,
        reward_model: RewardModel,
        image_width: int = 84,
        image_height: int = 84,
        camera_name: str = "third_person",
    ):
        if gym is None or spaces is None:
            raise RuntimeError("gymnasium is required to instantiate HybridRobotEnv")

        super().__init__()
        self.backend = backend
        self.robot_spec = robot_spec
        self.task_spec = task_spec
        self.reward_model = reward_model
        self.assembler = ObservationAssembler(
            robot_spec=robot_spec,
            camera_name=camera_name,
            image_width=image_width,
            image_height=image_height,
        )

        proprio_dim = len(robot_spec.controlled_joints) * 2
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(image_height, image_width, 3),
                    dtype=np.uint8,
                ),
                "proprio": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(proprio_dim,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(robot_spec.controlled_joints),),
            dtype=np.float32,
        )

        self._episode_reward = 0.0
        self._step_count = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        super().reset(seed=seed)

        self._episode_reward = 0.0
        self._step_count = 0
        info = self.backend.reset(seed=seed)
        obs = self.assembler.assemble(self.backend)
        info.update({"task_id": self.task_spec.task_id})
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        previous_obs = self.assembler.assemble(self.backend)

        backend_info = self.backend.step(action)
        obs = self.assembler.assemble(self.backend)

        reward = self.reward_model.compute(previous_obs, action, obs, backend_info)
        self._episode_reward += reward
        self._step_count += 1

        success_info = dict(backend_info)
        success_info["episode_reward"] = self._episode_reward
        terminated = self.reward_model.is_success(obs, success_info)
        truncated = self._step_count >= self.task_spec.max_episode_steps

        info = dict(backend_info)
        info.update(
            {
                "task_id": self.task_spec.task_id,
                "episode_reward": self._episode_reward,
                "step_count": self._step_count,
                "is_success": bool(terminated),
            }
        )

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self.backend.render_rgb(
            camera_name=self.assembler.camera_name,
            width=self.assembler.image_width,
            height=self.assembler.image_height,
        )

    def close(self):
        self.backend.close()
