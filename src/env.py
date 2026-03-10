"""Gymnasium environment for Franka FR3 reach task with multi-input observations."""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from reward import compute_reward
from viewer import MujocoViewer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_XML = os.path.join(PROJECT_ROOT, "envs", "reach_scene.xml")

# Camera image dimensions
IMG_WIDTH = 64
IMG_HEIGHT = 64

# Workspace bounds for target randomization (reachable front workspace)
TARGET_LOW = np.array([0.2, -0.3, 0.15])
TARGET_HIGH = np.array([0.55, 0.3, 0.6])
TARGET_RADIUS_MIN = 0.3   # min distance from base (avoid too close)
TARGET_RADIUS_MAX = 0.6   # max distance from base (avoid unreachable)

MAX_EPISODE_STEPS = 200


class FrankaReachEnv(gym.Env):
    """FR3 reach task: move the end-effector to a green target sphere.

    Observation space (Dict / MultiInput):
        - "state": [joint_pos (7), joint_vel (7), ee_pos (3), target_pos (3)] = 20
        - "image": (64, 64, 3) uint8 camera image

    Action space: 7-dim continuous (delta joint position targets).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, img_width: int = IMG_WIDTH, img_height: int = IMG_HEIGHT):
        super().__init__()
        self.render_mode = render_mode
        self.img_width = int(img_width)
        self.img_height = int(img_height)

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = 0.002
        self.n_substeps = 10  # 20ms per step

        # Joint info
        self.n_joints = 7
        self.joint_names = [f"fr3_joint{i+1}" for i in range(self.n_joints)]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.joint_names]
        self.joint_qpos_adr = np.array([self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.joint_qvel_adr = np.array([self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.joint_low = self.model.jnt_range[self.joint_ids, 0].copy()
        self.joint_high = self.model.jnt_range[self.joint_ids, 1].copy()
        self.actuator_ids_np = np.array(self.actuator_ids, dtype=np.int32)

        # EE site
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

        # Target body
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")

        # Stereo cameras for depth perception
        self.left_camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "left_camera")
        self.right_camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "right_camera")

        # Lazy-init renderers (OpenGL context can't survive subprocess pickling)
        self._offscreen_renderer = None
        self._viewer = None

        # Action: delta joint positions, scaled
        self.action_scale = 0.05
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)

        # Observation
        state_dim = self.n_joints * 2 + 3  # qpos + qvel + ee_pos
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32),
            "image": spaces.Box(low=0, high=255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8),
        })

        # Home position from keyframe
        self.home_qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])

        self._step_count = 0

    def _get_renderer(self):
        if self._offscreen_renderer is None:
            self._offscreen_renderer = mujoco.Renderer(
                self.model,
                height=self.img_height,
                width=self.img_width,
            )
        return self._offscreen_renderer

    def _get_joint_qpos(self):
        return self.data.qpos[self.joint_qpos_adr].copy()

    def _get_joint_qvel(self):
        return self.data.qvel[self.joint_qvel_adr].copy()

    def _get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def _get_target_pos(self):
        return self.data.xpos[self.target_body_id].copy()

    def _randomize_target(self):
        # Rejection sample: box intersected with spherical shell around base
        while True:
            target_pos = self.np_random.uniform(low=TARGET_LOW, high=TARGET_HIGH)
            dist = np.linalg.norm(target_pos)
            if TARGET_RADIUS_MIN <= dist <= TARGET_RADIUS_MAX:
                break
        self.model.body_pos[self.target_body_id] = target_pos

    def _get_obs(self):
        qpos = self._get_joint_qpos()
        qvel = self._get_joint_qvel()
        ee_pos = self._get_ee_pos()

        state = np.concatenate([qpos, qvel, ee_pos]).astype(np.float32)

        # Render camera image
        renderer = self._get_renderer()
        renderer.update_scene(self.data, camera=self.camera_id)
        image = renderer.render()

        return {"state": state, "image": image}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Set home position
        for i, jid in enumerate(self.joint_ids):
            self.data.qpos[self.model.jnt_qposadr[jid]] = self.home_qpos[i]
            self.data.ctrl[self.actuator_ids[i]] = self.home_qpos[i]

        # Randomize target
        self._randomize_target()

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        delta = action * self.action_scale

        # Apply delta to current joint targets
        current_qpos = self._get_joint_qpos()
        target_qpos = np.clip(current_qpos + delta, self.joint_low, self.joint_high)

        # Set actuator controls (position targets)
        self.data.ctrl[self.actuator_ids_np] = target_qpos

        # Step simulation
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Compute reward
        ee_pos = self._get_ee_pos()
        target_pos = self._get_target_pos()
        reward, success, info = compute_reward(ee_pos, target_pos)

        terminated = success
        truncated = self._step_count >= MAX_EPISODE_STEPS
        info["is_success"] = success

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = MujocoViewer(
                    self.model, self.data, self.camera_id,
                    title="FR3 Reach - Camera POV",
                )
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = self._get_renderer()
            renderer.update_scene(self.data, camera=self.camera_id)
            return renderer.render()

    def close(self):
        if self._offscreen_renderer is not None:
            self._offscreen_renderer.close()
            self._offscreen_renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
