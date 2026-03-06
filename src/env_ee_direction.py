"""Gymnasium environment where action is desired end-effector movement direction."""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from reward import compute_reward
from viewer import MujocoViewer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_XML = os.path.join(PROJECT_ROOT, "mujoco_menagerie", "franka_fr3", "reach_scene.xml")

# Camera image dimensions
IMG_WIDTH = 64
IMG_HEIGHT = 64

# Workspace bounds for target randomization (reachable front workspace)
TARGET_LOW = np.array([0.2, -0.4, 0.1])
TARGET_HIGH = np.array([0.65, 0.4, 0.7])
TARGET_RADIUS_MIN = 0.25
TARGET_RADIUS_MAX = 0.75

MAX_EPISODE_STEPS = 200


class FrankaReachEEDirectionEnv(gym.Env):
    """FR3 reach task with Cartesian direction actions.

    Action is a 3D direction command in end-effector space.
    A damped least-squares Jacobian IK step maps EE delta to joint targets.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = 0.002
        self.n_substeps = 10

        self.n_joints = 7
        self.joint_names = [f"fr3_joint{i+1}" for i in range(self.n_joints)]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.joint_names]
        self.jnt_qpos_addrs = np.array([self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.jnt_dof_addrs = np.array([self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32)

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera")

        self._offscreen_renderer = None
        self._viewer = None

        # Action: desired EE direction delta (meters per env step).
        self.ee_delta_scale = 0.04
        self.ik_damping = 1e-3
        self.ik_gain = 0.8
        self.max_dq = 0.12
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        state_dim = self.n_joints * 2 + 3 + 3
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32),
            "image": spaces.Box(low=0, high=255, shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8),
        })

        self.home_qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853], dtype=np.float64)
        self.joint_lo = self.model.jnt_range[self.joint_ids, 0].astype(np.float64)
        self.joint_hi = self.model.jnt_range[self.joint_ids, 1].astype(np.float64)

        self._step_count = 0

    def _get_renderer(self):
        if self._offscreen_renderer is None:
            self._offscreen_renderer = mujoco.Renderer(self.model, height=IMG_HEIGHT, width=IMG_WIDTH)
        return self._offscreen_renderer

    def _get_joint_qpos(self):
        return self.data.qpos[self.jnt_qpos_addrs].copy()

    def _get_joint_qvel(self):
        return np.array([self.data.qvel[self.model.jnt_dofadr[jid]] for jid in self.joint_ids], dtype=np.float64)

    def _get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def _get_target_pos(self):
        return self.data.xpos[self.target_body_id].copy()

    def _randomize_target(self):
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
        target_pos = self._get_target_pos()
        state = np.concatenate([qpos, qvel, ee_pos, target_pos]).astype(np.float32)

        renderer = self._get_renderer()
        renderer.update_scene(self.data, camera=self.camera_id)
        image = renderer.render()
        return {"state": state, "image": image}

    def _ik_delta_to_dq(self, dx: np.ndarray) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.ee_site_id)
        j = jacp[:, self.jnt_dof_addrs]  # shape (3, 7)
        a = j @ j.T + (self.ik_damping ** 2) * np.eye(3, dtype=np.float64)
        dq = j.T @ np.linalg.solve(a, dx)
        dq = self.ik_gain * dq
        dq = np.clip(dq, -self.max_dq, self.max_dq)
        return dq

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[self.jnt_qpos_addrs] = self.home_qpos
        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = self.home_qpos[i]

        self._randomize_target()
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float64, copy=False)
        dx = action * self.ee_delta_scale

        q_now = self._get_joint_qpos()
        dq = self._ik_delta_to_dq(dx)
        q_target = np.clip(q_now + dq, self.joint_lo, self.joint_hi)

        for i, aid in enumerate(self.actuator_ids):
            self.data.ctrl[aid] = q_target[i]

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
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
                    self.model,
                    self.data,
                    self.camera_id,
                    title="FR3 Reach (EE Direction Control) - Camera POV",
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
