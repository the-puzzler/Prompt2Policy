"""Gymnasium environment for Franka FR3 exploration task.

The workspace is divided into a coarse 3D grid. Each new cell the
end-effector enters gives +1 reward. The goal is to visit as many
cells as possible within one episode.

The grid is visualized in render() but never shown in the robot's
observation camera.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from viewer import MujocoViewer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_XML = os.path.join(PROJECT_ROOT, "envs", "explore_scene.xml")

IMG_WIDTH = 64
IMG_HEIGHT = 64
MAX_EPISODE_STEPS = 300  # 6 seconds at 20ms/step

# Grid workspace bounds (reachable volume for the FR3 end-effector)
GRID_LOW = np.array([-0.78, -0.80, 0.05])
GRID_HIGH = np.array([0.78, 0.79, 1.12])
CELL_SIZE = 0.15  # 15 cm cells -> 11x11x8 = 968 cells
MIN_MANIPULABILITY = 0.05  # reject poses near singularities

# Reward parameters
R_NOVEL = 1.0      # reward for discovering a new voxel
C_ENERGY = 1e-6   # penalty coefficient for torque squared
SUCCESS_COVERAGE = 0.30  # episode success threshold based on visited-cell coverage


class FrankaExploreEnv(gym.Env):
    """FR3 exploration: visit as many workspace grid cells as possible."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode=None,
        img_width: int = IMG_WIDTH,
        img_height: int = IMG_HEIGHT,
        cell_size: float = CELL_SIZE,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.cell_size = cell_size

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = 0.002
        self.n_substeps = 10  # 20 ms per step

        # Joint info
        self.n_joints = 7
        self.joint_names = [f"fr3_joint{i+1}" for i in range(self.n_joints)]
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.joint_names
        ]
        self.actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in self.joint_names
        ]
        self.joint_qpos_adr = np.array(
            [self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32
        )
        self.joint_qvel_adr = np.array(
            [self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32
        )
        self.joint_low = self.model.jnt_range[self.joint_ids, 0].copy()
        self.joint_high = self.model.jnt_range[self.joint_ids, 1].copy()
        self.actuator_ids_np = np.array(self.actuator_ids, dtype=np.int32)

        # EE site
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )

        # Cameras: front + top-down
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_camera"
        )
        self.top_camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top_camera"
        )

        # Lazy-init renderers (OpenGL context can't survive subprocess pickling)
        self._offscreen_renderer = None
        self._top_renderer = None
        self._vis_renderer = None
        self._viewer = None

        # Action: XYZ + orientation joystick — [dx, dy, dz, droll, dpitch, dyaw]
        self.pos_scale = 0.05     # max 5 cm per step
        self.rot_scale = 0.1     # max 0.1 rad per step
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Grid setup
        self.grid_low = GRID_LOW.copy()
        self.grid_high = GRID_HIGH.copy()
        self.grid_dims = np.ceil(
            (self.grid_high - self.grid_low) / self.cell_size
        ).astype(int)
        self.total_cells = int(np.prod(self.grid_dims))
        self.visited = np.zeros(self.grid_dims, dtype=bool)

        # Observation: state + dual-camera image (front + top stacked as 6 channels)
        state_dim = self.n_joints + 3 + 9 + 6  # qpos + ee_pos + ee_rot + ee_twist
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
                ),
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.img_height, self.img_width, 6),
                    dtype=np.uint8,
                ),
            }
        )

        # Home position
        self.home_qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
        self._step_count = 0

    # -- Renderers ----------------------------------------------------------

    def _get_renderer(self):
        if self._offscreen_renderer is None:
            self._offscreen_renderer = mujoco.Renderer(
                self.model, height=self.img_height, width=self.img_width
            )
        return self._offscreen_renderer

    def _get_top_renderer(self):
        if self._top_renderer is None:
            self._top_renderer = mujoco.Renderer(
                self.model, height=self.img_height, width=self.img_width
            )
        return self._top_renderer

    def _get_vis_renderer(self):
        if self._vis_renderer is None:
            self._vis_renderer = mujoco.Renderer(
                self.model, height=self.img_height, width=self.img_width
            )
        return self._vis_renderer

    # -- Helpers ------------------------------------------------------------

    def _get_joint_qpos(self):
        return self.data.qpos[self.joint_qpos_adr].copy()

    def _get_joint_qvel(self):
        return self.data.qvel[self.joint_qvel_adr].copy()

    def _get_ee_twist(self):
        """Compute EE linear + angular velocity (6D twist) via Jacobian."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        J = np.vstack([jacp[:, self.joint_qvel_adr], jacr[:, self.joint_qvel_adr]])
        return (J @ self._get_joint_qvel()).astype(np.float32)  # 6D

    def _get_ee_rot(self):
        """Return EE orientation as flattened 3x3 rotation matrix (9D)."""
        return self.data.site_xmat[self.ee_site_id].copy().astype(np.float32)  # 9D

    def _twist_to_joint_delta(self, dx, dy, dz, droll, dpitch, dyaw):
        """Convert XYZ + orientation delta to joint delta via full Jacobian pseudoinverse."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        J = np.vstack([
            jacp[:, self.joint_qvel_adr],  # 3x7 translational
            jacr[:, self.joint_qvel_adr],  # 3x7 rotational
        ])  # 6x7
        twist = np.array([dx, dy, dz, droll, dpitch, dyaw])
        return np.linalg.pinv(J) @ twist  # 7-dim joint delta

    def _get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def _get_manipulability(self):
        """Compute manipulability index sqrt(det(J @ J.T)) for the EE site."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        # Extract only the 7 arm joints
        J = jacp[:, self.joint_qvel_adr]
        JJT = J @ J.T
        return np.sqrt(max(np.linalg.det(JJT), 0.0))

    def _ee_to_grid(self, ee_pos):
        """Convert EE position to grid cell indices, or None if outside."""
        if np.any(ee_pos < self.grid_low) or np.any(ee_pos >= self.grid_high):
            return None
        idx = ((ee_pos - self.grid_low) / self.cell_size).astype(int)
        return tuple(np.clip(idx, 0, self.grid_dims - 1))

    # -- Grid visualization -------------------------------------------------

    def _add_grid_vis(self, scene):
        """Add translucent cubes for visited cells to a MjvScene."""
        half = self.cell_size / 2
        size = np.array([half, half, half])
        mat = np.eye(3).flatten()
        rgba = np.array([0.2, 0.8, 0.2, 0.25], dtype=np.float32)

        for idx in zip(*np.where(self.visited)):
            if scene.ngeom >= scene.maxgeom:
                break
            pos = self.grid_low + (np.array(idx) + 0.5) * self.cell_size
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=size,
                pos=pos,
                mat=mat,
                rgba=rgba,
            )
            scene.ngeom += 1

    # -- Observation --------------------------------------------------------

    def _get_obs(self):
        qpos = self._get_joint_qpos()
        ee_pos = self._get_ee_pos()
        ee_rot = self._get_ee_rot()
        ee_twist = self._get_ee_twist()

        state = np.concatenate([qpos, ee_pos, ee_rot, ee_twist]).astype(np.float32)

        # Clean render — robot never sees the grid; front + top stacked as 6 channels
        front_renderer = self._get_renderer()
        front_renderer.update_scene(self.data, camera=self.camera_id)
        front_img = front_renderer.render()

        top_renderer = self._get_top_renderer()
        top_renderer.update_scene(self.data, camera=self.top_camera_id)
        top_img = top_renderer.render()

        image = np.concatenate([front_img, top_img], axis=2)

        return {"state": state, "image": image}

    # -- Core env methods ---------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize start config, rejecting if EE outside grid or near singularity
        while True:
            random_qpos = self.np_random.uniform(self.joint_low, self.joint_high)
            for i, jid in enumerate(self.joint_ids):
                self.data.qpos[self.model.jnt_qposadr[jid]] = random_qpos[i]
            mujoco.mj_forward(self.model, self.data)
            if self._ee_to_grid(self._get_ee_pos()) is None:
                continue
            if self._get_manipulability() < MIN_MANIPULABILITY:
                continue
            break
        for i in range(self.n_joints):
            self.data.ctrl[self.actuator_ids[i]] = random_qpos[i]
        self._step_count = 0

        # Reset grid
        self.visited[:] = False
        cell = self._ee_to_grid(self._get_ee_pos())
        if cell is not None:
            self.visited[cell] = True

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        dx, dy, dz = action[:3] * self.pos_scale
        droll, dpitch, dyaw = action[3:] * self.rot_scale

        dq = self._twist_to_joint_delta(dx, dy, dz, droll, dpitch, dyaw)
        current_qpos = self._get_joint_qpos()
        target_qpos = np.clip(current_qpos + dq, self.joint_low, self.joint_high)
        self.data.ctrl[self.actuator_ids_np] = target_qpos

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Reward: R_novel for new voxel - c_energy * ||τ||^2
        ee_pos = self._get_ee_pos()
        cell = self._ee_to_grid(ee_pos)

        # Novel voxel bonus
        novelty_reward = 0.0
        if cell is not None and not self.visited[cell]:
            self.visited[cell] = True
            novelty_reward = R_NOVEL

        # Energy penalty: penalize high torques to discourage flailing
        torques = self.data.qfrc_actuator[self.joint_qvel_adr]
        energy_penalty = C_ENERGY * np.sum(torques ** 2)

        reward = novelty_reward - energy_penalty

        n_visited = int(self.visited.sum())
        coverage = n_visited / self.total_cells
        success = coverage >= SUCCESS_COVERAGE

        terminated = False
        truncated = self._step_count >= MAX_EPISODE_STEPS
        info = {
            "cells_visited": n_visited,
            "coverage": coverage,
            "is_success": bool(success),
            "novelty_reward": novelty_reward,
            "energy_penalty": energy_penalty,
        }

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # -- Rendering ----------------------------------------------------------

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = MujocoViewer(
                    self.model,
                    self.data,
                    self.camera_id,
                    title="FR3 Explore",
                    scene_callback=self._add_grid_vis,
                )
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            renderer = self._get_vis_renderer()
            renderer.update_scene(self.data, camera=self.camera_id)
            self._add_grid_vis(renderer.scene)
            return renderer.render()

    def close(self):
        if self._offscreen_renderer is not None:
            self._offscreen_renderer.close()
            self._offscreen_renderer = None
        if self._top_renderer is not None:
            self._top_renderer.close()
            self._top_renderer = None
        if self._vis_renderer is not None:
            self._vis_renderer.close()
            self._vis_renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
