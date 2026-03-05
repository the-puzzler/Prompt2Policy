from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from prompt2policy.envs.backend import EnvBackend
from prompt2policy.robots.specs import RobotSpec
from prompt2policy.world.specs import WorldSpec


class NativeMuJoCoBackend(EnvBackend):
    def __init__(
        self,
        scene_path: Path,
        robot_spec: RobotSpec,
        world_spec: WorldSpec,
        physics_timestep: float = 0.002,
        control_timestep: float = 0.02,
    ):
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("MuJoCo is required for NativeMuJoCoBackend") from exc

        self._mujoco = mujoco
        self.scene_path = scene_path
        self.robot_spec = robot_spec
        self.world_spec = world_spec
        self.physics_timestep = physics_timestep
        self.control_timestep = control_timestep
        self._n_substeps = max(1, int(round(self.control_timestep / self.physics_timestep)))

        self.model = self._load_model_with_fallback(scene_path)
        self.model.opt.timestep = physics_timestep
        self.data = mujoco.MjData(self.model)

        self.renderer = None

        self._joint_ids = [self._resolve_joint_id(name) for name in self.robot_spec.controlled_joints]
        self._actuator_ids = [self._resolve_actuator_id(name) for name in self.robot_spec.controlled_joints]
        self._joint_qpos_adr = [self.model.jnt_qposadr[joint_id] for joint_id in self._joint_ids]
        self._joint_qvel_adr = [self.model.jnt_dofadr[joint_id] for joint_id in self._joint_ids]

        self._eef_site_id = self._resolve_site_id(self.robot_spec.eef_site_name)
        self._goal_body_name = self._resolve_goal_body_name(world_spec)
        self._goal_position_hint = self._resolve_goal_position_hint(world_spec)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            np.random.seed(seed)

        self._mujoco.mj_resetData(self.model, self.data)

        home_key_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key_id >= 0:
            self._mujoco.mj_resetDataKeyframe(self.model, self.data, home_key_id)

        self._mujoco.mj_forward(self.model, self.data)
        return self.get_info()

    def step(self, action: np.ndarray) -> dict[str, Any]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        if any(act_id >= 0 for act_id in self._actuator_ids):
            ctrl = np.copy(self.data.ctrl)
            for idx, actuator_id in enumerate(self._actuator_ids):
                if actuator_id < 0:
                    continue
                low, high = self.model.actuator_ctrlrange[actuator_id]
                center = 0.5 * (low + high)
                span = 0.5 * (high - low)
                ctrl[actuator_id] = center + span * self.robot_spec.action_scale * float(action[idx])
            self.data.ctrl[:] = ctrl
        else:
            for idx, qpos_adr in enumerate(self._joint_qpos_adr):
                self.data.qpos[qpos_adr] = float(
                    self.data.qpos[qpos_adr] + self.robot_spec.action_scale * action[idx]
                )
            self._mujoco.mj_forward(self.model, self.data)

        for _ in range(self._n_substeps):
            self._mujoco.mj_step(self.model, self.data)

        return self.get_info()

    def render_rgb(self, camera_name: str, width: int, height: int) -> np.ndarray:
        if self.renderer is None:
            self.renderer = self._mujoco.Renderer(self.model, height=height, width=width)
        elif self.renderer.width != width or self.renderer.height != height:
            self.renderer.close()
            self.renderer = self._mujoco.Renderer(self.model, height=height, width=width)

        try:
            self.renderer.update_scene(self.data, camera=camera_name)
        except Exception:
            # Fallback to free camera when a named camera is unavailable.
            self.renderer.update_scene(self.data)
        rgb = self.renderer.render()
        return rgb.astype(np.uint8)

    def get_proprio(self, joint_names: Sequence[str]) -> np.ndarray:
        qpos_vals = []
        qvel_vals = []

        for joint_name in joint_names:
            joint_id = self._resolve_joint_id(joint_name)
            qpos_adr = self.model.jnt_qposadr[joint_id]
            qvel_adr = self.model.jnt_dofadr[joint_id]
            qpos_vals.append(float(self.data.qpos[qpos_adr]))
            qvel_vals.append(float(self.data.qvel[qvel_adr]))

        return np.asarray(qpos_vals + qvel_vals, dtype=np.float32)

    def get_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "time": float(self.data.time),
            "contact_pairs": self._contact_pairs(),
        }

        eef_position = self._eef_position()
        if eef_position is not None:
            info["eef_position"] = eef_position.tolist()

        goal_position = self._goal_position()
        if goal_position is not None:
            info["goal_position"] = goal_position.tolist()

        if eef_position is not None and goal_position is not None:
            info["distance_to_goal"] = float(np.linalg.norm(eef_position - goal_position))

        return info

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _resolve_joint_id(self, joint_name: str) -> int:
        joint_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Unknown joint name: {joint_name}")
        return joint_id

    def _resolve_actuator_id(self, actuator_name: str) -> int:
        return self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)

    def _resolve_site_id(self, site_name: str) -> int:
        return self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_SITE, site_name)

    def _resolve_goal_body_name(self, world_spec: WorldSpec) -> str | None:
        for obj in world_spec.objects:
            if obj.role == "goal":
                return obj.name
        return None

    def _resolve_goal_position_hint(self, world_spec: WorldSpec) -> np.ndarray | None:
        for obj in world_spec.objects:
            if obj.role == "goal":
                return np.array(obj.pos, dtype=np.float64)
        return None

    def _eef_position(self) -> np.ndarray | None:
        if self._eef_site_id < 0:
            return None
        return np.array(self.data.site_xpos[self._eef_site_id], dtype=np.float64)

    def _goal_position(self) -> np.ndarray | None:
        if self._goal_body_name is None:
            return self._goal_position_hint
        body_id = self._mujoco.mj_name2id(
            self.model,
            self._mujoco.mjtObj.mjOBJ_BODY,
            self._goal_body_name,
        )
        if body_id < 0:
            return self._goal_position_hint
        return np.array(self.data.xpos[body_id], dtype=np.float64)

    def _contact_pairs(self) -> set[tuple[str, str]]:
        contacts: set[tuple[str, str]] = set()
        for idx in range(self.data.ncon):
            contact = self.data.contact[idx]
            geom1_name = self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            contacts.add((geom1_name or "", geom2_name or ""))
        return contacts

    def _load_model_with_fallback(self, scene_path: Path):
        try:
            return self._mujoco.MjModel.from_xml_path(str(scene_path))
        except Exception:
            fallback_scene = Path(self.robot_spec.scene_xml_path)
            if not fallback_scene.is_absolute():
                fallback_scene = (Path.cwd() / fallback_scene).resolve()
            return self._mujoco.MjModel.from_xml_path(str(fallback_scene))
