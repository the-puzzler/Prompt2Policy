from __future__ import annotations

import base64
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
import time
from typing import Any

import numpy as np

from prompt2policy.envs.backend import EnvBackend
from prompt2policy.robots.specs import RobotSpec
from prompt2policy.world.mcp_adapter import MCPWorldAdapter
from prompt2policy.world.specs import WorldSpec


class MCPEnvBackend(EnvBackend):
    def __init__(
        self,
        workspace_root: Path,
        scene_path: Path,
        robot_spec: RobotSpec,
        world_spec: WorldSpec,
        model_id: str,
        control_timestep: float = 0.02,
    ):
        self.robot_spec = robot_spec
        self.control_timestep = control_timestep
        self.mcp = MCPWorldAdapter(workspace_root=workspace_root)
        self.mcp.instantiate_scene(scene_path=scene_path, model_id=model_id)

        self._joint_count = len(robot_spec.controlled_joints)
        self._goal_position_hint = self._extract_goal_position_hint(world_spec)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        del seed
        self.mcp.reset()
        return self.get_info()

    def step(self, action: np.ndarray) -> dict[str, Any]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        self.mcp.set_joint_positions(action.tolist())
        time.sleep(self.control_timestep)
        return self.get_info()

    def render_rgb(self, camera_name: str, width: int, height: int) -> np.ndarray:
        del camera_name
        render = self.mcp.capture_render(width=width, height=height)
        image_data = render.get("image_data")
        if not image_data:
            return np.zeros((height, width, 3), dtype=np.uint8)

        try:
            from PIL import Image
        except ImportError:
            return np.zeros((height, width, 3), dtype=np.uint8)

        decoded = base64.b64decode(image_data)
        image = Image.open(BytesIO(decoded)).convert("RGB")
        image = image.resize((width, height))
        return np.asarray(image, dtype=np.uint8)

    def get_proprio(self, joint_names: Sequence[str]) -> np.ndarray:
        del joint_names

        state = self.mcp.get_state()
        qpos = np.array(state.get("qpos", []), dtype=np.float32)
        qvel = np.array(state.get("qvel", []), dtype=np.float32)

        if qpos.size < self._joint_count:
            qpos = np.pad(qpos, (0, self._joint_count - qpos.size))
        if qvel.size < self._joint_count:
            qvel = np.pad(qvel, (0, self._joint_count - qvel.size))

        qpos = qpos[: self._joint_count]
        qvel = qvel[: self._joint_count]
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def get_info(self) -> dict[str, Any]:
        state = self.mcp.get_state()
        info: dict[str, Any] = {
            "time": float(state.get("time", 0.0)),
            "contact_pairs": set(),
        }

        xpos = state.get("xpos")
        if isinstance(xpos, list) and len(xpos) >= 6:
            eef = np.array(xpos[:3], dtype=np.float64)
            goal = np.array(xpos[3:6], dtype=np.float64)
            info["eef_position"] = eef.tolist()
            info["goal_position"] = goal.tolist()
            info["distance_to_goal"] = float(np.linalg.norm(eef - goal))
            return info

        # Fallback when viewer state does not expose xpos.
        qpos = np.array(state.get("qpos", []), dtype=np.float64)
        if qpos.size >= 3 and self._goal_position_hint is not None:
            eef = qpos[:3]
            goal = self._goal_position_hint
            info["eef_position"] = eef.tolist()
            info["goal_position"] = goal.tolist()
            info["distance_to_goal"] = float(np.linalg.norm(eef - goal))
        elif self._goal_position_hint is not None:
            info["goal_position"] = self._goal_position_hint.tolist()
            info["distance_to_goal"] = float(np.linalg.norm(self._goal_position_hint))

        return info

    def close(self) -> None:
        self.mcp.close()

    def _extract_goal_position_hint(self, world_spec: WorldSpec) -> np.ndarray | None:
        for obj in world_spec.objects:
            if obj.role == "goal":
                return np.array(obj.pos, dtype=np.float64)
        return None
