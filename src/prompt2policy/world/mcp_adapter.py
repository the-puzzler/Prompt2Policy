from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


class MCPWorldAdapter:
    def __init__(self, workspace_root: Path, host: str = "localhost", port: int = 8888):
        self.workspace_root = workspace_root
        self.host = host
        self.port = port
        self._client = None
        self._model_id: str | None = None

    def connect(self) -> None:
        if self._client is not None:
            return

        mcp_src = self.workspace_root / "mujoco_mcp" / "src"
        if str(mcp_src) not in sys.path:
            sys.path.append(str(mcp_src))

        from mujoco_mcp.viewer_client import MuJoCoViewerClient  # pylint: disable=import-error

        client = MuJoCoViewerClient(host=self.host, port=self.port)
        if not client.connect():
            raise RuntimeError("Failed to connect to MuJoCo MCP viewer server")
        self._client = client

    def instantiate_scene(self, scene_path: Path, model_id: str) -> dict[str, Any]:
        self.connect()
        assert self._client is not None

        response = self._client.load_model(str(scene_path), model_id=model_id)
        if not response.get("success", False):
            raise RuntimeError(f"MCP load_model failed: {response.get('error', 'unknown error')}")

        self._model_id = model_id
        return response

    def get_state(self) -> dict[str, Any]:
        self.connect()
        assert self._client is not None

        response = self._client.get_state(model_id=self._model_id)
        if not response.get("success", False):
            raise RuntimeError(f"MCP get_state failed: {response.get('error', 'unknown error')}")
        return response

    def set_joint_positions(self, positions: list[float]) -> None:
        self.connect()
        assert self._client is not None

        response = self._client.set_joint_positions(positions, model_id=self._model_id)
        if not response.get("success", False):
            raise RuntimeError(
                f"MCP set_joint_positions failed: {response.get('error', 'unknown error')}"
            )

    def capture_render(self, width: int, height: int) -> dict[str, Any]:
        self.connect()
        assert self._client is not None

        response = self._client.capture_render(model_id=self._model_id, width=width, height=height)
        if not response.get("success", False):
            raise RuntimeError(
                f"MCP capture_render failed: {response.get('error', 'unknown error')}"
            )
        return response

    def reset(self) -> None:
        self.connect()
        assert self._client is not None

        response = self._client.reset_simulation(model_id=self._model_id)
        if not response.get("success", False):
            raise RuntimeError(f"MCP reset failed: {response.get('error', 'unknown error')}")

    def close(self) -> None:
        if self._client is None:
            return

        try:
            if self._model_id:
                self._client.send_command({"type": "close_model", "model_id": self._model_id})
        finally:
            self._client.disconnect()
            self._client = None
            self._model_id = None
