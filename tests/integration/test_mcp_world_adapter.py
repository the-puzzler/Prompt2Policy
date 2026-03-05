from __future__ import annotations

from pathlib import Path

import pytest

from prompt2policy.world.mcp_adapter import MCPWorldAdapter


@pytest.mark.integration
def test_mcp_world_adapter_loads_scene_when_viewer_reachable():
    workspace_root = Path(__file__).resolve().parents[2]
    scene_path = workspace_root / "mujoco_menagerie" / "franka_fr3" / "scene.xml"

    adapter = MCPWorldAdapter(workspace_root=workspace_root)

    try:
        adapter.connect()
    except Exception as exc:
        pytest.skip(f"MCP viewer not reachable: {exc}")

    try:
        response = adapter.instantiate_scene(scene_path=scene_path, model_id="mcp_test_fr3")
        assert response.get("success", False)

        state = adapter.get_state()
        assert "qpos" in state
    finally:
        adapter.close()
