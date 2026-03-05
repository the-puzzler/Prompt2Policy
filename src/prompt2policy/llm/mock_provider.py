from __future__ import annotations

from copy import deepcopy
from typing import Any


class MockLLMProvider:
    """Deterministic mock provider for stage generation.

    It receives a payload containing previous stage/task and metrics, and returns
    a conservative difficulty increase while keeping schema-compatible fields.
    """

    def __init__(self, force_invalid_output: bool = False):
        self.force_invalid_output = force_invalid_output

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.force_invalid_output:
            return {"bad": "payload"}

        previous_task = deepcopy(payload["previous_task"])
        metrics = payload.get("metrics", {})

        next_task = deepcopy(previous_task)
        next_task["parent_task_id"] = previous_task["task_id"]
        next_task["task_id"] = f"{previous_task['task_id']}_lvl{int(previous_task.get('difficulty_level', 1.0) * 10) + 1}"
        next_task["difficulty_level"] = float(previous_task.get("difficulty_level", 1.0) + 0.1)
        next_task["goal_description"] = (
            f"{previous_task['goal_description']} | harder variant after mean_reward={metrics.get('mean_reward', 0.0):.2f}"
        )
        next_task["success_threshold_reward"] = float(previous_task.get("success_threshold_reward", 50.0) + 5.0)

        # Increase goal difficulty by shifting goal object slightly if available.
        world_spec = next_task.get("world_spec", {})
        for obj in world_spec.get("objects", []):
            if obj.get("role") == "goal":
                pos = obj.get("pos", [0.5, 0.0, 0.4])
                obj["pos"] = [float(pos[0] + 0.02), float(pos[1] + 0.02), float(pos[2])]

        return next_task
