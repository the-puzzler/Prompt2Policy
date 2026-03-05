from __future__ import annotations

import pytest
from pydantic import ValidationError

from prompt2policy.curriculum.specs import TaskSpec
from prompt2policy.rewards.specs import RewardSpec


def test_task_spec_validation_rejects_missing_goal_description():
    bad_payload = {
        "task_id": "bad_task",
        "world_spec": {
            "base_scene": "mujoco_menagerie/franka_fr3/scene.xml",
            "objects": [],
            "cameras": [],
            "mcp_override_payload": None,
        },
        "reward_spec": {
            "terms": [],
            "success_metric": "distance_to_goal",
            "success_threshold": 0.05,
        },
        "success_threshold_reward": 1.0,
        "max_episode_steps": 100,
        "eval_episodes": 2,
    }

    with pytest.raises(ValidationError):
        TaskSpec.model_validate(bad_payload)


def test_reward_spec_validation_rejects_contact_without_source():
    bad_reward = {
        "terms": [
            {
                "type": "contact",
                "weight": 1.0,
                "target": "goal_sphere",
            }
        ],
        "success_metric": "distance_to_goal",
        "success_threshold": 0.05,
    }

    with pytest.raises(ValidationError):
        RewardSpec.model_validate(bad_reward)
