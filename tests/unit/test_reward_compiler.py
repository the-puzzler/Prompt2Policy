from __future__ import annotations

import numpy as np

from prompt2policy.rewards.model import compile_reward_model
from prompt2policy.rewards.specs import RewardSpec


def test_reward_dsl_deterministic_computation():
    spec = RewardSpec.model_validate(
        {
            "terms": [
                {"type": "distance", "weight": 2.0, "target": "goal"},
                {"type": "action_l2", "weight": 0.5},
                {"type": "time_penalty", "weight": 0.1, "value": -1.0},
                {"type": "goal_bonus", "weight": 3.0, "target": "goal", "threshold": 0.2},
            ],
            "success_metric": "distance_to_goal",
            "success_threshold": 0.2,
        }
    )

    reward_model = compile_reward_model(spec)

    obs = {"image": np.zeros((84, 84, 3), dtype=np.uint8), "proprio": np.zeros((14,), dtype=np.float32)}
    next_obs = obs
    action = np.array([0.5, -0.5], dtype=np.float32)
    info = {"distance_to_goal": 0.1, "contact_pairs": set()}

    reward = reward_model.compute(obs, action, next_obs, info)

    expected_distance = 2.0 * (-0.1)
    expected_l2 = 0.5 * (-(0.25 + 0.25))
    expected_time = 0.1 * (-1.0)
    expected_bonus = 3.0 * 1.0

    assert np.isclose(reward, expected_distance + expected_l2 + expected_time + expected_bonus)
    assert reward_model.is_success(next_obs, info)
