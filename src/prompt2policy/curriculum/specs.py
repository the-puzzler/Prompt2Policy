from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from prompt2policy.rewards.specs import RewardSpec
from prompt2policy.world.specs import WorldSpec


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    parent_task_id: str | None = None
    difficulty_level: float = 1.0
    goal_description: str
    world_spec: WorldSpec
    reward_spec: RewardSpec
    success_threshold_reward: float = 50.0
    max_episode_steps: int = 300
    eval_episodes: int = 5
