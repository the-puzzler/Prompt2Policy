from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


RewardTermType = Literal["distance", "contact", "time_penalty", "action_l2", "goal_bonus"]


class RewardTermSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: RewardTermType
    weight: float = 1.0
    source: str | None = None
    target: str | None = None
    threshold: float | None = None
    value: float | None = None

    @model_validator(mode="after")
    def _validate_term_requirements(self) -> "RewardTermSpec":
        if self.type in {"distance", "contact", "goal_bonus"}:
            if not self.target:
                raise ValueError(f"Reward term '{self.type}' requires 'target'")
        if self.type == "contact" and not self.source:
            raise ValueError("Reward term 'contact' requires 'source'")
        if self.type == "goal_bonus" and self.threshold is None:
            raise ValueError("Reward term 'goal_bonus' requires 'threshold'")
        if self.type == "time_penalty" and self.value is None:
            object.__setattr__(self, "value", -1.0)
        return self


class RewardSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    terms: list[RewardTermSpec] = Field(default_factory=list)
    success_metric: Literal["distance_to_goal", "reward"] = "distance_to_goal"
    success_threshold: float = 0.05

    @property
    def termination_threshold(self) -> float:
        return self.success_threshold
