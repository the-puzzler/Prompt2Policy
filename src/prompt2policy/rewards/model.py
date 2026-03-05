from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .specs import RewardSpec, RewardTermSpec


class RewardModel(Protocol):
    def compute(
        self,
        observation: dict[str, np.ndarray],
        action: np.ndarray,
        next_observation: dict[str, np.ndarray],
        info: dict[str, Any],
    ) -> float:
        ...

    def is_success(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> bool:
        ...


@dataclass
class CompiledRewardModel:
    spec: RewardSpec

    def compute(
        self,
        observation: dict[str, np.ndarray],
        action: np.ndarray,
        next_observation: dict[str, np.ndarray],
        info: dict[str, Any],
    ) -> float:
        total = 0.0
        for term in self.spec.terms:
            total += term.weight * self._term_value(term, action, info)
        return float(total)

    def is_success(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> bool:
        termination_threshold = self.spec.termination_threshold
        if self.spec.success_metric == "distance_to_goal":
            distance = float(info.get("distance_to_goal", np.inf))
            return distance <= termination_threshold

        reward_value = float(info.get("episode_reward", -np.inf))
        return reward_value >= termination_threshold

    def _term_value(self, term: RewardTermSpec, action: np.ndarray, info: dict[str, Any]) -> float:
        if term.type == "distance":
            distance = float(info.get("distance_to_goal", np.inf))
            if not np.isfinite(distance):
                return -1e3
            return -distance

        if term.type == "contact":
            contact_pairs = info.get("contact_pairs", set())
            source = term.source or ""
            target = term.target or ""
            return 1.0 if (source, target) in contact_pairs or (target, source) in contact_pairs else 0.0

        if term.type == "time_penalty":
            return float(term.value if term.value is not None else -1.0)

        if term.type == "action_l2":
            return -float(np.square(action).sum())

        if term.type == "goal_bonus":
            distance = float(info.get("distance_to_goal", np.inf))
            threshold = float(term.threshold if term.threshold is not None else 0.05)
            return 1.0 if distance <= threshold else 0.0

        return 0.0


def compile_reward_model(spec: RewardSpec) -> RewardModel:
    return CompiledRewardModel(spec)
