from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PromotionRule:
    window_size: int = 3

    def should_promote(
        self,
        metrics_history: list[dict[str, float]],
        threshold_reward: float | None = None,
        promotion_threshold: float | None = None,
    ) -> bool:
        if not metrics_history:
            return False

        if promotion_threshold is None and threshold_reward is None:
            raise ValueError("Either 'promotion_threshold' or 'threshold_reward' must be provided.")

        resolved_threshold = (
            float(promotion_threshold)
            if promotion_threshold is not None
            else float(threshold_reward)
        )
        window = metrics_history[-self.window_size :]
        rewards = [float(item.get("mean_reward", 0.0)) for item in window]
        rolling_mean = float(np.mean(rewards))
        return rolling_mean >= resolved_threshold
