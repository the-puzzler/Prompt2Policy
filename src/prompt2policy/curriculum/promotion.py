from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PromotionRule:
    window_size: int = 3

    def should_promote(self, metrics_history: list[dict[str, float]], threshold_reward: float) -> bool:
        if not metrics_history:
            return False

        window = metrics_history[-self.window_size :]
        rewards = [float(item.get("mean_reward", 0.0)) for item in window]
        rolling_mean = float(np.mean(rewards))
        return rolling_mean >= threshold_reward
