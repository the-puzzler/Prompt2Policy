from __future__ import annotations

from prompt2policy.curriculum.promotion import PromotionRule


def test_promotion_rule_uses_rolling_mean_window():
    rule = PromotionRule(window_size=2)

    history = [
        {"mean_reward": 1.0},
        {"mean_reward": 5.0},
        {"mean_reward": 7.0},
    ]

    assert rule.should_promote(history, threshold_reward=6.0)
    assert not rule.should_promote(history, threshold_reward=6.5)
