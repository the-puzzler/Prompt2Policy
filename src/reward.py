"""Reward function for Franka FR3 reach task."""

import numpy as np


def compute_reward(ee_pos: np.ndarray, target_pos: np.ndarray) -> tuple[float, bool, dict]:
    """Compute reward based on distance between end-effector and target.

    Args:
        ee_pos: End-effector position (3,).
        target_pos: Target position (3,).

    Returns:
        reward: Scalar reward value.
        success: Whether the target was reached.
        info: Dict with extra metrics.
    """
    distance = np.linalg.norm(ee_pos - target_pos)
    success_threshold = 0.06

    # Dense reward: negative distance
    reward = -distance

    # Bonus for reaching the target
    success = distance < success_threshold
    if success:
        reward += 1.0

    return reward, success, {"distance": distance}
