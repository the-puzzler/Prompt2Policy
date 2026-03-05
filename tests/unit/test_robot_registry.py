from __future__ import annotations

import pytest

from prompt2policy.robots.registry import get_default_registry


def test_robot_registry_returns_fr3_and_rejects_unknown():
    registry = get_default_registry()

    fr3 = registry.get("fr3")
    assert fr3.menagerie_model == "franka_fr3"
    assert len(fr3.controlled_joints) == 7

    with pytest.raises(KeyError):
        registry.get("does_not_exist")
