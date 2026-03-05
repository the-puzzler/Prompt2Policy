from __future__ import annotations

from dataclasses import dataclass, field

from prompt2policy.robots.specs import RobotSpec
from prompt2policy.world.specs import CameraSpec


DEFAULT_FR3_SPEC = RobotSpec(
    robot_id="fr3",
    menagerie_model="franka_fr3",
    scene_xml_path="mujoco_menagerie/franka_fr3/scene.xml",
    controlled_joints=[
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
    ],
    action_mode="position",
    camera_defaults=[
        CameraSpec(
            name="third_person",
            pos=[1.2, -1.2, 0.9],
            quat=[0.856, 0.217, 0.372, 0.274],
            fovy=45.0,
            width=84,
            height=84,
        )
    ],
    eef_site_name="attachment_site",
    action_scale=0.2,
)


@dataclass
class RobotRegistry:
    _specs: dict[str, RobotSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._specs:
            self.register(DEFAULT_FR3_SPEC)

    def register(self, spec: RobotSpec) -> None:
        self._specs[spec.robot_id] = spec

    def get(self, robot_id: str) -> RobotSpec:
        if robot_id not in self._specs:
            available = ", ".join(sorted(self._specs))
            raise KeyError(f"Unknown robot_id '{robot_id}'. Available: {available}")
        return self._specs[robot_id]


def get_default_registry() -> RobotRegistry:
    return RobotRegistry()
