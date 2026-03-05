from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from prompt2policy.world.specs import CameraSpec


class RobotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robot_id: str
    menagerie_model: str
    scene_xml_path: str
    controlled_joints: list[str] = Field(min_length=1)
    action_mode: str = "position"
    camera_defaults: list[CameraSpec] = Field(default_factory=list)
    eef_site_name: str = "attachment_site"
    action_scale: float = 0.2
