from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CameraSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    pos: list[float] = Field(default_factory=lambda: [1.2, -1.2, 0.9], min_length=3, max_length=3)
    quat: list[float] | None = Field(default=None, min_length=4, max_length=4)
    fovy: float = 45.0
    width: int = 84
    height: int = 84


class ObjectSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    shape: Literal["sphere", "box", "cylinder"] = "sphere"
    size: list[float] = Field(default_factory=lambda: [0.03], min_length=1)
    pos: list[float] = Field(default_factory=lambda: [0.5, 0.0, 0.4], min_length=3, max_length=3)
    rgba: list[float] = Field(default_factory=lambda: [0.2, 0.8, 0.2, 1.0], min_length=4, max_length=4)
    role: Literal["goal", "obstacle", "prop"] = "prop"


class WorldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_scene: str
    objects: list[ObjectSpec] = Field(default_factory=list)
    cameras: list[CameraSpec] = Field(default_factory=list)
    mcp_override_payload: dict[str, Any] | None = None
