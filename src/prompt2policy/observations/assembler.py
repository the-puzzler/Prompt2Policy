from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prompt2policy.envs.backend import EnvBackend
from prompt2policy.robots.specs import RobotSpec


@dataclass
class ObservationAssembler:
    robot_spec: RobotSpec
    camera_name: str = "third_person"
    image_width: int = 84
    image_height: int = 84

    def assemble(self, backend: EnvBackend) -> dict[str, np.ndarray]:
        image = backend.render_rgb(
            camera_name=self.camera_name,
            width=self.image_width,
            height=self.image_height,
        )
        image = self._resize_if_needed(image)

        proprio = backend.get_proprio(self.robot_spec.controlled_joints)
        return {
            "image": image.astype(np.uint8),
            "proprio": proprio.astype(np.float32),
        }

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        if image.shape[:2] == (self.image_height, self.image_width):
            return image

        src_h, src_w = image.shape[:2]
        y_idx = (np.linspace(0, src_h - 1, self.image_height)).astype(np.int32)
        x_idx = (np.linspace(0, src_w - 1, self.image_width)).astype(np.int32)
        return image[y_idx][:, x_idx]
