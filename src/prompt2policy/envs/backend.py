from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np


class EnvBackend(Protocol):
    def reset(self, seed: int | None = None) -> dict[str, Any]:
        ...

    def step(self, action: np.ndarray) -> dict[str, Any]:
        ...

    def render_rgb(self, camera_name: str, width: int, height: int) -> np.ndarray:
        ...

    def get_proprio(self, joint_names: Sequence[str]) -> np.ndarray:
        ...

    def get_info(self) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...
