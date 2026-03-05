from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class AlgorithmConfig:
    total_timesteps: int = 50_000
    algorithm_params: dict[str, Any] = field(default_factory=dict)
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    gamma: float = 0.99
    seed: int = 0
    device: str = "auto"
    verbose: int = 0

    def param(self, key: str, default: Any = None) -> Any:
        if key in self.algorithm_params:
            return self.algorithm_params[key]
        return getattr(self, key, default)


class AlgorithmAdapter(Protocol):
    def train(self, env: Any, eval_env: Any, config: AlgorithmConfig) -> dict[str, float]:
        ...

    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        ...

    def save(self, path: str) -> None:
        ...

    @classmethod
    def load(cls, path: str, env: Any | None = None) -> "AlgorithmAdapter":
        ...
