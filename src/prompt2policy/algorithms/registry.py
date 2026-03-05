from __future__ import annotations

from prompt2policy.algorithms.interface import AlgorithmAdapter
from prompt2policy.algorithms.ppo_adapter import PPOAdapter


def create_algorithm(name: str) -> AlgorithmAdapter:
    normalized = name.lower().strip()
    if normalized == "ppo":
        return PPOAdapter()
    raise ValueError(f"Unsupported algorithm '{name}'. Only 'ppo' is implemented in v1.")
