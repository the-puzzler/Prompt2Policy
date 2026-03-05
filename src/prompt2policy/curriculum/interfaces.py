from __future__ import annotations

from typing import Protocol

from prompt2policy.curriculum.specs import TaskSpec


class CurriculumGenerator(Protocol):
    def generate_next_stage(self, previous_stage: TaskSpec, metrics: dict[str, float]) -> TaskSpec:
        ...
