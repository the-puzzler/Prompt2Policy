from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from prompt2policy.curriculum.specs import TaskSpec


class TaskParseError(RuntimeError):
    pass


def parse_task_spec(payload: dict[str, Any]) -> TaskSpec:
    try:
        return TaskSpec.model_validate(payload)
    except ValidationError as exc:
        raise TaskParseError(str(exc)) from exc
