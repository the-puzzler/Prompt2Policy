from __future__ import annotations

from typing import Any, Protocol


class LLMProvider(Protocol):
    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...
