from __future__ import annotations

from pathlib import Path

from prompt2policy.config.loader import load_task_spec, save_task_spec
from prompt2policy.curriculum.generator import LLMCurriculumGenerator
from prompt2policy.llm.mock_provider import MockLLMProvider


def generate_next_stage_task(
    *,
    previous_task_path: Path,
    output_path: Path,
    mean_reward: float,
    force_invalid_output: bool = False,
) -> None:
    previous_task = load_task_spec(previous_task_path)
    provider = MockLLMProvider(force_invalid_output=force_invalid_output)
    generator = LLMCurriculumGenerator(provider=provider)

    metrics = {"mean_reward": mean_reward}
    next_task = generator.generate_next_stage(previous_task, metrics)
    save_task_spec(output_path, next_task)
