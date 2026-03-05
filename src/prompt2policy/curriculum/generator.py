from __future__ import annotations

from copy import deepcopy
import logging

from prompt2policy.curriculum.interfaces import CurriculumGenerator
from prompt2policy.curriculum.specs import TaskSpec
from prompt2policy.llm.interfaces import LLMProvider
from prompt2policy.llm.parser import TaskParseError, parse_task_spec


LOGGER = logging.getLogger("prompt2policy.curriculum_generator")


class LLMCurriculumGenerator(CurriculumGenerator):
    def __init__(self, provider: LLMProvider, fallback_difficulty_increment: float = 0.1):
        self.provider = provider
        self.fallback_difficulty_increment = fallback_difficulty_increment

    def generate_next_stage(self, previous_stage: TaskSpec, metrics: dict[str, float]) -> TaskSpec:
        LOGGER.info(
            "Generating next stage from task=%s with metrics=%s",
            previous_stage.task_id,
            metrics,
        )
        payload = {
            "previous_task": previous_stage.model_dump(mode="json"),
            "metrics": metrics,
        }
        response = self.provider.generate(payload)

        try:
            task_spec = parse_task_spec(response)
            LOGGER.info("LLM stage generation succeeded: %s", task_spec.task_id)
            return task_spec
        except TaskParseError:
            LOGGER.warning("LLM output failed validation, using fallback task generation")
            return self._fallback_task(previous_stage)

    def _fallback_task(self, previous_stage: TaskSpec) -> TaskSpec:
        stage = deepcopy(previous_stage)

        stage.parent_task_id = previous_stage.task_id
        stage.task_id = f"{previous_stage.task_id}_fallback"
        stage.difficulty_level = previous_stage.difficulty_level + self.fallback_difficulty_increment
        stage.goal_description = f"{previous_stage.goal_description} | fallback harder variant"
        stage.success_threshold_reward = previous_stage.success_threshold_reward + 2.0

        for obj in stage.world_spec.objects:
            if obj.role == "goal":
                obj.pos = [obj.pos[0] + 0.01, obj.pos[1], obj.pos[2]]

        return stage
