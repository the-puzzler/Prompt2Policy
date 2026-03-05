from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from prompt2policy.algorithms.interface import AlgorithmConfig
from prompt2policy.algorithms.registry import create_algorithm
from prompt2policy.config.loader import (
    load_experiment_config,
    load_robot_spec,
    load_task_spec,
    save_task_spec,
)
from prompt2policy.curriculum.generator import LLMCurriculumGenerator
from prompt2policy.curriculum.promotion import PromotionRule
from prompt2policy.envs.factory import create_env
from prompt2policy.llm.mock_provider import MockLLMProvider
from prompt2policy.rewards.model import compile_reward_model
from prompt2policy.world.builder import WorldBuilder


LOGGER = logging.getLogger("prompt2policy.curriculum_runner")


class CurriculumRunner:
    def __init__(self, workspace_root: Path, experiment_config_path: Path):
        self.workspace_root = workspace_root
        self.experiment_config_path = experiment_config_path

    def run(self) -> dict[str, Any]:
        LOGGER.info("Loading experiment config from %s", self.experiment_config_path)
        experiment = load_experiment_config(self.experiment_config_path)
        robot_spec = load_robot_spec(self._resolve(experiment.robot_config_path))
        current_task = load_task_spec(self._resolve(experiment.seed_task_path))

        output_dir = self._resolve(experiment.runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "Run '%s' started | backend=%s | algorithm=%s | max_stages=%d | output_dir=%s",
            experiment.name,
            experiment.runtime.backend,
            experiment.training.algorithm,
            experiment.curriculum.max_stages,
            output_dir,
        )
        LOGGER.info(
            "Seed task loaded | task_id=%s | difficulty=%.3f | success_threshold_reward=%.3f",
            current_task.task_id,
            current_task.difficulty_level,
            current_task.success_threshold_reward,
        )

        world_builder = WorldBuilder(workspace_root=self.workspace_root)
        promotion_rule = PromotionRule(window_size=experiment.curriculum.promotion_window_size)
        llm_provider = self._build_llm_provider(experiment)
        curriculum_generator = LLMCurriculumGenerator(
            provider=llm_provider,
            fallback_difficulty_increment=experiment.curriculum.fallback_difficulty_increment,
        )

        all_metrics: list[dict[str, float]] = []
        generated_tasks_dir = output_dir / "generated_tasks"
        generated_tasks_dir.mkdir(exist_ok=True)

        for stage_idx in range(experiment.curriculum.max_stages):
            stage_id = f"stage_{stage_idx:03d}"
            stage_dir = output_dir / stage_id
            stage_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(
                "=== Stage %s/%s | stage_id=%s | task_id=%s ===",
                stage_idx + 1,
                experiment.curriculum.max_stages,
                stage_id,
                current_task.task_id,
            )

            scene_path = world_builder.compose_scene(
                world_spec=current_task.world_spec,
                output_path=stage_dir / "scene.xml",
                default_camera=robot_spec.camera_defaults[0],
            )
            LOGGER.info("Scene composed at %s", scene_path)

            reward_model = compile_reward_model(current_task.reward_spec)
            LOGGER.info("Reward model compiled with %d terms", len(current_task.reward_spec.terms))

            LOGGER.info("Creating training environment (backend=%s)", experiment.runtime.backend)
            env = create_env(
                workspace_root=self.workspace_root,
                backend_name=experiment.runtime.backend,
                scene_path=scene_path,
                robot_spec=robot_spec,
                task_spec=current_task,
                reward_model=reward_model,
                image_width=experiment.runtime.image_width,
                image_height=experiment.runtime.image_height,
                camera_name=experiment.runtime.camera_name,
                control_timestep=experiment.runtime.control_timestep,
                physics_timestep=experiment.runtime.physics_timestep,
                model_id=f"{current_task.task_id}_{stage_id}",
            )

            LOGGER.info("Creating evaluation environment (backend=%s)", experiment.runtime.backend)
            eval_env = create_env(
                workspace_root=self.workspace_root,
                backend_name=experiment.runtime.backend,
                scene_path=scene_path,
                robot_spec=robot_spec,
                task_spec=current_task,
                reward_model=reward_model,
                image_width=experiment.runtime.image_width,
                image_height=experiment.runtime.image_height,
                camera_name=experiment.runtime.camera_name,
                control_timestep=experiment.runtime.control_timestep,
                physics_timestep=experiment.runtime.physics_timestep,
                model_id=f"{current_task.task_id}_{stage_id}_eval",
            )

            algorithm = create_algorithm(experiment.training.algorithm)
            algo_config = AlgorithmConfig(
                total_timesteps=experiment.training.total_timesteps,
                learning_rate=experiment.training.learning_rate,
                n_steps=experiment.training.n_steps,
                batch_size=experiment.training.batch_size,
                gamma=experiment.training.gamma,
                seed=experiment.seed,
                verbose=1 if LOGGER.isEnabledFor(logging.INFO) else 0,
            )
            LOGGER.info(
                "Training started | total_timesteps=%d | learning_rate=%g | n_steps=%d | batch_size=%d",
                algo_config.total_timesteps,
                algo_config.learning_rate,
                algo_config.n_steps,
                algo_config.batch_size,
            )
            train_metrics = algorithm.train(env=env, eval_env=eval_env, config=algo_config)
            LOGGER.info("Training completed | train_metrics=%s", train_metrics)

            LOGGER.info("Evaluation started | episodes=%d", current_task.eval_episodes)
            eval_metrics = algorithm.evaluate(eval_env, episodes=current_task.eval_episodes)
            LOGGER.info("Evaluation completed | eval_metrics=%s", eval_metrics)

            metrics = {
                **train_metrics,
                **eval_metrics,
            }
            all_metrics.append(metrics)

            self._save_json(stage_dir / "metrics.json", metrics)
            self._save_json(stage_dir / "task.json", current_task.model_dump(mode="json"))
            save_task_spec(stage_dir / "task.yaml", current_task)
            LOGGER.info("Artifacts saved in %s", stage_dir)

            checkpoint_path = stage_dir / "policy_checkpoint"
            algorithm.save(str(checkpoint_path))
            LOGGER.info("Checkpoint saved at %s", checkpoint_path)

            env.close()
            eval_env.close()
            LOGGER.info("Stage environments closed")

            if stage_idx >= experiment.curriculum.max_stages - 1:
                LOGGER.info("Reached configured max stages (%d), stopping", experiment.curriculum.max_stages)
                break

            should_promote = promotion_rule.should_promote(
                all_metrics, current_task.success_threshold_reward
            )
            LOGGER.info(
                "Promotion check | should_promote=%s | threshold_reward=%.3f",
                should_promote,
                current_task.success_threshold_reward,
            )
            if not should_promote:
                LOGGER.info("Promotion criteria not met, stopping curriculum")
                break

            next_task = curriculum_generator.generate_next_stage(current_task, metrics)
            save_task_spec(generated_tasks_dir / f"{next_task.task_id}.yaml", next_task)
            LOGGER.info(
                "Next stage generated | previous_task=%s | next_task=%s | difficulty=%.3f",
                current_task.task_id,
                next_task.task_id,
                next_task.difficulty_level,
            )
            current_task = next_task

        summary = {
            "run_name": experiment.name,
            "stages_completed": len(all_metrics),
            "metrics": all_metrics,
            "output_dir": str(output_dir),
        }
        self._save_json(output_dir / "run_summary.json", summary)
        LOGGER.info("Run completed | stages_completed=%d | summary=%s", len(all_metrics), output_dir / "run_summary.json")
        return summary

    def _resolve(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return (self.workspace_root / path).resolve()

    def _save_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)

    def _build_llm_provider(self, experiment):
        provider_name = experiment.curriculum.llm_provider.lower()
        if provider_name == "mock":
            LOGGER.info(
                "Using LLM provider 'mock' (force_invalid_output=%s)",
                experiment.curriculum.force_invalid_llm_output,
            )
            return MockLLMProvider(
                force_invalid_output=experiment.curriculum.force_invalid_llm_output,
            )

        raise ValueError(f"Unsupported llm provider '{experiment.curriculum.llm_provider}'")
