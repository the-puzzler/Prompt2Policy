from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from prompt2policy.training.curriculum_runner import CurriculumRunner
from prompt2policy.training.evaluator import evaluate_checkpoint
from prompt2policy.training.stage_generation import generate_next_stage_task


def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v: debug)",
    )


def _configure_logging(log_level: str, verbose_count: int) -> None:
    resolved_level = logging.DEBUG if verbose_count > 0 else getattr(logging, log_level.upper())
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt2Policy CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run curriculum training")
    train_parser.add_argument(
        "--experiment",
        type=str,
        default="configs/experiment.yaml",
        help="Path to experiment YAML",
    )
    _add_logging_args(train_parser)

    gen_parser = subparsers.add_parser("generate-stage", help="Generate next task stage")
    gen_parser.add_argument("--previous-task", type=str, required=True)
    gen_parser.add_argument("--output", type=str, required=True)
    gen_parser.add_argument("--mean-reward", type=float, required=True)
    gen_parser.add_argument("--force-invalid-output", action="store_true")
    _add_logging_args(gen_parser)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate saved checkpoint")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--robot-config", type=str, default="configs/robot_fr3.yaml")
    eval_parser.add_argument("--task", type=str, default="tasks/seed/fr3_reaching.yaml")
    eval_parser.add_argument("--backend", type=str, default="native")
    eval_parser.add_argument("--episodes", type=int, default=5)
    _add_logging_args(eval_parser)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(log_level=args.log_level, verbose_count=args.verbose)

    workspace_root = Path.cwd()

    if args.command == "train":
        runner = CurriculumRunner(
            workspace_root=workspace_root,
            experiment_config_path=(workspace_root / args.experiment).resolve(),
        )
        summary = runner.run()
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "generate-stage":
        generate_next_stage_task(
            previous_task_path=(workspace_root / args.previous_task).resolve(),
            output_path=(workspace_root / args.output).resolve(),
            mean_reward=float(args.mean_reward),
            force_invalid_output=bool(args.force_invalid_output),
        )
        print(json.dumps({"status": "ok", "output": str((workspace_root / args.output).resolve())}))
        return 0

    if args.command == "evaluate":
        metrics = evaluate_checkpoint(
            workspace_root=workspace_root,
            checkpoint_path=(workspace_root / args.checkpoint).resolve(),
            robot_config_path=(workspace_root / args.robot_config).resolve(),
            task_path=(workspace_root / args.task).resolve(),
            backend_name=args.backend,
            episodes=int(args.episodes),
        )
        print(json.dumps(metrics, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
