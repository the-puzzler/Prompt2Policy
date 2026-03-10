"""Training script for FR3 tasks with PPO and filesystem run logging."""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecTransposeImage

from env import FrankaReachEnv
from explore_env import FrankaExploreEnv
from history_wrapper import StateHistoryWrapper
from ppo_config import PPO_PARAMS

ENV_CLASSES = {
    "reach": FrankaReachEnv,
    "explore": FrankaExploreEnv,
}


class RenderCallback(BaseCallback):
    """Calls render() on the first training env each step."""

    def _on_step(self) -> bool:
        self.training_env.envs[0].render()
        return True


class RunStatsCallback(BaseCallback):
    """Writes rollout stats to CSV and saves periodic summary plots."""

    def __init__(self, run_dir: Path, eval_callback: EvalCallback, plot_every: int = 5):
        super().__init__()
        self.run_dir = run_dir
        self.eval_callback = eval_callback
        self.plot_every = max(int(plot_every), 1)
        self.iteration = 0

        self.csv_path = self.run_dir / "stats.csv"
        self.plot_path = self.run_dir / "training_plots.png"
        self._train_metric_keys = [
            "train/approx_kl",
            "train/clip_fraction",
            "train/clip_range",
            "train/entropy_loss",
            "train/explained_variance",
            "train/learning_rate",
            "train/loss",
            "train/n_updates",
            "train/policy_gradient_loss",
            "train/std",
            "train/value_loss",
        ]

        self.steps_history: list[int] = []
        self.train_reward_history: list[float] = []
        self.eval_reward_history: list[float] = []
        self.train_success_history: list[float] = []
        self.eval_success_history: list[float] = []

        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "iteration",
                    "timesteps",
                    "train_avg_reward",
                    "eval_avg_reward",
                    "train_success_rate",
                    "eval_success_rate",
                    "rollout_ep_len_mean",
                    "rollout_ep_rew_mean",
                    "rollout_success_rate",
                    "time_fps",
                    "time_iterations",
                    "time_time_elapsed",
                    "time_total_timesteps",
                    "train_approx_kl",
                    "train_clip_fraction",
                    "train_clip_range",
                    "train_entropy_loss",
                    "train_explained_variance",
                    "train_learning_rate",
                    "train_loss",
                    "train_n_updates",
                    "train_policy_gradient_loss",
                    "train_std",
                    "train_value_loss",
                ]
            )

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.iteration += 1
        timesteps = int(self.model.num_timesteps)

        train_avg_reward = self._get_train_avg_reward()
        train_success_rate = self._get_train_success_rate()
        eval_avg_reward, eval_success_rate = self._get_eval_stats()
        sb3_metrics = self._get_sb3_console_metrics()

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    self.iteration,
                    timesteps,
                    self._fmt(train_avg_reward),
                    self._fmt(eval_avg_reward),
                    self._fmt(train_success_rate),
                    self._fmt(eval_success_rate),
                    self._fmt(sb3_metrics["rollout/ep_len_mean"]),
                    self._fmt(sb3_metrics["rollout/ep_rew_mean"]),
                    self._fmt(sb3_metrics["rollout/success_rate"]),
                    self._fmt(sb3_metrics["time/fps"]),
                    self._fmt(sb3_metrics["time/iterations"]),
                    self._fmt(sb3_metrics["time/time_elapsed"]),
                    self._fmt(sb3_metrics["time/total_timesteps"]),
                    self._fmt(sb3_metrics["train/approx_kl"]),
                    self._fmt(sb3_metrics["train/clip_fraction"]),
                    self._fmt(sb3_metrics["train/clip_range"]),
                    self._fmt(sb3_metrics["train/entropy_loss"]),
                    self._fmt(sb3_metrics["train/explained_variance"]),
                    self._fmt(sb3_metrics["train/learning_rate"]),
                    self._fmt(sb3_metrics["train/loss"]),
                    self._fmt(sb3_metrics["train/n_updates"]),
                    self._fmt(sb3_metrics["train/policy_gradient_loss"]),
                    self._fmt(sb3_metrics["train/std"]),
                    self._fmt(sb3_metrics["train/value_loss"]),
                ]
            )

        self.steps_history.append(timesteps)
        self.train_reward_history.append(train_avg_reward)
        self.eval_reward_history.append(eval_avg_reward)
        self.train_success_history.append(train_success_rate)
        self.eval_success_history.append(eval_success_rate)

        if self.iteration % self.plot_every == 0:
            self._save_plot()

    def _on_training_end(self) -> None:
        self._save_plot()

    def _get_train_avg_reward(self) -> float:
        if not getattr(self.model, "ep_info_buffer", None):
            return np.nan
        rewards = [float(ep_info["r"]) for ep_info in self.model.ep_info_buffer if "r" in ep_info]
        if not rewards:
            return np.nan
        return float(np.mean(rewards))

    def _get_train_success_rate(self) -> float:
        if not getattr(self.model, "ep_info_buffer", None):
            return np.nan
        successes = [
            float(ep_info["is_success"])
            for ep_info in self.model.ep_info_buffer
            if "is_success" in ep_info
        ]
        if not successes:
            return np.nan
        return float(np.mean(successes))

    def _get_eval_stats(self) -> tuple[float, float]:
        raw_eval_reward = float(getattr(self.eval_callback, "last_mean_reward", np.nan))
        eval_avg_reward = raw_eval_reward if np.isfinite(raw_eval_reward) else np.nan
        eval_success_rate = np.nan

        eval_successes = getattr(self.eval_callback, "evaluations_successes", None)
        if eval_successes is not None and len(eval_successes) > 0:
            latest_eval_successes = eval_successes[-1]
            if len(latest_eval_successes) > 0:
                eval_success_rate = float(np.mean(latest_eval_successes))

        return eval_avg_reward, eval_success_rate

    def _save_plot(self) -> None:
        if not self.steps_history:
            return

        fig, (ax_reward, ax_success) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax_reward.plot(self.steps_history, self.train_reward_history, label="train_avg_reward", linewidth=1.8)
        ax_reward.plot(self.steps_history, self.eval_reward_history, label="eval_avg_reward", linewidth=1.8)
        ax_reward.set_ylabel("Average Reward")
        ax_reward.grid(True, alpha=0.3)
        ax_reward.legend()

        ax_success.plot(self.steps_history, self.train_success_history, label="train_success_rate", linewidth=1.8)
        ax_success.plot(self.steps_history, self.eval_success_history, label="eval_success_rate", linewidth=1.8)
        ax_success.set_xlabel("Timesteps")
        ax_success.set_ylabel("Success Rate")
        ax_success.set_ylim(-0.05, 1.05)
        ax_success.grid(True, alpha=0.3)
        ax_success.legend()

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=140)
        plt.close(fig)

    @staticmethod
    def _fmt(value: float) -> str:
        if np.isnan(value):
            return "nan"
        return f"{value:.6f}"

    def _get_sb3_console_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}

        ep_infos = list(getattr(self.model, "ep_info_buffer", []))
        ep_success = list(getattr(self.model, "ep_success_buffer", []))
        metrics["rollout/ep_len_mean"] = self._safe_mean([float(ep_info["l"]) for ep_info in ep_infos if "l" in ep_info])
        metrics["rollout/ep_rew_mean"] = self._safe_mean([float(ep_info["r"]) for ep_info in ep_infos if "r" in ep_info])
        metrics["rollout/success_rate"] = self._safe_mean([float(v) for v in ep_success])

        time_elapsed = max((time.time_ns() - self.model.start_time) / 1e9, np.finfo(float).eps)
        metrics["time/fps"] = float(int((self.model.num_timesteps - self.model._num_timesteps_at_start) / time_elapsed))
        metrics["time/iterations"] = float(self.iteration)
        metrics["time/time_elapsed"] = float(int(time_elapsed))
        metrics["time/total_timesteps"] = float(self.model.num_timesteps)

        logger_values = getattr(self.model.logger, "name_to_value", {})
        for key in self._train_metric_keys:
            value = logger_values.get(key, np.nan)
            metrics[key] = float(value) if isinstance(value, (int, float, np.floating)) else np.nan

        return metrics

    @staticmethod
    def _safe_mean(values: list[float]) -> float:
        if not values:
            return np.nan
        return float(np.mean(values))


def make_env(seed, render_mode=None, env_cls=FrankaReachEnv, history_len: int = 5, history_stride: int = 2):
    def _init():
        env = env_cls(render_mode=render_mode)
        env = StateHistoryWrapper(env, history_len=history_len, history_stride=history_stride)
        env.reset(seed=seed)
        return env

    return _init


def _create_run_dir(runs_dir: Path, env_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"{env_name}_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = runs_dir / f"{env_name}_{timestamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_config(
    run_dir: Path,
    args: argparse.Namespace,
    resolved_device: str,
    vec_env_type: str,
    model_stats: dict | None = None,
) -> None:
    def _jsonable(value):
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, type):
            return value.__name__
        if callable(value):
            return getattr(value, "__name__", repr(value))
        try:
            json.dumps(value)
            return value
        except TypeError:
            return repr(value)

    config = {
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "env": args.env,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "eval_freq": args.eval_freq,
        "seed": args.seed,
        "device_arg": args.device,
        "resolved_device": resolved_device,
        "vec_env": vec_env_type,
        "render": args.render,
        "pretrained_model": args.pretrained_model,
        "plot_every": args.plot_every,
        "history_len": args.history_len,
        "history_stride": args.history_stride,
        "ppo_params": _jsonable(PPO_PARAMS),
    }
    if model_stats is not None:
        config["model"] = _jsonable(model_stats)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--runs-dir", type=str, default="./runs")
    parser.add_argument("--plot-every", type=int, default=5, help="Save plots every N rollout iterations")
    parser.add_argument("--history-len", type=int, default=5, help="Number of state history taps")
    parser.add_argument("--history-stride", type=int, default=2, help="Spacing between history taps in env steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--vec-env", type=str, default="subproc", choices=["subproc", "dummy"])
    parser.add_argument("--render", action="store_true", help="Open MuJoCo viewer from the camera POV during training")
    parser.add_argument("--env", type=str, default="reach", choices=list(ENV_CLASSES.keys()), help="Environment to train on")
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="Path to a pretrained model zip to fine-tune (e.g. runs/explore_YYYYMMDD_HHMMSS/best/best_model.zip)",
    )
    args = parser.parse_args()

    env_cls = ENV_CLASSES[args.env]

    runs_dir = Path(args.runs_dir)
    run_dir = _create_run_dir(runs_dir, args.env)

    cuda_available = torch.cuda.is_available()
    if args.device == "auto":
        device = "cuda" if cuda_available else "cpu"
    else:
        device = args.device
    if device == "cuda" and not cuda_available:
        print("CUDA requested but not available, falling back to CPU.")
        device = "cpu"

    print(
        f"Torch={torch.__version__} CUDA available={cuda_available} "
        f"CUDA devices={torch.cuda.device_count() if cuda_available else 0} Selected device={device}"
    )
    if device == "cuda":
        print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")

    rm = "human" if args.render else None
    env_fns = [
        make_env(
            args.seed + i,
            render_mode=rm,
            env_cls=env_cls,
            history_len=args.history_len,
            history_stride=args.history_stride,
        )
        for i in range(args.n_envs)
    ]
    vec_env_type = args.vec_env
    if args.render and vec_env_type == "subproc":
        print("Render mode requires in-process envs. Switching vec env from subproc to dummy.")
        vec_env_type = "dummy"
    if vec_env_type == "subproc":
        # Real CPU parallelism across env workers.
        train_envs = VecMonitor(SubprocVecEnv(env_fns, start_method="spawn"), info_keywords=("is_success",))
    else:
        train_envs = VecMonitor(DummyVecEnv(env_fns), info_keywords=("is_success",))
    print(f"Vectorized env type: {vec_env_type} (n_envs={args.n_envs})")

    # Eval env (VecTransposeImage to match training env wrapping)
    eval_env = VecTransposeImage(
        VecMonitor(
            DummyVecEnv(
                [
                    make_env(
                        args.seed + 100,
                        env_cls=env_cls,
                        history_len=args.history_len,
                        history_stride=args.history_stride,
                    )
                ]
            ),
            info_keywords=("is_success",),
        )
    )

    if args.pretrained_model:
        print(f"Loading pretrained model from: {args.pretrained_model}")
        model = PPO.load(
            args.pretrained_model,
            env=train_envs,
            device=device,
        )
    else:
        model = PPO(
            "MultiInputPolicy",
            train_envs,
            verbose=1,
            seed=args.seed,
            device=device,
            **PPO_PARAMS,
        )

    model_stats = {
        "total_params": int(sum(p.numel() for p in model.policy.parameters())),
        "trainable_params": int(sum(p.numel() for p in model.policy.parameters() if p.requires_grad)),
    }
    _write_config(run_dir, args, device, vec_env_type, model_stats=model_stats)
    print(
        f"Model params: total={model_stats['total_params']:,} "
        f"trainable={model_stats['trainable_params']:,}"
    )
    print(f"SB3 model device: {model.device}")
    print("\n=== Policy Network Architecture ===")
    print(model.policy)
    print("===================================\n")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "eval"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    callbacks = [eval_callback, RunStatsCallback(run_dir=run_dir, eval_callback=eval_callback, plot_every=args.plot_every)]
    if args.render:
        callbacks.append(RenderCallback())

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    model.save(str(run_dir / "final_model"))

    train_envs.close()
    eval_env.close()
    print(f"Training complete. Outputs saved to {run_dir}/")


if __name__ == "__main__":
    main()
