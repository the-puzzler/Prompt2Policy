"""Profiling script for PPO training pipeline throughput and parameter counts.

Usage:
    cd src
    python time_trial.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from types import MethodType

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecTransposeImage

from env import FrankaReachEnv
from ppo_config import PPO_PARAMS


def make_env(seed: int, render_mode: str | None = None):
    def _init():
        env = FrankaReachEnv(render_mode=render_mode)
        env.reset(seed=seed)
        return env

    return _init


@dataclass
class PipelineStats:
    rollout_times_s: list[float] = field(default_factory=list)
    rollout_steps: list[int] = field(default_factory=list)
    train_times_s: list[float] = field(default_factory=list)
    eval_times_s: list[float] = field(default_factory=list)
    eval_calls: int = 0

    def record_rollout(self, duration_s: float, steps: int) -> None:
        self.rollout_times_s.append(duration_s)
        self.rollout_steps.append(steps)

    def record_train(self, duration_s: float) -> None:
        self.train_times_s.append(duration_s)

    def record_eval(self, duration_s: float) -> None:
        self.eval_times_s.append(duration_s)
        self.eval_calls += 1


class TimedEvalCallback(EvalCallback):
    """EvalCallback that records wall time spent inside triggered eval runs."""

    def __init__(self, stats: PipelineStats, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stats = stats

    def _on_step(self) -> bool:
        will_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        t0 = time.perf_counter()
        keep_going = super()._on_step()
        dt = time.perf_counter() - t0
        if will_eval:
            self._stats.record_eval(dt)
        return keep_going


def _param_id_set(module) -> set[int]:
    return {id(p) for p in module.parameters() if p.requires_grad}


def summarize_params(model: PPO) -> dict[str, int]:
    policy = model.policy

    total = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    shared_ids = _param_id_set(policy.features_extractor)
    shared_count = sum(p.numel() for p in policy.features_extractor.parameters() if p.requires_grad)

    actor_ids = _param_id_set(policy.mlp_extractor.policy_net) | _param_id_set(policy.action_net)
    if hasattr(policy, "log_std") and isinstance(policy.log_std, torch.nn.Parameter):
        actor_ids.add(id(policy.log_std))

    critic_ids = _param_id_set(policy.mlp_extractor.value_net) | _param_id_set(policy.value_net)

    id_to_param = {id(p): p for p in policy.parameters() if p.requires_grad}
    actor_exclusive_ids = actor_ids - critic_ids - shared_ids
    critic_exclusive_ids = critic_ids - actor_ids - shared_ids

    actor_exclusive = sum(id_to_param[pid].numel() for pid in actor_exclusive_ids)
    critic_exclusive = sum(id_to_param[pid].numel() for pid in critic_exclusive_ids)

    return {
        "total_trainable": total,
        "shared_extractor": shared_count,
        "actor_exclusive": actor_exclusive,
        "critic_exclusive": critic_exclusive,
        "actor_with_shared": actor_exclusive + shared_count,
        "critic_with_shared": critic_exclusive + shared_count,
    }


def install_timers(model: PPO, stats: PipelineStats) -> None:
    orig_collect = model.collect_rollouts
    orig_train = model.train

    def timed_collect(self, env, callback, rollout_buffer, n_rollout_steps):
        t0 = time.perf_counter()
        result = orig_collect(env, callback, rollout_buffer, n_rollout_steps)
        dt = time.perf_counter() - t0
        stats.record_rollout(dt, int(n_rollout_steps * env.num_envs))
        return result

    def timed_train(self):
        t0 = time.perf_counter()
        result = orig_train()
        dt = time.perf_counter() - t0
        stats.record_train(dt)
        return result

    model.collect_rollouts = MethodType(timed_collect, model)
    model.train = MethodType(timed_train, model)


def fmt_ratio(part_s: float, total_s: float) -> str:
    if total_s <= 0:
        return "0.0%"
    return f"{100.0 * part_s / total_s:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--vec-env", type=str, default="subproc", choices=["subproc", "dummy"])
    parser.add_argument("--save-path", type=str, default="./runs/fr3_reach_profile")
    args = parser.parse_args()

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

    env_fns = [make_env(args.seed + i, render_mode=None) for i in range(args.n_envs)]
    if args.vec_env == "subproc":
        train_envs = VecMonitor(SubprocVecEnv(env_fns, start_method="spawn"))
    else:
        train_envs = VecMonitor(DummyVecEnv(env_fns))

    eval_env = VecTransposeImage(VecMonitor(DummyVecEnv([make_env(args.seed + 100)])))

    stats = PipelineStats()
    model = PPO(
        "MultiInputPolicy",
        train_envs,
        verbose=1,
        seed=args.seed,
        device=device,
        tensorboard_log=f"{args.save_path}/tb",
        **PPO_PARAMS,
    )
    install_timers(model, stats)

    eval_callback = TimedEvalCallback(
        stats,
        eval_env,
        best_model_save_path=f"{args.save_path}/best",
        log_path=f"{args.save_path}/eval",
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )

    params = summarize_params(model)
    print("\n=== Parameter Counts (Trainable) ===")
    print(f"total_trainable:      {params['total_trainable']:,}")
    print(f"shared_extractor:     {params['shared_extractor']:,}")
    print(f"actor_exclusive:      {params['actor_exclusive']:,}")
    print(f"critic_exclusive:     {params['critic_exclusive']:,}")
    print(f"actor_with_shared:    {params['actor_with_shared']:,}")
    print(f"critic_with_shared:   {params['critic_with_shared']:,}")

    wall_t0 = time.perf_counter()
    model.learn(total_timesteps=args.total_timesteps, callback=[eval_callback])
    wall_s = time.perf_counter() - wall_t0

    rollout_s = sum(stats.rollout_times_s)
    train_s = sum(stats.train_times_s)
    eval_s = sum(stats.eval_times_s)
    other_s = max(wall_s - rollout_s - train_s - eval_s, 0.0)
    total_steps = sum(stats.rollout_steps)

    print("\n=== Pipeline Timing Summary ===")
    print(f"total_wall_s:         {wall_s:.2f}")
    print(f"rollout_s:            {rollout_s:.2f} ({fmt_ratio(rollout_s, wall_s)})")
    print(f"train_update_s:       {train_s:.2f} ({fmt_ratio(train_s, wall_s)})")
    print(f"eval_s:               {eval_s:.2f} ({fmt_ratio(eval_s, wall_s)}) from {stats.eval_calls} eval calls")
    print(f"other_overhead_s:     {other_s:.2f} ({fmt_ratio(other_s, wall_s)})")
    print(f"profiled_env_steps:   {total_steps}")
    if rollout_s > 0 and total_steps > 0:
        print(f"rollout_steps_per_s:  {total_steps / rollout_s:.1f}")
    if wall_s > 0 and total_steps > 0:
        print(f"end_to_end_steps_per_s: {total_steps / wall_s:.1f}")

    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()
