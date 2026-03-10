"""Training script for FR3 reach task with PPO + MultiInputPolicy."""

import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from env import FrankaReachEnv
from explore_env import FrankaExploreEnv
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


def make_env(seed, render_mode=None, env_cls=FrankaReachEnv):
    def _init():
        env = env_cls(render_mode=render_mode)
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--save-path", type=str, default="./runs/fr3_reach")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--vec-env", type=str, default="subproc", choices=["subproc", "dummy"])
    parser.add_argument("--render", action="store_true", help="Open MuJoCo viewer from the camera POV during training")
    parser.add_argument("--env", type=str, default="reach", choices=list(ENV_CLASSES.keys()), help="Environment to train on")
    parser.add_argument("--pretrained-model", type=str, default=None, help="Path to a pretrained model zip to fine-tune (e.g. runs/fr3_explore/best/best_model)")
    args = parser.parse_args()

    env_cls = ENV_CLASSES[args.env]

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
    env_fns = [make_env(args.seed + i, render_mode=rm, env_cls=env_cls) for i in range(args.n_envs)]
    vec_env_type = args.vec_env
    if args.render and vec_env_type == "subproc":
        print("Render mode requires in-process envs. Switching vec env from subproc to dummy.")
        vec_env_type = "dummy"
    if vec_env_type == "subproc":
        # Real CPU parallelism across env workers.
        train_envs = VecMonitor(SubprocVecEnv(env_fns, start_method="spawn"))
    else:
        train_envs = VecMonitor(DummyVecEnv(env_fns))
    print(f"Vectorized env type: {vec_env_type} (n_envs={args.n_envs})")

    # Eval env (VecTransposeImage to match training env wrapping)
    eval_env = VecTransposeImage(VecMonitor(DummyVecEnv([make_env(args.seed + 100, env_cls=env_cls)])))

    if args.pretrained_model:
        print(f"Loading pretrained model from: {args.pretrained_model}")
        model = PPO.load(
            args.pretrained_model,
            env=train_envs,
            device=device,
            tensorboard_log=f"{args.save_path}/tb",
        )
    else:
        model = PPO(
            "MultiInputPolicy",
            train_envs,
            verbose=1,
            seed=args.seed,
            device=device,
            tensorboard_log=f"{args.save_path}/tb",
            **PPO_PARAMS,
        )
    print(f"SB3 model device: {model.device}")
    print("\n=== Policy Network Architecture ===")
    print(model.policy)
    print("===================================\n")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{args.save_path}/best",
        log_path=f"{args.save_path}/eval",
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    callbacks = [eval_callback]
    if args.render:
        callbacks.append(RenderCallback())

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    model.save(f"{args.save_path}/final_model")

    train_envs.close()
    eval_env.close()
    print(f"Training complete. Model saved to {args.save_path}/")


if __name__ == "__main__":
    main()
