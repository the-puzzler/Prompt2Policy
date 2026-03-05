"""Training script for FR3 reach task with PPO + MultiInputPolicy."""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from env import FrankaReachEnv
from ppo_config import PPO_PARAMS


class RenderCallback(BaseCallback):
    """Calls render() on the first training env each step."""

    def _on_step(self) -> bool:
        self.training_env.envs[0].render()
        return True


def make_env(seed, render_mode=None):
    def _init():
        env = FrankaReachEnv(render_mode=render_mode)
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--save-path", type=str, default="./runs/fr3_reach")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true", help="Open MuJoCo viewer from the camera POV during training")
    args = parser.parse_args()

    # DummyVecEnv runs all envs in-process (MuJoCo's OpenGL renderer
    # doesn't support multiple processes on macOS).
    # All envs must share the same render_mode; the RenderCallback
    # only calls render() on the first one.
    rm = "human" if args.render else None
    env_fns = [make_env(args.seed + i, render_mode=rm) for i in range(args.n_envs)]
    train_envs = VecMonitor(DummyVecEnv(env_fns))

    # Eval env (VecTransposeImage to match training env wrapping)
    eval_env = VecTransposeImage(VecMonitor(DummyVecEnv([make_env(args.seed + 100)])))

    model = PPO(
        "MultiInputPolicy",
        train_envs,
        verbose=1,
        seed=args.seed,
        tensorboard_log=f"{args.save_path}/tb",
        **PPO_PARAMS,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{args.save_path}/best",
        log_path=f"{args.save_path}/eval",
        eval_freq=args.eval_freq // args.n_envs,
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
