"""Play a trained PPO policy in the FR3 environments."""

import argparse
import os

from stable_baselines3 import PPO

from env import FrankaReachEnv
from env_ee_direction import FrankaReachEEDirectionEnv


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_EE = os.path.join(PROJECT_ROOT, "src", "runs", "fr3_reach_ee_direction", "final_model.zip")
DEFAULT_MODEL_JOINT = os.path.join(PROJECT_ROOT, "src", "runs", "fr3_reach", "final_model.zip")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained FR3 PPO policy.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_EE)
    parser.add_argument("--env-type", type=str, default="auto", choices=["auto", "joint", "ee-direction"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions instead of deterministic.")
    parser.add_argument("--render-mode", type=str, default="human", choices=["human", "rgb_array", "none"])
    return parser.parse_args()


def _resolve_env_type(model_path: str, env_type: str) -> str:
    if env_type != "auto":
        return env_type
    low = model_path.lower()
    if "ee_direction" in low:
        return "ee-direction"
    return "joint"


def _make_env(env_type: str, render_mode: str):
    rm = None if render_mode == "none" else render_mode
    if env_type == "ee-direction":
        return FrankaReachEEDirectionEnv(render_mode=rm)
    return FrankaReachEnv(render_mode=rm)


def main() -> None:
    args = _parse_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1.")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1.")

    env_type = _resolve_env_type(args.model_path, args.env_type)
    env = _make_env(env_type, args.render_mode)

    model = PPO.load(args.model_path)
    print(f"Loaded model: {args.model_path}")
    print(f"Environment type: {env_type}")
    print(f"Render mode: {args.render_mode}")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_return = 0.0
        ep_len = 0
        success = False

        while not done and ep_len < args.max_steps:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            done = bool(terminated or truncated)
            success = bool(info.get("is_success", False))

            if args.render_mode == "human":
                env.render()

        print(
            f"Episode {ep + 1}/{args.episodes} "
            f"return={ep_return:.3f} len={ep_len} success={success}"
        )

    env.close()


if __name__ == "__main__":
    main()
