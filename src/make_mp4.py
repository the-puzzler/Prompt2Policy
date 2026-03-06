"""Render a saved PPO policy rollout to an MP4 file."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import PPO

from env import FrankaReachEnv, MAX_EPISODE_STEPS


def resolve_model_path(raw_path: str) -> str:
    p = Path(raw_path)
    if p.exists():
        return p.as_posix()
    if p.suffix == ".zip":
        no_zip = p.with_suffix("")
        if no_zip.exists():
            return no_zip.as_posix()
    else:
        with_zip = p.with_suffix(".zip")
        if with_zip.exists():
            return with_zip.as_posix()
    raise FileNotFoundError(f"Model not found at '{raw_path}' (also tried zip/no-zip variants)")


def infer_env_image_size(model: PPO) -> tuple[int, int]:
    img_space = model.observation_space.spaces["image"]
    shape = tuple(img_space.shape)
    if len(shape) != 3:
        raise ValueError(f"Unexpected image observation shape: {shape}")
    # If channels-first, shape is (C, H, W); otherwise (H, W, C).
    if shape[0] in (1, 3, 4):
        return int(shape[2]), int(shape[1])  # width, height
    return int(shape[1]), int(shape[0])  # width, height


def adapt_obs_for_model(obs: dict, model: PPO) -> dict:
    expected = tuple(model.observation_space.spaces["image"].shape)
    image = obs["image"]
    if tuple(image.shape) == expected:
        return obs
    if len(expected) == 3 and expected[0] in (1, 3, 4) and image.ndim == 3:
        # Convert HWC -> CHW when the loaded policy expects channels-first.
        if image.shape[0] == expected[1] and image.shape[1] == expected[2] and image.shape[2] == expected[0]:
            converted = np.transpose(image, (2, 0, 1))
            return {**obs, "image": converted}
    raise ValueError(
        f"Observation image shape {image.shape} does not match model expectation {expected}"
    )


def resize_frame_nearest(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    in_h, in_w = frame.shape[:2]
    if in_w == out_w and in_h == out_h:
        return frame
    y_idx = np.clip((np.arange(out_h) * (in_h / out_h)).astype(np.int32), 0, in_h - 1)
    x_idx = np.clip((np.arange(out_w) * (in_w / out_w)).astype(np.int32), 0, in_w - 1)
    return frame[y_idx[:, None], x_idx[None, :], :]


def resize_for_model_input(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    in_h, in_w = frame.shape[:2]
    if in_w == out_w and in_h == out_h:
        return frame
    # Prefer cheap box downsampling when integer ratio; fallback to nearest otherwise.
    if in_h % out_h == 0 and in_w % out_w == 0 and in_h >= out_h and in_w >= out_w:
        fy = in_h // out_h
        fx = in_w // out_w
        reshaped = frame.reshape(out_h, fy, out_w, fx, 3)
        return reshaped.mean(axis=(1, 3)).astype(np.uint8)
    return resize_frame_nearest(frame, out_w, out_h)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="./runs/fr3_reach/best/best_model.zip",
        help="Path to saved SB3 PPO model .zip",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./runs/fr3_reach/rollout.mp4",
        help="Output MP4 path",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to render")
    parser.add_argument("--seed", type=int, default=42, help="Base seed (episode i uses seed+i)")
    parser.add_argument("--fps", type=int, default=50, help="Output video FPS")
    parser.add_argument("--width", type=int, default=640, help="Output video width")
    parser.add_argument("--height", type=int, default=640, help="Output video height")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_EPISODE_STEPS,
        help="Per-episode step cap",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device used by SB3 model for inference",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions (default is deterministic)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(args.model_path)
    model = PPO.load(model_path, device=args.device)

    model_img_w, model_img_h = infer_env_image_size(model)
    env = FrankaReachEnv(
        render_mode="rgb_array",
        img_width=max(args.width, model_img_w),
        img_height=max(args.height, model_img_h),
    )

    frame_count = 0
    rewards: list[float] = []
    successes = 0

    with imageio.get_writer(output_path.as_posix(), fps=args.fps, codec="libx264") as writer:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            ep_reward = 0.0
            ep_success = False
            done = False
            steps = 0

            frame = resize_frame_nearest(np.asarray(obs["image"], dtype=np.uint8), args.width, args.height)
            writer.append_data(frame)
            frame_count += 1

            while not done and steps < args.max_steps:
                obs_for_model = {
                    **obs,
                    "image": resize_for_model_input(
                        np.asarray(obs["image"], dtype=np.uint8), model_img_w, model_img_h
                    ),
                }
                model_obs = adapt_obs_for_model(obs_for_model, model)
                action, _ = model.predict(model_obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                done = bool(terminated or truncated)
                steps += 1

                frame = resize_frame_nearest(np.asarray(obs["image"], dtype=np.uint8), args.width, args.height)
                writer.append_data(frame)
                frame_count += 1

                if bool(info.get("is_success", False)):
                    ep_success = True

            rewards.append(ep_reward)
            if ep_success:
                successes += 1
            print(f"Episode {ep + 1}/{args.episodes}: steps={steps} reward={ep_reward:.3f}")

    env.close()

    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    print(f"\nSaved video: {output_path}")
    print(f"Loaded model: {model_path}")
    print(f"Policy obs image size: {model_img_w}x{model_img_h}")
    print(f"Video size: {args.width}x{args.height}")
    print(f"Frames: {frame_count}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean episode reward: {mean_reward:.3f}")
    print(f"Successes (episodes): {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
