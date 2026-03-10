"""Render a saved PPO policy rollout to an MP4 file."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np
from stable_baselines3 import PPO

from env import FrankaReachEnv, MAX_EPISODE_STEPS
from explore_env import FrankaExploreEnv
from history_wrapper import StateHistoryWrapper

ENV_CLASSES = {
    "reach": FrankaReachEnv,
    "explore": FrankaExploreEnv,
}


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
    # SB3 VecTransposeImage stores images as (C, H, W); detect by checking if first dim is small.
    if shape[0] < shape[1] and shape[0] < shape[2]:
        return int(shape[2]), int(shape[1])  # CHW -> width=shape[2], height=shape[1]
    return int(shape[1]), int(shape[0])  # HWC -> width=shape[1], height=shape[0]


def adapt_obs_for_model(obs: dict, model: PPO) -> dict:
    expected_spaces = model.observation_space.spaces
    if "image" not in expected_spaces:
        raise KeyError("Loaded model observation space does not contain 'image'")

    expected = tuple(expected_spaces["image"].shape)
    image = obs["image"]
    if tuple(image.shape) == expected:
        out = dict(obs)
    else:
        # Convert HWC -> CHW when the loaded policy expects channels-first.
        if image.ndim == 3 and expected[0] < expected[1]:
            if image.shape[2] == expected[0] and image.shape[0] == expected[1] and image.shape[1] == expected[2]:
                out = {**obs, "image": np.transpose(image, (2, 0, 1))}
            else:
                raise ValueError(
                    f"Observation image shape {image.shape} does not match model expectation {expected}"
                )
        else:
            raise ValueError(
                f"Observation image shape {image.shape} does not match model expectation {expected}"
            )

    # Keep only keys expected by the model (supports both legacy and history-enabled policies).
    missing = [k for k in expected_spaces.keys() if k not in out]
    if missing:
        raise KeyError(f"Observation is missing keys required by model: {missing}")
    return {k: out[k] for k in expected_spaces.keys()}


def resize_frame_nearest(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    in_h, in_w = frame.shape[:2]
    if in_w == out_w and in_h == out_h:
        return frame
    y_idx = np.clip((np.arange(out_h) * (in_h / out_h)).astype(np.int32), 0, in_h - 1)
    x_idx = np.clip((np.arange(out_w) * (in_w / out_w)).astype(np.int32), 0, in_w - 1)
    return frame[y_idx[:, None], x_idx[None, :], :]


def resize_for_model_input(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    in_h, in_w, c = frame.shape
    if in_w == out_w and in_h == out_h:
        return frame
    # Prefer cheap box downsampling when integer ratio; fallback to nearest otherwise.
    if in_h % out_h == 0 and in_w % out_w == 0 and in_h >= out_h and in_w >= out_w:
        fy = in_h // out_h
        fx = in_w // out_w
        reshaped = frame.reshape(out_h, fy, out_w, fx, c)
        return reshaped.mean(axis=(1, 3)).astype(np.uint8)
    return resize_frame_nearest(frame, out_w, out_h)


def make_three_pane_frame(
    human_frame: np.ndarray,
    front_img: np.ndarray,
    top_img: np.ndarray,
    pane_w: int,
    pane_h: int,
    model_w: int,
    model_h: int,
) -> np.ndarray:
    """Compose three panes: human view | front camera | top camera."""
    left = resize_frame_nearest(human_frame, pane_w, pane_h)
    front_native = resize_for_model_input(front_img, model_w, model_h)
    mid = resize_frame_nearest(front_native, pane_w, pane_h)
    top_native = resize_for_model_input(top_img, model_w, model_h)
    right = resize_frame_nearest(top_native, pane_w, pane_h)
    return np.concatenate([left, mid, right], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="./runs/fr3_explore_v2/best/best_model.zip",
        help="Path to saved SB3 PPO model .zip",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./runs/fr3_explore_v2/rollout.mp4",
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
    parser.add_argument(
        "--env",
        type=str,
        default="reach",
        choices=list(ENV_CLASSES.keys()),
        help="Environment type",
    )
    parser.add_argument("--history-len", type=int, default=5, help="Number of state history taps")
    parser.add_argument("--history-stride", type=int, default=2, help="Spacing between history taps in env steps")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(args.model_path)
    model = PPO.load(model_path, device=args.device)

    model_img_w, model_img_h = infer_env_image_size(model)
    env_cls = ENV_CLASSES[args.env]
    base_env = env_cls(
        render_mode="rgb_array",
        img_width=max(args.width, model_img_w),
        img_height=max(args.height, model_img_h),
    )
    env = StateHistoryWrapper(base_env, history_len=args.history_len, history_stride=args.history_stride)

    frame_count = 0
    rewards: list[float] = []
    successes = 0

    def build_frame(obs_image: np.ndarray, human_frame: np.ndarray) -> np.ndarray:
        front_img = obs_image[:, :, :3]
        top_img = obs_image[:, :, 3:]
        return make_three_pane_frame(
            human_frame=human_frame,
            front_img=front_img,
            top_img=top_img,
            pane_w=args.width,
            pane_h=args.height,
            model_w=model_img_w,
            model_h=model_img_h,
        )

    with imageio.get_writer(output_path.as_posix(), fps=args.fps, codec="libx264") as writer:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            ep_reward = 0.0
            ep_success = False
            done = False
            steps = 0

            obs_image = np.asarray(obs["image"], dtype=np.uint8)
            human_frame = env.render()
            if human_frame is None:
                human_frame = obs_image[:, :, :3]
            writer.append_data(build_frame(obs_image, human_frame))
            frame_count += 1

            while not done and steps < args.max_steps:
                obs_image = np.asarray(obs["image"], dtype=np.uint8)
                obs_for_model = {
                    **obs,
                    "image": resize_for_model_input(obs_image, model_img_w, model_img_h),
                }
                model_obs = adapt_obs_for_model(obs_for_model, model)
                action, _ = model.predict(model_obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                done = bool(terminated or truncated)
                steps += 1

                next_obs_image = np.asarray(obs["image"], dtype=np.uint8)
                human_frame = env.render()
                if human_frame is None:
                    human_frame = next_obs_image[:, :, :3]
                writer.append_data(build_frame(next_obs_image, human_frame))
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
    print(f"Video pane size (each view): {args.width}x{args.height}")
    print(f"Final video size: {args.width * 3}x{args.height} (left=human, mid=front cam, right=top cam)")
    print(f"Frames: {frame_count}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean episode reward: {mean_reward:.3f}")
    print(f"Successes (episodes): {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
