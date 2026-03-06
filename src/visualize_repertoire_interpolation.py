"""Animate joint-space interpolation between two MAP-Elites repertoire elites.

The animation has three synchronized panels:
1) Robot camera view (MuJoCo offscreen render)
2) Top-down repertoire map with desired straight line and measured EE path
3) Joint trajectories q1..q7 with current interpolation cursor
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import mujoco
import numpy as np

# Force non-interactive backend for headless machines.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REPERTOIRE = os.path.join(PROJECT_ROOT, "src", "runs", "fr3_map_elites", "repertoire.csv")
DEFAULT_SCENE_XML = os.path.join(PROJECT_ROOT, "mujoco_menagerie", "franka_fr3", "reach_scene.xml")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "src", "runs", "fr3_map_elites", "interpolation_demo.mp4")
FR3_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
EE_SITE_NAME = "attachment_site"
CAMERA_NAME = "front_camera"


@dataclass
class EliteRow:
    row_index: int
    x_bin: int
    y_bin: int
    z_bin: int
    x: float
    y: float
    z: float
    q: np.ndarray
    score: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize interpolation between two repertoire elites.")
    parser.add_argument("--repertoire-csv", type=str, default=DEFAULT_REPERTOIRE)
    parser.add_argument("--scene-xml", type=str, default=DEFAULT_SCENE_XML)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT, help="Output file (.mp4 or .gif).")
    parser.add_argument("--start-cell", type=str, default="", help='Cell as "x,y,z".')
    parser.add_argument("--end-cell", type=str, default="", help='Cell as "x,y,z".')
    parser.add_argument("--start-index", type=int, default=-1, help="Row index in CSV.")
    parser.add_argument("--end-index", type=int, default=-1, help="Row index in CSV.")
    parser.add_argument("--steps", type=int, default=120, help="Interpolation frames.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--path-mode",
        type=str,
        default="joint",
        choices=["joint", "cartesian"],
        help="Interpolation mode: straight in joint space or repertoire-waypoint Cartesian approximation.",
    )
    parser.add_argument(
        "--waypoint-count",
        type=int,
        default=9,
        help="Number of straight-line Cartesian samples used to choose repertoire anchors in cartesian mode.",
    )
    parser.add_argument(
        "--no-robot-render",
        action="store_true",
        help="Disable MuJoCo camera rendering (useful for headless systems).",
    )
    parser.add_argument(
        "--auto-joint-penalty",
        type=float,
        default=0.08,
        help="Auto pair selection score = descriptor_distance - penalty*joint_distance.",
    )
    return parser.parse_args()


def _parse_cell(cell_str: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in cell_str.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid cell '{cell_str}'. Expected x,y,z.")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _load_repertoire(path: str) -> list[EliteRow]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing repertoire CSV: {path}")

    rows: list[EliteRow] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"x_bin", "y_bin", "z_bin", "r", "theta", "z", "score", "q1", "q2", "q3", "q4", "q5", "q6", "q7"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV is missing required columns.")
        for idx, rw in enumerate(reader):
            r = float(rw["r"])
            theta = float(rw["theta"])
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = float(rw["z"])
            q = np.array([float(rw[f"q{i}"]) for i in range(1, 8)], dtype=np.float64)
            rows.append(
                EliteRow(
                    row_index=idx,
                    x_bin=int(rw["x_bin"]),
                    y_bin=int(rw["y_bin"]),
                    z_bin=int(rw["z_bin"]),
                    x=float(x),
                    y=float(y),
                    z=z,
                    q=q,
                    score=float(rw["score"]),
                )
            )
    if not rows:
        raise ValueError("Repertoire CSV is empty.")
    return rows


def _pick_by_cell(rows: list[EliteRow], cell: tuple[int, int, int]) -> EliteRow:
    for r in rows:
        if (r.x_bin, r.y_bin, r.z_bin) == cell:
            return r
    raise ValueError(f"Cell {cell} not found in repertoire.")


def _pick_by_index(rows: list[EliteRow], idx: int) -> EliteRow:
    if idx < 0 or idx >= len(rows):
        raise ValueError(f"Row index {idx} out of range [0,{len(rows)-1}].")
    return rows[idx]


def _pick_auto_pair(rows: list[EliteRow], joint_penalty: float) -> tuple[EliteRow, EliteRow]:
    best: tuple[float, int, int] | None = None
    for i, a in enumerate(rows):
        for j in range(i + 1, len(rows)):
            b = rows[j]
            desc_dist = float(np.linalg.norm(np.array([a.x - b.x, a.y - b.y, a.z - b.z], dtype=np.float64)))
            joint_dist = float(np.linalg.norm(a.q - b.q))
            value = desc_dist - joint_penalty * joint_dist
            if best is None or value > best[0]:
                best = (value, i, j)
    if best is None:
        raise ValueError("Unable to select pair from repertoire.")
    return rows[best[1]], rows[best[2]]


def _select_pair(rows: list[EliteRow], args: argparse.Namespace) -> tuple[EliteRow, EliteRow]:
    has_cell = bool(args.start_cell) and bool(args.end_cell)
    has_index = args.start_index >= 0 and args.end_index >= 0
    if has_cell and has_index:
        raise ValueError("Use either --start/end-cell or --start/end-index, not both.")
    if has_cell:
        return _pick_by_cell(rows, _parse_cell(args.start_cell)), _pick_by_cell(rows, _parse_cell(args.end_cell))
    if has_index:
        return _pick_by_index(rows, args.start_index), _pick_by_index(rows, args.end_index)
    return _pick_auto_pair(rows, args.auto_joint_penalty)


def _build_q_path(q_start: np.ndarray, q_end: np.ndarray, steps: int) -> np.ndarray:
    if steps < 2:
        raise ValueError("--steps must be >= 2.")
    alpha = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    return (1.0 - alpha[:, None]) * q_start[None, :] + alpha[:, None] * q_end[None, :]


def _build_linear_xyz_path(start_xyz: np.ndarray, end_xyz: np.ndarray, steps: int) -> np.ndarray:
    if steps < 2:
        raise ValueError("--steps must be >= 2.")
    alpha = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    return (1.0 - alpha[:, None]) * start_xyz[None, :] + alpha[:, None] * end_xyz[None, :]


def _build_repertoire_waypoint_q_path(
    rows: list[EliteRow],
    start: EliteRow,
    end: EliteRow,
    steps: int,
    waypoint_count: int,
) -> tuple[np.ndarray, np.ndarray, list[EliteRow]]:
    if steps < 2:
        raise ValueError("--steps must be >= 2.")
    if waypoint_count < 2:
        raise ValueError("--waypoint-count must be >= 2.")

    line_xyz = _build_linear_xyz_path(
        np.array([start.x, start.y, start.z], dtype=np.float64),
        np.array([end.x, end.y, end.z], dtype=np.float64),
        waypoint_count,
    )
    row_xyz = np.array([[r.x, r.y, r.z] for r in rows], dtype=np.float64)

    anchors: list[EliteRow] = [start]
    for i in range(1, waypoint_count - 1):
        target = line_xyz[i]
        d = np.linalg.norm(row_xyz - target[None, :], axis=1)
        pick = rows[int(np.argmin(d))]
        if pick.row_index != anchors[-1].row_index:
            anchors.append(pick)
    if anchors[-1].row_index != end.row_index:
        anchors.append(end)

    segs = len(anchors) - 1
    if segs <= 0:
        return np.repeat(start.q[None, :], steps, axis=0), line_xyz, anchors

    base = (steps - 1) // segs
    rem = (steps - 1) % segs
    chunks = []
    for s in range(segs):
        n = base + (1 if s < rem else 0)
        a = anchors[s].q
        b = anchors[s + 1].q
        t = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
        seg_q = (1.0 - t[:, None]) * a[None, :] + t[:, None] * b[None, :]
        if s > 0:
            seg_q = seg_q[1:]
        chunks.append(seg_q)
    return np.vstack(chunks), line_xyz, anchors


def _compute_ee_path(model: mujoco.MjModel, data: mujoco.MjData, q_path: np.ndarray) -> np.ndarray:
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in FR3_JOINT_NAMES]
    qpos_addrs = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int32)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)

    ee = np.zeros((q_path.shape[0], 3), dtype=np.float64)
    for i, q in enumerate(q_path):
        data.qpos[qpos_addrs] = q
        mujoco.mj_forward(model, data)
        ee[i] = data.site_xpos[site_id].copy()
    return ee


def _render_frames(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_path: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray | None:
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in FR3_JOINT_NAMES]
    qpos_addrs = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int32)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    try:
        renderer = mujoco.Renderer(model, width=width, height=height)
    except Exception as exc:  # pragma: no cover - depends on host OpenGL setup
        print(f"Robot rendering unavailable, using fallback panel: {exc}")
        return None

    frames = np.zeros((q_path.shape[0], height, width, 3), dtype=np.uint8)
    for i, q in enumerate(q_path):
        data.qpos[qpos_addrs] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera_id)
        frames[i] = renderer.render()
    renderer.close()
    return frames


def _build_animation(
    rows: list[EliteRow],
    start: EliteRow,
    end: EliteRow,
    q_path: np.ndarray,
    ee_path: np.ndarray,
    desired_xyz_path: np.ndarray | None,
    rgb_frames: np.ndarray | None,
    out_path: str,
    fps: int,
    mode: str,
) -> None:
    fig = plt.figure(figsize=(16, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.15])
    ax_robot = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_joints = fig.add_subplot(gs[0, 2])

    # Panel 1: Robot render if available, otherwise EE XYZ trajectory.
    if rgb_frames is not None:
        im = ax_robot.imshow(rgb_frames[0])
        ax_robot.set_title("Robot Joint Interpolation")
        ax_robot.axis("off")
        fallback_lines = None
    else:
        (line_x,) = ax_robot.plot([], [], color="tab:red", label="x(t)")
        (line_y,) = ax_robot.plot([], [], color="tab:green", label="y(t)")
        (line_z,) = ax_robot.plot([], [], color="tab:blue", label="z(t)")
        ax_robot.set_title("End-Effector Coordinates")
        ax_robot.set_xlabel("Interpolation t")
        ax_robot.set_ylabel("Position (m)")
        ax_robot.grid(alpha=0.25)
        ax_robot.legend(loc="best", fontsize=8)
        ax_robot.set_xlim(0.0, 1.0)
        lo = float(np.min(ee_path))
        hi = float(np.max(ee_path))
        pad = 0.06 * max(1e-6, hi - lo)
        ax_robot.set_ylim(lo - pad, hi + pad)
        im = None
        fallback_lines = (line_x, line_y, line_z)

    # Panel 2: Top-down repertoire map.
    all_x = np.array([r.x for r in rows], dtype=np.float64)
    all_y = np.array([r.y for r in rows], dtype=np.float64)
    all_score = np.array([r.score for r in rows], dtype=np.float64)
    sc = ax_map.scatter(all_x, all_y, c=all_score, cmap="viridis_r", s=12, alpha=0.6)
    if desired_xyz_path is None:
        ax_map.plot([start.x, end.x], [start.y, end.y], "w--", linewidth=1.5, label="Desired line")
    else:
        ax_map.plot(desired_xyz_path[:, 0], desired_xyz_path[:, 1], "w--", linewidth=1.5, label="Desired line")
    (traj_line,) = ax_map.plot([], [], color="tab:red", linewidth=2.0, label="Measured EE path")
    (curr_pt,) = ax_map.plot([], [], "o", color="tab:red", markersize=6)
    ax_map.plot(start.x, start.y, "o", color="lime", markersize=7, label="Start")
    ax_map.plot(end.x, end.y, "o", color="cyan", markersize=7, label="End")
    ax_map.set_title("Repertoire Top-Down (XY)")
    ax_map.set_xlabel("X")
    ax_map.set_ylabel("Y")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.legend(loc="best", fontsize=8)
    cbar = fig.colorbar(sc, ax=ax_map, fraction=0.045, pad=0.02)
    cbar.set_label("Score")

    # Panel 3: Joint trajectories.
    t = np.linspace(0.0, 1.0, q_path.shape[0], dtype=np.float64)
    for j in range(q_path.shape[1]):
        ax_joints.plot(t, q_path[:, j], linewidth=1.4, label=f"q{j+1}")
    cursor = ax_joints.axvline(0.0, color="k", linestyle="--", linewidth=1.2)
    ax_joints.set_title("Joint Trajectories")
    ax_joints.set_xlabel("Interpolation t")
    ax_joints.set_ylabel("Joint Position (rad)")
    ax_joints.grid(alpha=0.25)
    ax_joints.legend(ncol=2, fontsize=8, loc="best")

    status_text = fig.text(
        0.5,
        0.02,
        (
            f"Start cell=({start.x_bin},{start.y_bin},{start.z_bin})  "
            f"End cell=({end.x_bin},{end.y_bin},{end.z_bin})  "
            f"Mode={mode}"
        ),
        ha="center",
        va="center",
        fontsize=10,
    )
    _ = status_text

    def _update(frame_idx: int):
        artists = []
        if rgb_frames is not None and im is not None:
            im.set_data(rgb_frames[frame_idx])
            artists.append(im)
        elif fallback_lines is not None:
            line_x, line_y, line_z = fallback_lines
            t_curr = t[: frame_idx + 1]
            line_x.set_data(t_curr, ee_path[: frame_idx + 1, 0])
            line_y.set_data(t_curr, ee_path[: frame_idx + 1, 1])
            line_z.set_data(t_curr, ee_path[: frame_idx + 1, 2])
            artists.extend([line_x, line_y, line_z])
        traj_line.set_data(ee_path[: frame_idx + 1, 0], ee_path[: frame_idx + 1, 1])
        curr_pt.set_data([ee_path[frame_idx, 0]], [ee_path[frame_idx, 1]])
        cursor.set_xdata([t[frame_idx], t[frame_idx]])
        artists.extend([traj_line, curr_pt, cursor])
        return tuple(artists)

    anim = FuncAnimation(fig, _update, frames=q_path.shape[0], interval=1000.0 / max(fps, 1), blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if out_path.lower().endswith(".gif"):
        anim.save(out_path, writer=PillowWriter(fps=fps))
    else:
        try:
            anim.save(out_path, fps=fps, dpi=150)
        except Exception:
            fallback = os.path.splitext(out_path)[0] + ".gif"
            anim.save(fallback, writer=PillowWriter(fps=fps))
            out_path = fallback

    plt.close(fig)
    print(f"Saved interpolation demo: {out_path}")


def main() -> None:
    args = _parse_args()
    rows = _load_repertoire(args.repertoire_csv)
    start, end = _select_pair(rows, args)

    model = mujoco.MjModel.from_xml_path(args.scene_xml)
    data = mujoco.MjData(model)

    line_start = np.array([start.x, start.y, start.z], dtype=np.float64)
    line_end = np.array([end.x, end.y, end.z], dtype=np.float64)
    desired_xyz_path = _build_linear_xyz_path(line_start, line_end, args.steps)
    if args.path_mode == "joint":
        q_path = _build_q_path(start.q, end.q, args.steps)
    else:
        q_path, _, anchors = _build_repertoire_waypoint_q_path(
            rows=rows,
            start=start,
            end=end,
            steps=args.steps,
            waypoint_count=args.waypoint_count,
        )
        print(f"Repertoire anchors used: {len(anchors)}")

    ee_path = _compute_ee_path(model, data, q_path)
    rgb_frames = None if args.no_robot_render else _render_frames(model, data, q_path, width=args.width, height=args.height)

    desc_dist = float(np.linalg.norm(np.array([start.x - end.x, start.y - end.y, start.z - end.z], dtype=np.float64)))
    joint_dist = float(np.linalg.norm(start.q - end.q))
    print(
        "Selected pair:",
        f"start_cell=({start.x_bin},{start.y_bin},{start.z_bin})",
        f"end_cell=({end.x_bin},{end.y_bin},{end.z_bin})",
        f"descriptor_dist={desc_dist:.4f}",
        f"joint_dist={joint_dist:.4f}",
    )
    line_dir = line_end - line_start
    line_norm = float(np.linalg.norm(line_dir))
    if line_norm > 1e-9:
        unit = line_dir / line_norm
        proj_len = (ee_path - line_start[None, :]) @ unit
        closest = line_start[None, :] + proj_len[:, None] * unit[None, :]
        dev = np.linalg.norm(ee_path - closest, axis=1)
        print(f"Max distance from straight XYZ line: {float(np.max(dev)):.4f} m")

    _build_animation(
        rows=rows,
        start=start,
        end=end,
        q_path=q_path,
        ee_path=ee_path,
        desired_xyz_path=desired_xyz_path,
        rgb_frames=rgb_frames,
        out_path=args.out,
        fps=args.fps,
        mode=args.path_mode,
    )


if __name__ == "__main__":
    main()
