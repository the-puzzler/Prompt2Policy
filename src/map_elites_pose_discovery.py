"""MAP-Elites pose discovery for Franka FR3 in MuJoCo.

This script searches over 7D joint configurations and fills a 3D archive
using a cylindrical descriptor (r, theta, z) from the attachment site.
Feasibility is checked by interpolating from home to the candidate pose and
rejecting disallowed collisions.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
from dataclasses import dataclass

import mujoco
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_XML = os.path.join(PROJECT_ROOT, "mujoco_menagerie", "franka_fr3", "reach_scene.xml")
FR3_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
DEFAULT_HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=np.float64)
MOVEMENT_WEIGHT = 1.0
EVENNESS_WEIGHT = 0.8


@dataclass
class Elite:
    q: np.ndarray
    score: float
    descriptor: tuple[float, float, float]


@dataclass
class EvalResult:
    valid: bool
    q: np.ndarray
    score: float
    cell: tuple[int, int, int]
    descriptor: tuple[float, float, float]


class WorkerContext:
    def __init__(
        self,
        scene_xml: str,
        grid_bins: int,
        interp_steps: int,
        r_min: float,
        r_max: float,
        z_min: float,
        z_max: float,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)

        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in FR3_JOINT_NAMES
        ]
        self.qpos_addrs = np.array([self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.joint_ranges = self.model.jnt_range[self.joint_ids].copy()
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

        self.home_qpos = DEFAULT_HOME_QPOS.copy()
        self.grid_bins = grid_bins
        self.interp_steps = interp_steps

        self.r_min = r_min
        self.r_max = r_max
        self.z_min = z_min
        self.z_max = z_max

        self.allowed_pairs = self._compute_home_allowed_contact_pairs()
        self.range_spans = self.joint_ranges[:, 1] - self.joint_ranges[:, 0]
        self.range_spans[self.range_spans <= 0.0] = 1.0

    def _compute_home_allowed_contact_pairs(self) -> set[tuple[int, int]]:
        self.data.qpos[self.qpos_addrs] = self.home_qpos
        mujoco.mj_forward(self.model, self.data)
        pairs: set[tuple[int, int]] = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            pairs.add((g1, g2) if g1 < g2 else (g2, g1))
        return pairs

    def _has_disallowed_contact(self) -> bool:
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            pair = (g1, g2) if g1 < g2 else (g2, g1)
            if pair not in self.allowed_pairs:
                return True
        return False

    def _descriptor_from_data(self) -> tuple[float, float, float]:
        pos = self.data.site_xpos[self.ee_site_id]
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        r = float(np.hypot(x, y))
        theta = float(np.arctan2(y, x))
        return (r, theta, z)

    def _bin_1d(self, val: float, lo: float, hi: float) -> int:
        if hi <= lo:
            return 0
        alpha = (val - lo) / (hi - lo)
        b = int(np.floor(alpha * self.grid_bins))
        return int(np.clip(b, 0, self.grid_bins - 1))

    def _to_cell(self, descriptor: tuple[float, float, float]) -> tuple[int, int, int]:
        r, theta, z = descriptor
        r_bin = self._bin_1d(r, self.r_min, self.r_max)
        theta_bin = self._bin_1d(theta, -np.pi, np.pi)
        z_bin = self._bin_1d(z, self.z_min, self.z_max)
        return (r_bin, theta_bin, z_bin)

    def _score(self, q: np.ndarray) -> float:
        # Local competition rule for simplicity with shared motion:
        # 1) minimize total normalized movement from home
        # 2) minimize unevenness across joints
        d = (q - self.home_qpos) / self.range_spans
        d_abs = np.abs(d)
        movement = float(np.mean(d_abs))
        unevenness = float(np.std(d_abs))
        return MOVEMENT_WEIGHT * movement + EVENNESS_WEIGHT * unevenness

    def evaluate(self, q_target: np.ndarray) -> EvalResult:
        q = np.clip(q_target, self.joint_ranges[:, 0], self.joint_ranges[:, 1]).astype(np.float64, copy=False)

        # Interpolate in joint space from home to target and reject on collision.
        for alpha in np.linspace(0.0, 1.0, self.interp_steps):
            q_interp = self.home_qpos + alpha * (q - self.home_qpos)
            if np.any(q_interp < self.joint_ranges[:, 0]) or np.any(q_interp > self.joint_ranges[:, 1]):
                return EvalResult(
                    valid=False,
                    q=q,
                    score=float("inf"),
                    cell=(-1, -1, -1),
                    descriptor=(0.0, 0.0, 0.0),
                )

            self.data.qpos[self.qpos_addrs] = q_interp
            mujoco.mj_forward(self.model, self.data)
            if self._has_disallowed_contact():
                return EvalResult(
                    valid=False,
                    q=q,
                    score=float("inf"),
                    cell=(-1, -1, -1),
                    descriptor=(0.0, 0.0, 0.0),
                )

        descriptor = self._descriptor_from_data()
        return EvalResult(valid=True, q=q, score=self._score(q), cell=self._to_cell(descriptor), descriptor=descriptor)


WORKER_CTX: WorkerContext | None = None


def _worker_init(
    scene_xml: str,
    grid_bins: int,
    interp_steps: int,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
) -> None:
    global WORKER_CTX
    WORKER_CTX = WorkerContext(
        scene_xml=scene_xml,
        grid_bins=grid_bins,
        interp_steps=interp_steps,
        r_min=r_min,
        r_max=r_max,
        z_min=z_min,
        z_max=z_max,
    )


def _evaluate_candidate(q_target: np.ndarray) -> EvalResult:
    if WORKER_CTX is None:
        raise RuntimeError("Worker context not initialized.")
    return WORKER_CTX.evaluate(q_target)


def _build_candidates(
    rng: np.random.Generator,
    batch_size: int,
    archive: dict[tuple[int, int, int], Elite],
    joint_ranges: np.ndarray,
    random_prob: float,
    mutation_sigma: float,
) -> list[np.ndarray]:
    spans = joint_ranges[:, 1] - joint_ranges[:, 0]
    candidates: list[np.ndarray] = []

    for _ in range(batch_size):
        use_random = (not archive) or (rng.random() < random_prob)
        if use_random:
            q = rng.uniform(joint_ranges[:, 0], joint_ranges[:, 1])
        else:
            keys = list(archive.keys())
            parent = archive[keys[int(rng.integers(0, len(keys)))]].q
            noise = rng.normal(0.0, mutation_sigma * spans)
            q = np.clip(parent + noise, joint_ranges[:, 0], joint_ranges[:, 1])
        candidates.append(q.astype(np.float64, copy=False))
    return candidates


def _save_repertoire(
    archive: dict[tuple[int, int, int], Elite],
    out_dir: str,
    grid_bins: int,
) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "repertoire.csv")
    txt_path = os.path.join(out_dir, "repertoire.txt")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "x_bin",
            "y_bin",
            "z_bin",
            "r",
            "theta",
            "z",
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "q7",
            "score",
        ])
        for x in range(grid_bins):
            for y in range(grid_bins):
                for z in range(grid_bins):
                    cell = (x, y, z)
                    if cell not in archive:
                        continue
                    elite = archive[cell]
                    writer.writerow([x, y, z, *elite.descriptor, *elite.q.tolist(), elite.score])

    with open(txt_path, "w", encoding="utf-8") as f:
        for x in range(grid_bins):
            for y in range(grid_bins):
                for z in range(grid_bins):
                    cell = (x, y, z)
                    if cell not in archive:
                        continue
                    elite = archive[cell]
                    q_str = ", ".join(f"{v:.6f}" for v in elite.q.tolist())
                    f.write(f"({x},{y},{z}) : [{q_str}] : {elite.score:.8f}\n")

    return csv_path, txt_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAP-Elites pose discovery for FR3.")
    parser.add_argument("--scene-xml", type=str, default=SCENE_XML)
    parser.add_argument("--grid-bins", type=int, default=4, help="Bins per descriptor dimension.")
    parser.add_argument("--evaluations", type=int, default=1000)
    parser.add_argument("--interp-steps", type=int, default=50)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-prob", type=float, default=0.30, help="Probability of global random sample.")
    parser.add_argument("--mutation-sigma", type=float, default=0.12, help="Mutation std as fraction of joint span.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--r-min", type=float, default=0.0)
    parser.add_argument("--r-max", type=float, default=1.0)
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=1.2)
    parser.add_argument("--out-dir", type=str, default=os.path.join(PROJECT_ROOT, "src", "runs", "fr3_map_elites"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.random_prob < 0.0 or args.random_prob > 1.0:
        raise ValueError("--random-prob must be in [0,1].")

    model = mujoco.MjModel.from_xml_path(args.scene_xml)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in FR3_JOINT_NAMES]
    joint_ranges = model.jnt_range[joint_ids].copy()

    rng = np.random.default_rng(args.seed)
    archive: dict[tuple[int, int, int], Elite] = {}
    valid_evals = 0

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(
            args.scene_xml,
            args.grid_bins,
            args.interp_steps,
            args.r_min,
            args.r_max,
            args.z_min,
            args.z_max,
        ),
    ) as pool:
        eval_count = 0
        while eval_count < args.evaluations:
            remaining = args.evaluations - eval_count
            batch_size = min(args.batch_size, remaining)

            candidates = _build_candidates(
                rng=rng,
                batch_size=batch_size,
                archive=archive,
                joint_ranges=joint_ranges,
                random_prob=args.random_prob,
                mutation_sigma=args.mutation_sigma,
            )
            results = pool.map(_evaluate_candidate, candidates)

            for res in results:
                if not res.valid:
                    continue
                valid_evals += 1
                current = archive.get(res.cell)
                if current is None or res.score < current.score:
                    archive[res.cell] = Elite(q=res.q, score=res.score, descriptor=res.descriptor)

            eval_count += len(results)
            if eval_count % max(1, args.batch_size * 2) == 0 or eval_count == args.evaluations:
                print(
                    f"[{eval_count:4d}/{args.evaluations}] "
                    f"valid={valid_evals:4d} "
                    f"filled={len(archive):2d}/{args.grid_bins ** 3}"
                )

    csv_path, txt_path = _save_repertoire(archive, args.out_dir, args.grid_bins)

    print("\nDone.")
    print(f"Evaluations: {args.evaluations}")
    print(f"Valid evaluations: {valid_evals}")
    print(f"Filled cells: {len(archive)}/{args.grid_bins ** 3}")
    print("Descriptor basis: cylindrical (r, theta, z)")
    print(f"Generator mix: random={args.random_prob:.2f}, mutation={1.0 - args.random_prob:.2f}")
    print(
        "Local competition: minimize movement + unevenness "
        f"(w_move={MOVEMENT_WEIGHT}, w_even={EVENNESS_WEIGHT})"
    )
    print(f"CSV repertoire: {csv_path}")
    print(f"Text repertoire: {txt_path}")


if __name__ == "__main__":
    main()
