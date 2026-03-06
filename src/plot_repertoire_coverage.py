"""Plot repertoire coverage and score maps from MAP-Elites output."""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

# Force non-interactive backend so this works on headless machines too.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REPERTOIRE = os.path.join(PROJECT_ROOT, "src", "runs", "fr3_map_elites", "repertoire.csv")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FR3 repertoire coverage and score.")
    parser.add_argument("--repertoire-csv", type=str, default=DEFAULT_REPERTOIRE)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--prefix", type=str, default="repertoire")
    parser.add_argument("--show", action="store_true", help="Also open interactive windows.")
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--density-bins", type=int, default=24)
    return parser.parse_args()


def _read_rows(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing repertoire CSV: {path}")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("Repertoire CSV is empty.")
    if reader.fieldnames is None:
        raise ValueError("CSV header missing.")
    required = {"score", "x_bin", "y_bin", "z_bin"}
    if not required.issubset(set(reader.fieldnames)):
        raise ValueError("CSV missing required columns.")
    return rows


def _extract_xyz_score(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Preferred path: direct cylindrical descriptor columns.
    if {"r", "theta", "z"}.issubset(rows[0].keys()):
        r = np.array([float(rw["r"]) for rw in rows], dtype=np.float64)
        theta = np.array([float(rw["theta"]) for rw in rows], dtype=np.float64)
        z = np.array([float(rw["z"]) for rw in rows], dtype=np.float64)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
    else:
        # Fallback for old CSV format: use bin centers as a coarse proxy.
        xb = np.array([float(rw["x_bin"]) for rw in rows], dtype=np.float64)
        yb = np.array([float(rw["y_bin"]) for rw in rows], dtype=np.float64)
        zb = np.array([float(rw["z_bin"]) for rw in rows], dtype=np.float64)
        x = xb + 0.5
        y = yb + 0.5
        z = zb + 0.5
    score = np.array([float(rw["score"]) for rw in rows], dtype=np.float64)
    return x, y, z, score


def _plot_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, score: np.ndarray, out_path: str, dpi: int) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=score, cmap="viridis_r", s=46, alpha=0.95, edgecolors="k", linewidths=0.2)
    ax.set_title("Repertoire Points (3D), Colored by Local Competition Score")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    cbar = fig.colorbar(sc, ax=ax, pad=0.12)
    cbar.set_label("Score (lower is better)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_topdown(
    x: np.ndarray,
    y: np.ndarray,
    score: np.ndarray,
    out_path: str,
    dpi: int,
    density_bins: int,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    sc = ax1.scatter(x, y, c=score, cmap="viridis_r", s=54, alpha=0.95, edgecolors="k", linewidths=0.2)
    ax1.set_title("Top-Down (XY), Colored by Score")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(sc, ax=ax1)
    cbar.set_label("Score (lower is better)")

    # Coverage density map for quick occupancy perception.
    hb = ax2.hexbin(x, y, gridsize=density_bins, cmap="magma", mincnt=1)
    ax2.set_title("Top-Down Coverage Density")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect("equal", adjustable="box")
    cbar2 = fig.colorbar(hb, ax=ax2)
    cbar2.set_label("Cells per hex")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    rows = _read_rows(args.repertoire_csv)
    x, y, z, score = _extract_xyz_score(rows)

    out_dir = args.out_dir if args.out_dir else os.path.dirname(os.path.abspath(args.repertoire_csv))
    os.makedirs(out_dir, exist_ok=True)
    out_3d = os.path.join(out_dir, f"{args.prefix}_3d_score.png")
    out_top = os.path.join(out_dir, f"{args.prefix}_topdown.png")

    _plot_3d(x, y, z, score, out_3d, args.dpi)
    _plot_topdown(x, y, score, out_top, args.dpi, args.density_bins)

    print(f"Saved 3D score plot: {out_3d}")
    print(f"Saved top-down plot: {out_top}")

    if args.show:
        # Reload images for display through backend-agnostic imshow.
        import matplotlib.image as mpimg

        img1 = mpimg.imread(out_3d)
        img2 = mpimg.imread(out_top)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(img1)
        ax1.set_title("3D Score Plot")
        ax1.axis("off")
        ax2.imshow(img2)
        ax2.set_title("Top-Down Plot")
        ax2.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
