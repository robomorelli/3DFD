"""
Side-by-side v0 / v1 plots with custom thresholds and proximity-to-mastic logic.

v0 metric : bnd_mean_mm
  normal threshold   : <= -0.19
  near-mastic thresh : <= -0.13

v1 metric : k_worst_mean_mm
  normal threshold   : <= -0.21
  near-mastic thresh : <= -0.14

"near mastic" : rivet centre within (mastic_radius + 2*mean_spacing) of any
                real mastic (radius < 80 mm).

Visual style:
  defective rivet  → filled red circle + white score
  ok rivet         → thin gray circle outline + black text of the metric value
  foot_ok=False    → gray dashed outline + value with "?"

Usage:
  python make_sidebyside_thresh.py
  python make_sidebyside_thresh.py --v1-csv comparator_output_zero/all_measurements_v1.csv
                                   --out-dir comparison_output/sidebyside_zero
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree

from virtual_comparator import find_holes

THRESH_V0_NORMAL  = -0.19
THRESH_V0_MASTIC  = -0.13
THRESH_V1_NORMAL  = -0.21
THRESH_V1_MASTIC  = -0.14

MASTIC_PROX_ROWS  = 2


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ply-dir",  default="data/pc")
    p.add_argument("--v0-csv",   default="comparator_output_v1/all_measurements.csv")
    p.add_argument("--v1-csv",   default="comparator_output_v1/all_measurements_v1.csv")
    p.add_argument("--out-dir",  default="comparison_output/sidebyside_thresh")
    return p.parse_args()


def load_mastics(ply_path):
    mesh  = trimesh.load(ply_path, process=False)
    pts   = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces,    dtype=np.int32)
    _, mastics = find_holes(faces, pts)
    return mastics, pts


def near_mastic_mask(rivet_xy, mastics, mean_spacing):
    prox = MASTIC_PROX_ROWS * mean_spacing
    mask = np.zeros(len(rivet_xy), dtype=bool)
    for m in mastics:
        mc = m["center"][:2]
        mr = m["radius_mm"]
        d  = np.linalg.norm(rivet_xy - mc, axis=1)
        mask |= (d - mr) < prox
    return mask


def make_panel(ax, pts, mastics, df, metric_col, thresh_normal, thresh_mastic,
               mean_spacing, title):
    step = max(1, len(pts) // 200_000)
    ax.scatter(pts[::step, 0], pts[::step, 1],
               s=0.1, c="lightgray", rasterized=True, zorder=1)

    for m in mastics:
        mv = pts[m["verts"]]
        ax.scatter(mv[:, 0], mv[:, 1], s=1, c="orange",
                   alpha=0.5, linewidths=0, zorder=2)

    rivet_xy = df[["cx", "cy"]].values
    near     = near_mastic_mask(rivet_xy, mastics, mean_spacing)
    thresh   = np.where(near, thresh_mastic, thresh_normal)
    vals     = df[metric_col].values
    foot_ok_col = "foot_ok" in df.columns

    for i, row in df.iterrows():
        idx    = list(df.index).index(i)
        v      = vals[idx]
        cx, cy = row["cx"], row["cy"]
        r      = row["hole_r_mm"]
        thr    = thresh[idx]
        fok    = bool(row["foot_ok"]) if foot_ok_col else True

        if not fok:
            circle = plt.Circle((cx, cy), r, color="gray",
                                 fill=False, linewidth=0.6,
                                 linestyle="--", zorder=3)
            ax.add_patch(circle)
            ax.annotate(f"{v:.2f}?", (cx, cy), fontsize=3.5,
                        ha="center", va="center", color="gray", zorder=5)
        elif v <= thr:
            circle = plt.Circle((cx, cy), r, color="red",
                                 zorder=4, linewidth=0)
            ax.add_patch(circle)
            ax.annotate(f"{v:.2f}", (cx, cy), fontsize=3.5,
                        ha="center", va="center", color="white",
                        fontweight="bold", zorder=5)
        else:
            circle = plt.Circle((cx, cy), r, color="gray",
                                 fill=False, linewidth=0.6, zorder=3)
            ax.add_patch(circle)
            ax.annotate(f"{v:.2f}", (cx, cy), fontsize=3.5,
                        ha="center", va="center", color="black", zorder=5)

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("X (mm)", fontsize=7)
    ax.set_ylabel("Y (mm)", fontsize=7)
    ax.tick_params(labelsize=6)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    v0_all = pd.read_csv(args.v0_csv)
    v1_all = pd.read_csv(args.v1_csv)

    ply_files = sorted(glob.glob(f"{args.ply_dir}/*.ply"))
    for ply_path in ply_files:
        label = os.path.splitext(os.path.basename(ply_path))[0]

        df0 = v0_all[v0_all["surface"] == label]
        df1 = v1_all[v1_all["surface"] == label]
        if df0.empty or df1.empty:
            print(f"  skip {label} — missing CSV data")
            continue

        print(f"  {label} …", end=" ", flush=True)
        mastics, pts = load_mastics(ply_path)

        n = len(df1)
        if n >= 2:
            centers = df1[["cx","cy"]].values
            tree    = cKDTree(centers)
            nn_d, _ = tree.query(centers, k=2)
            mean_spacing = float(nn_d[:, 1].mean())
        else:
            mean_spacing = 20.0

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(label, fontsize=11, fontweight="bold")

        make_panel(axes[0], pts, mastics, df0,
                   metric_col="bnd_mean_mm",
                   thresh_normal=THRESH_V0_NORMAL,
                   thresh_mastic=THRESH_V0_MASTIC,
                   mean_spacing=mean_spacing,
                   title=f"v0  (thr {THRESH_V0_NORMAL} | near-mastic {THRESH_V0_MASTIC})")

        make_panel(axes[1], pts, mastics, df1,
                   metric_col="k_worst_mean_mm",
                   thresh_normal=THRESH_V1_NORMAL,
                   thresh_mastic=THRESH_V1_MASTIC,
                   mean_spacing=mean_spacing,
                   title=f"v1  (thr {THRESH_V1_NORMAL} | near-mastic {THRESH_V1_MASTIC})")

        plt.tight_layout()
        out = f"{args.out_dir}/{label}_compare_thresh.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("done")

    print(f"\n→ {args.out_dir}/")


if __name__ == "__main__":
    main()
