"""
Side-by-side: v1 con azzeramento vs v1 senza azzeramento.

Left  : k_worst_mean_mm  from comparator_output_zero   (auto-zero ON)
Right : k_worst_mean_mm  from comparator_output_nozero (auto-zero OFF)

Thresholds (v1):
  normal     : <= -0.21
  near-mastic: <= -0.14

Output: comparison_output/sidebyside_zerovsnozero/
"""

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

PLY_DIR       = "data/pc"
CSV_ZERO      = "comparator_output_zero/all_measurements_v1.csv"
CSV_NOZERO    = "comparator_output_nozero/all_measurements_v1.csv"
OUT_DIR       = "comparison_output/sidebyside_zerovsnozero"
os.makedirs(OUT_DIR, exist_ok=True)

THRESH_NORMAL = -0.21
THRESH_MASTIC = -0.14
MASTIC_PROX_ROWS = 2

df_zero   = pd.read_csv(CSV_ZERO)
df_nozero = pd.read_csv(CSV_NOZERO)


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
        d = np.linalg.norm(rivet_xy - m["center"][:2], axis=1)
        mask |= (d - m["radius_mm"]) < prox
    return mask


def make_panel(ax, pts, mastics, df, mean_spacing, title):
    step = max(1, len(pts) // 200_000)
    ax.scatter(pts[::step, 0], pts[::step, 1],
               s=0.1, c="lightgray", rasterized=True, zorder=1)

    for m in mastics:
        mv = pts[m["verts"]]
        ax.scatter(mv[:, 0], mv[:, 1], s=1, c="orange",
                   alpha=0.5, linewidths=0, zorder=2)

    rivet_xy = df[["cx", "cy"]].values
    near   = near_mastic_mask(rivet_xy, mastics, mean_spacing)
    thresh = np.where(near, THRESH_MASTIC, THRESH_NORMAL)
    vals   = df["k_worst_mean_mm"].values
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


for ply_path in sorted(glob.glob(f"{PLY_DIR}/*.ply")):
    label = os.path.splitext(os.path.basename(ply_path))[0]

    dz  = df_zero  [df_zero  ["surface"] == label]
    dnz = df_nozero[df_nozero["surface"] == label]
    if dz.empty or dnz.empty:
        print(f"  skip {label} — missing CSV data")
        continue

    print(f"  {label} …", end=" ", flush=True)
    mastics, pts = load_mastics(ply_path)

    if len(dz) >= 2:
        centers = dz[["cx","cy"]].values
        tree    = cKDTree(centers)
        nn_d, _ = tree.query(centers, k=2)
        mean_spacing = float(nn_d[:, 1].mean())
    else:
        mean_spacing = 20.0

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(label, fontsize=11, fontweight="bold")

    make_panel(axes[0], pts, mastics, dz, mean_spacing,
               title=f"v1  CON azzeramento  (thr {THRESH_NORMAL} | mastic {THRESH_MASTIC})")

    make_panel(axes[1], pts, mastics, dnz, mean_spacing,
               title=f"v1  SENZA azzeramento  (thr {THRESH_NORMAL} | mastic {THRESH_MASTIC})")

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/{label}_zerovsnozero.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("done")

print(f"\n→ {OUT_DIR}/")
