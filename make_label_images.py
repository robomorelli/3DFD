"""
Generate annotated surface images and default label CSV files for rivet annotation.

For each surface in the merged dataset (mesh3D preferred over pc):
  1. Detects rivet holes
  2. Plots the point cloud with each rivet clearly numbered
  3. Saves <surface>_labels.png in --out-dir/
  4. Saves <surface>_labels.csv with columns:
       hole_idx, cx, cy, cz, label
     label = 0 (OK) by default — change to 1 = warn, 2 = defect

After annotating, merge all CSVs and train:
  python train_zero_selector.py --gt-csv labels/all_labels.csv

Usage:
  python make_label_images.py
  python make_label_images.py --out-dir labels --workers 4
  python make_label_images.py --surface Surface140_clean   # single surface
"""

import argparse
import csv
import glob
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate rivet label images and default CSV files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mesh-dir",  default="data/mesh3D")
    p.add_argument("--pc-dir",    default="data/pc")
    p.add_argument("--out-dir",   default="labels")
    p.add_argument("--surface",   default=None,
                   help="Process only this surface label (e.g. Surface140_clean). "
                        "Default: all merged surfaces.")
    p.add_argument("--hole-r-min", type=float, default=1.0)
    p.add_argument("--hole-r-max", type=float, default=15.0)
    p.add_argument("--dpi",        type=int,   default=200)
    p.add_argument("--fig-w",      type=float, default=14.0, help="Figure width (inches)")
    p.add_argument("--fig-h",      type=float, default=12.0, help="Figure height (inches)")
    p.add_argument("--workers",    type=int,   default=4)
    return p.parse_args()


# ── File list ─────────────────────────────────────────────────────────────────
def _surface_id(path):
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+)", name)
    return m.group(1) if m else None


def _build_file_list(mesh_dir, pc_dir):
    mesh_files = sorted(glob.glob(os.path.join(mesh_dir, "*.ply")))
    pc_files   = sorted(glob.glob(os.path.join(pc_dir,   "*.ply")))
    by_id = {}
    for p in mesh_files:
        sid = _surface_id(p)
        if sid and sid not in by_id:
            by_id[sid] = p
    for p in pc_files:
        sid = _surface_id(p)
        if sid and sid not in by_id:
            by_id[sid] = p
    return sorted(by_id.values())


# ── Per-surface worker ────────────────────────────────────────────────────────
def process_one(ply_path, out_dir, args_dict):
    from virtual_comparator import find_holes

    a     = args_dict
    label = os.path.splitext(os.path.basename(ply_path))[0]
    t0    = time.time()

    mesh  = trimesh.load(ply_path, process=False)
    pts   = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces,    dtype=np.int32)

    holes, mastics = find_holes(faces, pts,
                                r_min=a["hole_r_min"], r_max=a["hole_r_max"])
    if not holes:
        return {"label": label, "n_holes": 0, "elapsed": round(time.time() - t0, 1)}

    # ── Annotated PNG ─────────────────────────────────────────────────────────
    step = max(1, len(pts) // 400_000)
    fig, ax = plt.subplots(figsize=(a["fig_w"], a["fig_h"]))
    fig.suptitle(
        f"{label}  —  {len(holes)} rivets\n"
        "Number = hole_idx  ·  Default label = 0 (OK)\n"
        "Change to 1 = warn  /  2 = defect in the CSV",
        fontsize=11, fontweight="bold",
    )

    ax.scatter(pts[::step, 0], pts[::step, 1], c=pts[::step, 2],
               s=0.15, cmap="gray", rasterized=True, alpha=0.4)

    # Determine font size: scale with rivet density
    # Aim for labels that don't overlap; reduce if many holes
    n    = len(holes)
    fsize = max(4, min(9, int(120 / max(n, 1) ** 0.5)))

    for i, hole in enumerate(holes):
        c   = hole["center"]
        r   = hole["radius_mm"]
        idx = i + 1   # 1-based

        # Hole circle (light blue, semi-transparent)
        ax.add_patch(plt.Circle((c[0], c[1]), r,
                                color="steelblue", fill=True,
                                alpha=0.6, linewidth=0))
        # Crown ring for visual reference (~7mm from edge)
        ax.add_patch(plt.Circle((c[0], c[1]), r + 7,
                                color="steelblue", fill=False,
                                alpha=0.25, linewidth=0.5, linestyle="--"))

        # Index label: white text with dark border for contrast
        ax.annotate(
            str(idx), (c[0], c[1]),
            fontsize=fsize, ha="center", va="center",
            fontweight="bold", color="white",
            path_effects=[
                matplotlib.patheffects.withStroke(linewidth=1.5, foreground="black")
            ],
        )

    # Mastic zones (orange)
    for m in mastics:
        mv = pts[m["verts"]]
        ax.scatter(mv[:, 0], mv[:, 1], s=0.8, c="orange",
                   alpha=0.4, linewidths=0, zorder=3)

    legend_elems = [
        mpatches.Patch(color="steelblue", label="rivet hole (default label=0)"),
        mpatches.Patch(color="orange",    label="mastic zone"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="upper right")
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")

    out_png = os.path.join(out_dir, f"{label}_labels.png")
    plt.tight_layout()
    fig.savefig(out_png, dpi=a["dpi"], bbox_inches="tight")
    plt.close(fig)

    # ── Default label CSV ─────────────────────────────────────────────────────
    out_csv = os.path.join(out_dir, f"{label}_labels.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hole_idx", "cx", "cy", "cz", "label"])
        for i, hole in enumerate(holes):
            c = hole["center"]
            w.writerow([i + 1,
                        round(float(c[0]), 2),
                        round(float(c[1]), 2),
                        round(float(c[2]), 2),
                        0])

    elapsed = round(time.time() - t0, 1)
    return {"label": label, "n_holes": len(holes), "elapsed": elapsed,
            "png": out_png, "csv": out_csv}


# ── Merge all labelled CSVs into one ─────────────────────────────────────────
def merge_label_csvs(out_dir):
    """
    Concatenate all per-surface label CSVs into a single all_labels.csv,
    adding a 'surface' column.  Skips rows with label=0 to keep the
    combined file focused on labeled (warn/defect) examples.

    Note: all_labels.csv includes ALL rows (label 0, 1, 2) — the training
    script uses them all.
    """
    all_rows = []
    for csv_path in sorted(glob.glob(os.path.join(out_dir, "*_labels.csv"))):
        surface = os.path.basename(csv_path).replace("_labels.csv", "")
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["surface"] = surface
                all_rows.append(row)

    if not all_rows:
        return None

    out_path = os.path.join(out_dir, "all_labels.csv")
    fieldnames = ["surface", "hole_idx", "cx", "cy", "cz", "label"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames,
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.surface:
        # Single surface mode
        ply = None
        for d in [args.mesh_dir, args.pc_dir]:
            for f in glob.glob(os.path.join(d, "*.ply")):
                if os.path.splitext(os.path.basename(f))[0] == args.surface:
                    ply = f
                    break
            if ply:
                break
        if not ply:
            print(f"ERROR: surface '{args.surface}' not found.")
            return
        file_list = [ply]
    else:
        file_list = _build_file_list(args.mesh_dir, args.pc_dir)

    print(f"\nGenerating label images for {len(file_list)} surfaces → {args.out_dir}/")
    args_dict = {
        "hole_r_min": args.hole_r_min,
        "hole_r_max": args.hole_r_max,
        "dpi":        args.dpi,
        "fig_w":      args.fig_w,
        "fig_h":      args.fig_h,
    }

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, f, args.out_dir, args_dict): f
                   for f in file_list}
        done = 0
        for fut in as_completed(futures):
            res   = fut.result()
            done += 1
            print(f"  [{done:>2}/{len(file_list)}] {res['label']:<30}  "
                  f"{res['n_holes']:>3} rivets  {res['elapsed']}s")

    # Merge into all_labels.csv
    merged = merge_label_csvs(args.out_dir)
    if merged:
        print(f"\nMerged CSV → {merged}")
    print(f"\nWorkflow:")
    print(f"  1. Open each *_labels.png in {args.out_dir}/ to identify rivet IDs")
    print(f"  2. Edit each *_labels.csv: set label=1 (warn) or 2 (defect)")
    print(f"  3. Run: python train_zero_selector.py --gt-csv {args.out_dir}/all_labels.csv")
    print(f"     (regenerate all_labels.csv first with: python make_label_images.py --no-plots)")
    print(f"\nDone.\n")


if __name__ == "__main__":
    # Ensure path effects import is available in workers too
    import matplotlib.patheffects
    main()