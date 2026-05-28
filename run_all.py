"""
Run virtual_comparator on all PLY files and aggregate results.

Usage:
  python run_all.py
  python run_all.py --workers 4 --threshold -0.2 --no-interactive
"""

import argparse
import glob
import os
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import core functions directly (avoids subprocess overhead)
import sys
sys.path.insert(0, os.path.dirname(__file__))
from virtual_comparator import (
    find_holes, measure, local_frame,
)
import trimesh
from scipy.spatial import cKDTree


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ply-dir",    default="data/pc",       help="Directory with PLY files")
    p.add_argument("--out-dir",    default="comparator_output_v1", help="Output directory")
    p.add_argument("--feet-radius", type=float, default=20.0)
    p.add_argument("--probe-radius", type=float, default=6.0)
    p.add_argument("--threshold",   type=float, default=-0.2)
    p.add_argument("--hole-r-min",  type=float, default=1.0)
    p.add_argument("--hole-r-max",  type=float, default=15.0)
    p.add_argument("--workers",     type=int,   default=4,   help="Parallel workers")
    p.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=False,
                   help="Save per-surface HTML (slow for 35 files)")
    p.add_argument("--plots",       action=argparse.BooleanOptionalAction, default=True,
                   help="Save per-surface PNG")
    return p.parse_args()


def process_one(ply_path, out_dir, feet_radius, probe_radius,
                threshold, hole_r_min, hole_r_max, save_plot, save_html):
    """Process a single PLY — runs in a worker process."""
    from virtual_comparator import make_static_plot, make_interactive_plot
    label = os.path.splitext(os.path.basename(ply_path))[0]
    t0    = time.time()

    mesh  = trimesh.load(ply_path, process=False)
    pts   = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces,    dtype=np.int32)
    kdtree = cKDTree(pts)

    holes, mastics = find_holes(faces, pts, r_min=hole_r_min, r_max=hole_r_max)
    if not holes:
        return {"label": label, "n_holes": 0, "n_mastics": len(mastics),
                "n_defects": 0, "worst_dev": None, "rows": [], "elapsed": time.time()-t0}

    results = []
    for i, hole in enumerate(holes):
        res = measure(hole["center"].copy(), pts, kdtree, feet_radius,
                      boundary_verts=hole["verts"], hole_radius=hole["radius_mm"])
        if res is None:
            continue
        res["hole_idx"]  = i
        res["hole_r"]    = hole["radius_mm"]
        res["feet_r"]    = feet_radius
        res["corrected"] = res["boundary_mean"]   # no zero offset in batch mode
        results.append(res)

    defects = [r for r in results if r["corrected"] <= threshold]

    if results and save_plot:
        out_png = os.path.join(out_dir, f"{label}_comparator.png")
        make_static_plot(pts, holes, mastics, results, 0.0, threshold, label, out_png)

    if results and save_html:
        out_html = os.path.join(out_dir, f"{label}_comparator.html")
        make_interactive_plot(pts, holes, mastics, results, 0.0, threshold, label, out_html)

    rows = [{
        "surface":        label,
        "hole_idx":       r["hole_idx"] + 1,
        "hole_r_mm":      round(r["hole_r"], 3),
        "bnd_mean_mm":    round(r["boundary_mean"], 4),
        "bnd_min_mm":     round(r["boundary_min"], 4),
        "bnd_p5_mm":      round(r["boundary_p5"], 4),
        "n_boundary_pts": r["n_probe"],
        "defect":         int(r["corrected"] <= threshold),
        "cx": round(r["center"][0], 2),
        "cy": round(r["center"][1], 2),
        "cz": round(r["center"][2], 2),
    } for r in results]

    return {
        "label":     label,
        "n_holes":   len(results),
        "n_mastics": len(mastics),
        "n_defects": len(defects),
        "worst_dev": min(r["corrected"] for r in results) if results else None,
        "rows":      rows,
        "elapsed":   round(time.time() - t0, 1),
    }


def summary_plot(summary_rows, threshold, out_path):
    labels    = [r["label"] for r in summary_rows]
    n_holes   = [r["n_holes"] for r in summary_rows]
    n_defects = [r["n_defects"] for r in summary_rows]
    worst     = [r["worst_dev"] if r["worst_dev"] is not None else 0.0
                 for r in summary_rows]

    x  = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(labels)*0.45), 10))
    fig.suptitle("Virtual Comparator — All surfaces summary", fontsize=13, fontweight="bold")

    # Bar: defects / total holes per surface
    ax = axes[0]
    ax.bar(x, n_holes,   label="total holes",   color="steelblue", alpha=0.6)
    ax.bar(x, n_defects, label="flagged defects", color="tomato",   alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("count"); ax.legend(fontsize=9)
    ax.set_title(f"Holes measured vs defects flagged  (threshold {threshold} mm)")

    # Line: worst pull-in per surface
    ax2 = axes[1]
    colors = ["tomato" if w <= threshold else "steelblue" for w in worst]
    ax2.bar(x, worst, color=colors, alpha=0.85)
    ax2.axhline(threshold, color="red", ls="--", lw=1.2, label=f"threshold {threshold}")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax2.set_ylabel("worst min deviation (mm)")
    ax2.set_title("Worst pull-in per surface")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ply_files = sorted(glob.glob(os.path.join(args.ply_dir, "*.ply")))
    print(f"\nFound {len(ply_files)} PLY files  —  workers={args.workers}")
    print(f"feet_radius={args.feet_radius} mm   probe_radius={args.probe_radius} mm   "
          f"threshold={args.threshold} mm\n")

    all_rows     = []
    summary_rows = []

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                process_one, f, args.out_dir,
                args.feet_radius, args.probe_radius, args.threshold,
                args.hole_r_min, args.hole_r_max,
                args.plots, args.interactive,
            ): f
            for f in ply_files
        }
        done = 0
        for fut in as_completed(futures):
            res  = fut.result()
            done += 1
            flag = f"  ⚠  {res['n_defects']} defects" if res["n_defects"] > 0 else ""
            mast = f"  ~{res.get('n_mastics',0)} mastic" if res.get('n_mastics',0) > 0 else ""
            print(f"  [{done:>2}/{len(ply_files)}] {res['label']:<25}  "
                  f"{res['n_holes']:>3} holes{mast}  {res['elapsed']:>5}s{flag}")
            all_rows.extend(res["rows"])
            summary_rows.append(res)

    print(f"\nTotal time: {time.time()-t_start:.1f}s")

    # Sort summary by label for consistent output
    summary_rows.sort(key=lambda r: r["label"])
    all_rows.sort(key=lambda r: (r["surface"], r["hole_idx"]))

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "all_measurements.csv")
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader(); w.writerows(all_rows)
        print(f"CSV  → {csv_path}  ({len(all_rows)} rows)")

    # ── Summary plot ──────────────────────────────────────────────────────────
    if args.plots:
        plot_path = os.path.join(args.out_dir, "summary.png")
        summary_plot(summary_rows, args.threshold, plot_path)
        print(f"Plot → {plot_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    total_holes   = sum(r["n_holes"]   for r in summary_rows)
    total_defects = sum(r["n_defects"] for r in summary_rows)
    surfaces_with_defects = sum(1 for r in summary_rows if r["n_defects"] > 0)
    print(f"\n{'='*55}")
    print(f"  Surfaces processed  : {len(summary_rows)}")
    print(f"  Total holes measured: {total_holes}")
    print(f"  Total defects found : {total_defects}  ({100*total_defects/max(total_holes,1):.1f}%)")
    print(f"  Surfaces with defects: {surfaces_with_defects}/{len(summary_rows)}")
    all_devs = [r["worst_dev"] for r in summary_rows if r["worst_dev"] is not None]
    if all_devs:
        worst_surface = min(summary_rows, key=lambda r: r["worst_dev"] or 0)
        print(f"  Global worst pull-in: {min(all_devs):.4f} mm  ({worst_surface['label']})")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()