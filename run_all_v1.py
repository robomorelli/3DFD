"""
Run virtual_comparator_v1 on all PLY files and aggregate results.

Output goes to comparator_output_v1/ — does NOT touch comparator_output/.

Usage:
  python run_all_v1.py
  python run_all_v1.py --workers 4 --threshold -0.2 --no-interactive
"""

import argparse
import csv
import glob
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ply-dir",          default="data/pc")
    p.add_argument("--out-dir",          default="comparator_output_v1")
    p.add_argument("--hole-r-min",        type=float, default=1.0)
    p.add_argument("--hole-r-max",        type=float, default=15.0)
    p.add_argument("--feet-radius",       type=float, default=13.7,
                   help="Circumradius of equilateral foot triangle (mm)")
    p.add_argument("--r-inner",           type=float, default=1.5)
    p.add_argument("--r-outer",           type=float, default=None,
                   help="Crown outer offset (mm). Default: auto from rivet spacing.")
    p.add_argument("--n-sectors",         type=int,   default=4)
    p.add_argument("--k-sectors",         type=int,   default=2)
    p.add_argument("--n-radial-bands",    type=int,   default=3)
    p.add_argument("--zero-probe-radius", type=float, default=4.0)
    p.add_argument("--foot-dist-max",     type=float, default=3.0)
    p.add_argument("--feet-radius-min",   type=float, default=6.0)
    p.add_argument("--green-thresh",         type=float, default=0.05)
    p.add_argument("--zero-nominal-thresh",  type=float, default=0.10)
    p.add_argument("--zero-from-edge-min",   type=float, default=13.0)
    p.add_argument("--zero-from-edge-max",   type=float, default=60.0)
    p.add_argument("--grid-res",         type=float, default=0.4)
    p.add_argument("--smooth-radius",    type=float, default=20.0)
    p.add_argument("--poly-degree",      type=int,   default=4)
    p.add_argument("--threshold",        type=float, default=-0.2)
    p.add_argument("--workers",          type=int,   default=4)
    p.add_argument("--no-zero", action="store_true", default=False,
                   help="Disable auto-zeroing: always use zero_offset=0")
    p.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=False,
                   help="Save per-surface HTML (slow for many files)")
    p.add_argument("--plots",       action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def process_one(ply_path, out_dir, args_dict):
    """Process a single PLY — runs in a worker process."""
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree

    from virtual_comparator import find_holes, build_mastic_boundary_tree
    from virtual_comparator_v1 import (
        compute_plane_at_point, find_zero_point, zero_reading_at,
        measure_crown_v1, make_static_plot, make_interactive_plot, save_csv,
        auto_r_outer,
    )
    from deviation_map import compute_deviation

    a   = dict(args_dict)   # local copy so we can modify r_outer per surface
    label = os.path.splitext(os.path.basename(ply_path))[0]
    t0    = time.time()

    mesh  = trimesh.load(ply_path, process=False)
    pts   = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces,    dtype=np.int32)
    kdtree = cKDTree(pts)

    holes, mastics = find_holes(faces, pts,
                                r_min=a["hole_r_min"], r_max=a["hole_r_max"])
    if not holes:
        return {"label": label, "n_holes": 0, "n_mastics": len(mastics),
                "n_defects": 0, "worst_dev": None, "rows": [],
                "elapsed": round(time.time() - t0, 1)}

    if a["r_outer"] is None:
        a["r_outer"] = auto_r_outer(holes)

    deviation, _, _, _ = compute_deviation(
        pts, a["grid_res"], a["smooth_radius"], a["poly_degree"],
    )

    all_centers    = [h["center"]    for h in holes]
    all_radii      = [h["radius_mm"] for h in holes]
    mastic_centers = [m["center"]    for m in mastics]
    # Use minor_r_mm (half-width) for centroid-based exclusion
    mastic_radii   = [m.get("minor_r_mm", m["radius_mm"]) for m in mastics]
    # Boundary KDTree for accurate exclusion along elongated mastics
    mastic_bound_tree, mastic_bound_r = build_mastic_boundary_tree(mastics, pts, buffer_r=3.0)

    results = []
    for i, hole in enumerate(holes):
        c = hole["center"].copy()
        r = hole["radius_mm"]

        other_c = [all_centers[j] for j in range(len(holes)) if j != i]
        other_r = [all_radii[j]   for j in range(len(holes)) if j != i]

        if a.get("no_zero"):
            zero_pt     = None
            zero_offset = 0.0
        else:
            zero_pt = find_zero_point(
                c, r, pts, kdtree, deviation,
                nominal_thresh=a["zero_nominal_thresh"],
                from_edge_min=a["zero_from_edge_min"],
                from_edge_max=a["zero_from_edge_max"],
                feet_radius=a["feet_radius"],
                other_centers=other_c, other_radii=other_r,
                mastic_centers=mastic_centers, mastic_radii=mastic_radii,
                mastic_bound_tree=mastic_bound_tree, mastic_bound_r=mastic_bound_r,
            )
            if zero_pt is None:
                zero_offset = 0.0
            else:
                zo = zero_reading_at(zero_pt, pts, kdtree,
                                     a["feet_radius"], a["zero_probe_radius"])
                zero_offset = zo if zo is not None else 0.0

        res = measure_crown_v1(
            c, r, pts, kdtree,
            feet_radius=a["feet_radius"],
            r_inner_mm=a["r_inner"],
            r_outer_mm=a["r_outer"],
            n_sectors=a["n_sectors"],
            k_worst=a["k_sectors"],
            n_radial_bands=a["n_radial_bands"],
            zero_offset=zero_offset,
            foot_dist_max=a["foot_dist_max"],
            feet_radius_min=a["feet_radius_min"],
            other_hole_centers=other_c,
            other_hole_radii=other_r,
            mastic_centers=mastic_centers,
            mastic_radii=mastic_radii,
            mastic_bound_tree=mastic_bound_tree,
            mastic_bound_r=mastic_bound_r,
        )
        if res is None:
            continue

        res["hole_idx"]    = i
        res["zero_center"] = zero_pt
        results.append(res)

    if not results:
        return {"label": label, "n_holes": 0, "n_mastics": len(mastics),
                "n_defects": 0, "worst_dev": None, "rows": [],
                "elapsed": round(time.time() - t0, 1)}

    defects = [r for r in results if r["k_worst_mean"] <= a["threshold"]]

    # Build a minimal args-like namespace for the plot functions
    import types
    plot_args = types.SimpleNamespace(**a)

    if a["plots"]:
        out_png = os.path.join(out_dir, f"{label}_comparator_v1.png")
        make_static_plot(pts, holes, mastics, results,
                         a["threshold"], label, out_png, plot_args)

    if a["interactive"]:
        out_html = os.path.join(out_dir, f"{label}_comparator_v1.html")
        make_interactive_plot(pts, holes, mastics, results,
                              a["threshold"], label, out_html, plot_args)

    csv_path = os.path.join(out_dir, f"{label}_comparator_v1.csv")
    save_csv(results, label, csv_path)

    rows = [{
        "surface":           label,
        "hole_idx":          r["hole_idx"] + 1,
        "hole_r_mm":         round(r["hole_r"], 3),
        "k_worst_mean_mm":   round(r["k_worst_mean"], 4),
        "coherent_mean_mm":  round(r["coherent_mean"], 4) if np.isfinite(r["coherent_mean"]) else "",
        "coherence":         round(r["coherence"], 3),
        "consensus_band":    r["consensus_band"],
        "crown_mean_mm":     round(r["crown_mean"], 4),
        "crown_p10_mm":      round(r["crown_p10"], 4),
        "crown_min_mm":      round(r["crown_min"], 4),
        "n_sectors_below":   r["n_sectors_below"],
        "n_sectors_pop":     r["n_sectors_pop"],
        "zero_offset_mm":    round(r["zero_offset"], 4),
        "has_zero":          int(r.get("zero_center") is not None),
        "foot_ok":           int(r["foot_ok"]),
        "actual_feet_r_mm":  round(r["feet_r"], 2),
        "foot_max_dist_mm":  round(r["foot_max_dist"], 3),
        "defect":            int(r["k_worst_mean"] <= a["threshold"] and r["foot_ok"]),
        "cx": round(r["center"][0], 2),
        "cy": round(r["center"][1], 2),
        "cz": round(r["center"][2], 2),
    } for r in results]

    return {
        "label":     label,
        "n_holes":   len(results),
        "n_mastics": len(mastics),
        "n_defects": len(defects),
        "worst_dev": min(r["k_worst_mean"] for r in results),
        "rows":      rows,
        "elapsed":   round(time.time() - t0, 1),
    }


def summary_plot(summary_rows, threshold, out_path):
    rows = sorted(summary_rows, key=lambda r: r["label"])
    labels    = [r["label"] for r in rows]
    n_holes   = [r["n_holes"]   for r in rows]
    n_defects = [r["n_defects"] for r in rows]
    worst     = [r["worst_dev"] if r["worst_dev"] is not None else 0.0
                 for r in rows]

    x   = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(labels) * 0.45), 10))
    fig.suptitle("Virtual Comparator v1 — All surfaces summary",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.bar(x, n_holes,   label="total holes",     color="steelblue", alpha=0.6)
    ax.bar(x, n_defects, label="flagged defects",  color="tomato",    alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("count"); ax.legend(fontsize=9)
    ax.set_title(f"Holes measured vs defects flagged  (threshold {threshold} mm)")

    ax2 = axes[1]
    colors = ["tomato" if w <= threshold else "steelblue" for w in worst]
    ax2.bar(x, worst, color=colors, alpha=0.85)
    ax2.axhline(threshold, color="red", ls="--", lw=1.2,
                label=f"threshold {threshold}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax2.set_ylabel("worst k_worst_mean (mm)")
    ax2.set_title("Worst pull-in per surface  (k_worst_mean metric)")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ply_files = sorted(glob.glob(os.path.join(args.ply_dir, "*.ply")))
    print(f"\nFound {len(ply_files)} PLY files   workers={args.workers}")
    r_outer_str = f"{args.r_outer:.1f}" if args.r_outer is not None else "auto"
    print(f"Feet R={args.feet_radius:.1f} mm   "
          f"Crown [{args.r_inner:.1f}..{r_outer_str}] mm   "
          f"K={args.k_sectors}/{args.n_sectors} sectors   "
          f"threshold={args.threshold} mm\n")

    args_dict = vars(args)

    all_rows, summary_rows = [], []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_one, f, args.out_dir, args_dict): f
            for f in ply_files
        }
        done = 0
        for fut in as_completed(futures):
            res   = fut.result()
            done += 1
            flag  = f"  ⚠  {res['n_defects']} defects" if res["n_defects"] > 0 else ""
            mast  = (f"  ~{res['n_mastics']} mastic"
                     if res.get("n_mastics", 0) > 0 else "")
            print(f"  [{done:>2}/{len(ply_files)}] {res['label']:<28}  "
                  f"{res['n_holes']:>3} holes{mast}  "
                  f"{res['elapsed']:>5}s{flag}")
            all_rows.extend(res["rows"])
            summary_rows.append(res)

    print(f"\nTotal time: {time.time() - t_start:.1f}s")

    summary_rows.sort(key=lambda r: r["label"])
    all_rows.sort(key=lambda r: (r["surface"], r["hole_idx"]))

    # ── Global CSV ────────────────────────────────────────────────────────────
    if all_rows:
        csv_path = os.path.join(args.out_dir, "all_measurements_v1.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)
        print(f"CSV  → {csv_path}  ({len(all_rows)} rows)")

    # ── Summary plot ──────────────────────────────────────────────────────────
    if args.plots:
        plot_path = os.path.join(args.out_dir, "summary_v1.png")
        summary_plot(summary_rows, args.threshold, plot_path)
        print(f"Plot → {plot_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    total_holes   = sum(r["n_holes"]   for r in summary_rows)
    total_defects = sum(r["n_defects"] for r in summary_rows)
    surf_defects  = sum(1 for r in summary_rows if r["n_defects"] > 0)
    all_devs = [r["worst_dev"] for r in summary_rows if r["worst_dev"] is not None]

    print(f"\n{'='*58}")
    print(f"  Surfaces processed   : {len(summary_rows)}")
    print(f"  Total holes measured : {total_holes}")
    print(f"  Total defects found  : {total_defects}  "
          f"({100 * total_defects / max(total_holes, 1):.1f}%)")
    print(f"  Surfaces with defects: {surf_defects}/{len(summary_rows)}")
    if all_devs:
        worst_s = min(summary_rows,
                      key=lambda r: r["worst_dev"] if r["worst_dev"] is not None else 0)
        print(f"  Global worst pull-in : {min(all_devs):.4f} mm  ({worst_s['label']})")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()