"""
Run virtual_comparator_v1 (auto-zero only) on all PLY files.

Results go to comparator_output_autozero/.
Produces per-surface PNG + global summary PNG + global CSV.

Orange warning zone is split by near/far mastic proximity:
  far  from mastic: orange [warn_far_lo .. warn_far_hi]  red < -warn_far_hi
  near mastic:      orange [warn_near_lo.. warn_near_hi] red < -warn_near_hi

Usage:
  python run_all_autozero.py
  python run_all_autozero.py --workers 4
  python run_all_autozero.py --warn-far-hi 0.21 --warn-near-hi 0.14
"""

import argparse
import csv
import glob
import os
import time
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ply-dir",  default="data/pc")
    p.add_argument("--out-dir",  default="comparator_output_autozero")

    # Geometry
    p.add_argument("--hole-r-min",        type=float, default=1.0)
    p.add_argument("--hole-r-max",        type=float, default=15.0)
    p.add_argument("--feet-radius",       type=float, default=13.7)
    p.add_argument("--r-inner",           type=float, default=1.5)
    p.add_argument("--r-outer",           type=float, default=None,
                   help="Crown outer offset (mm). Default: auto from rivet spacing.")
    p.add_argument("--n-sectors",         type=int,   default=4)
    p.add_argument("--k-sectors",         type=int,   default=2)
    p.add_argument("--n-radial-bands",    type=int,   default=3)
    p.add_argument("--zero-probe-radius", type=float, default=4.0)
    p.add_argument("--foot-dist-max",     type=float, default=3.0)
    p.add_argument("--feet-radius-min",   type=float, default=6.0)
    p.add_argument("--min-sector-coverage", type=float, default=0.3,
                   help="Frazione minima di punti attesi per includere un settore "
                        "(0 = disabilitato, 1 = tutti i punti necessari).")

    # Auto-zero
    p.add_argument("--zero-nominal-thresh",  type=float, default=0.10)
    p.add_argument("--zero-from-edge-min",   type=float, default=13.0)
    p.add_argument("--zero-from-edge-max",   type=float, default=60.0)
    p.add_argument("--zero-search", choices=["free", "bounded"], default="free",
                   help="free: exclude only hole body of neighbours (default). "
                        "bounded: also exclude their crown zone (may reduce candidate count).")
    p.add_argument("--green-thresh",         type=float, default=0.05)

    # Deviation map
    p.add_argument("--grid-res",      type=float, default=0.4)
    p.add_argument("--smooth-radius", type=float, default=20.0)
    p.add_argument("--poly-degree",   type=int,   default=4)

    # Detection threshold (kept for CSV defect column)
    p.add_argument("--threshold", type=float, default=-0.2)

    # Colour zones
    p.add_argument("--warn-lo",     type=float, default=0.14,
                   help="Orange zone start |pull-in| (mm)")
    p.add_argument("--warn-hi",     type=float, default=0.21,
                   help="Red start |pull-in| (mm)")
    p.add_argument("--critical-hi", type=float, default=0.60,
                   help="Black (critical) start |pull-in| (mm)")

    # Measurement mode
    p.add_argument("--measure-mode", choices=["plane", "deviation", "local-poly"],
                   default="plane",
                   help="plane: 3-foot local plane (default). " 
                        "deviation: deviation-map residual. "
                        "local-poly: polynomial fit to nominal ring outside crown.")
    p.add_argument("--local-poly-fit-radius", type=float, default=30.0)
    p.add_argument("--local-poly-degree",     type=int,   default=2)
    p.add_argument("--local-poly-method",     choices=["exclude", "robust"], default="exclude")
    p.add_argument("--max-crown-dev-range",   type=float, default=None,
                   help="[plane] skip rivets with crown deviation range > this (mm)")

    # Output
    p.add_argument("--workers",  type=int,  default=4)
    p.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--plots",       action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def process_one(ply_path, out_dir, args_dict):
    """Process a single PLY with auto-zero — runs in a worker process."""
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree

    from virtual_comparator import find_holes, build_mastic_boundary_tree
    from virtual_comparator_v1 import (
        find_zero_point, zero_reading_at, deviation_zero_reading,
        measure_crown_v1, make_static_plot, make_interactive_plot, save_csv,
        make_zero_plot, auto_r_outer,
    )
    from deviation_map import compute_deviation

    a     = dict(args_dict)
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

    MASTIC_R_MAX = 80.0
    all_centers    = [h["center"]    for h in holes]
    all_radii      = [h["radius_mm"] for h in holes]
    mastic_centers = [m["center"]    for m in mastics if m["radius_mm"] < MASTIC_R_MAX]
    mastic_radii   = [m.get("minor_r_mm", m["radius_mm"])
                      for m in mastics if m["radius_mm"] < MASTIC_R_MAX]
    mastic_bound_tree, mastic_bound_r = build_mastic_boundary_tree(mastics, pts, buffer_r=3.0)

    results = []
    for i, hole in enumerate(holes):
        c = hole["center"].copy()
        r = hole["radius_mm"]

        other_c = [all_centers[j] for j in range(len(holes)) if j != i]
        other_r = [all_radii[j]   for j in range(len(holes)) if j != i]

        # Auto-zero (always enabled in this script)
        zero_pt = find_zero_point(
            c, r, pts, kdtree, deviation,
            nominal_thresh=a["zero_nominal_thresh"],
            from_edge_min=a["zero_from_edge_min"],
            from_edge_max=a["zero_from_edge_max"],
            feet_radius=a["feet_radius"],
            other_centers=other_c, other_radii=other_r,
            mastic_centers=mastic_centers, mastic_radii=mastic_radii,
            mastic_bound_tree=mastic_bound_tree, mastic_bound_r=mastic_bound_r,
            zero_search=a["zero_search"],
            crown_buffer=a["r_outer"],
        )
        if zero_pt is None:
            zero_offset = 0.0
        else:
            if a["measure_mode"] == "deviation":
                zo = deviation_zero_reading(zero_pt, pts, kdtree,
                                            deviation, a["zero_probe_radius"])
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
            min_sector_coverage=a["min_sector_coverage"],
            deviation_arr=deviation,
            measure_mode=a["measure_mode"],
            local_poly_fit_radius=a["local_poly_fit_radius"],
            local_poly_degree=a["local_poly_degree"],
            local_poly_method=a["local_poly_method"],
        )
        if res is None:
            continue

        import math
        max_dr = a.get("max_crown_dev_range")
        if (max_dr is not None
                and math.isfinite(res.get("crown_dev_range", float("nan")))
                and res["crown_dev_range"] > max_dr):
            continue

        res["hole_idx"]    = i
        res["zero_center"] = zero_pt
        results.append(res)

    if not results:
        return {"label": label, "n_holes": 0, "n_mastics": len(mastics),
                "n_defects": 0, "worst_dev": None, "rows": [],
                "elapsed": round(time.time() - t0, 1)}

    defects = [res for res in results if res["k_worst_mean"] <= a["threshold"]]

    plot_args = types.SimpleNamespace(**a)

    if a["plots"]:
        out_png = os.path.join(out_dir, f"{label}_comparator_v1.png")
        make_static_plot(pts, holes, mastics, results,
                         a["threshold"], label, out_png, plot_args)

        zero_dir = os.path.join(out_dir, "zero_points")
        os.makedirs(zero_dir, exist_ok=True)
        out_zero = os.path.join(zero_dir, f"{label}_zero_points.png")
        make_zero_plot(pts, mastics, results, label, out_zero, plot_args)

    if a["interactive"]:
        out_html = os.path.join(out_dir, f"{label}_comparator_v1.html")
        make_interactive_plot(pts, holes, mastics, results,
                              a["threshold"], label, out_html, plot_args)

    csv_path = os.path.join(out_dir, f"{label}_comparator_v1.csv")
    save_csv(results, label, csv_path)

    rows = [{
        "surface":           label,
        "hole_idx":          res["hole_idx"] + 1,
        "hole_r_mm":         round(res["hole_r"], 3),
        "k_worst_mean_mm":   round(res["k_worst_mean"], 4),
        "coherent_mean_mm":  round(res["coherent_mean"], 4) if np.isfinite(res["coherent_mean"]) else "",
        "coherence":         round(res["coherence"], 3),
        "crown_mean_mm":     round(res["crown_mean"], 4),
        "crown_p10_mm":      round(res["crown_p10"], 4),
        "crown_min_mm":      round(res["crown_min"], 4),
        "n_sectors_below":   res["n_sectors_below"],
        "n_sectors_pop":     res["n_sectors_pop"],
        "n_sectors_void":    res.get("n_sectors_void", 0),
        "zero_offset_mm":    round(res["zero_offset"], 4),
        "has_zero":          int(res.get("zero_center") is not None),
        "foot_ok":           int(res["foot_ok"]),
        "actual_feet_r_mm":  round(res["feet_r"], 2),
        "foot_max_dist_mm":  round(res["foot_max_dist"], 3),
        "defect":            int(res["k_worst_mean"] <= a["threshold"] and res["foot_ok"]),
        "cx": round(res["center"][0], 2),
        "cy": round(res["center"][1], 2),
        "cz": round(res["center"][2], 2),
    } for res in results]

    return {
        "label":     label,
        "n_holes":   len(results),
        "n_mastics": len(mastics),
        "n_defects": len(defects),
        "worst_dev": min(res["k_worst_mean"] for res in results),
        "rows":      rows,
        "elapsed":   round(time.time() - t0, 1),
    }


def summary_plot(summary_rows, threshold, warn_lo, warn_hi, out_path, critical_hi=0.60):
    rows   = sorted(summary_rows, key=lambda r: r["label"])
    labels = [r["label"] for r in rows]
    worst  = [r["worst_dev"] if r["worst_dev"] is not None else 0.0 for r in rows]

    def bar_col(w):
        if w <= -critical_hi:  return "black"
        if w <= -warn_hi:      return "red"
        if w <= -warn_lo:      return "darkorange"
        return "steelblue"

    x   = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(labels) * 0.45), 10))
    fig.suptitle(
        f"Virtual Comparator v1 (auto-zero) — All surfaces summary\n"
        f"orange [{-warn_hi:.2f},{-warn_lo:.2f}]  ·  red [{-critical_hi:.2f},{-warn_hi:.2f}]  ·  black <{-critical_hi:.2f} mm",
        fontsize=13, fontweight="bold",
    )

    ax = axes[0]
    n_holes   = [r["n_holes"]   for r in rows]
    n_defects = [r["n_defects"] for r in rows]
    ax.bar(x, n_holes,   label="total holes",    color="steelblue", alpha=0.6)
    ax.bar(x, n_defects, label="flagged defects", color="tomato",   alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("count"); ax.legend(fontsize=9)
    ax.set_title(f"Holes measured vs defects flagged  (threshold {threshold} mm)")

    ax2 = axes[1]
    ax2.bar(x, worst, color=[bar_col(w) for w in worst], alpha=0.85)
    ax2.axhline(-critical_hi, color="black",     ls="-",  lw=1.5,
                label=f"critical  {-critical_hi:.2f} mm")
    ax2.axhline(-warn_hi,     color="red",        ls="-",  lw=1.5,
                label=f"alarm     {-warn_hi:.2f} mm")
    ax2.axhline(-warn_lo,     color="darkorange", ls="-",  lw=1.2,
                label=f"warn      {-warn_lo:.2f} mm")
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
    print(f"Feet R={args.feet_radius:.1f} mm   Crown [{args.r_inner:.1f}..{r_outer_str}] mm   "
          f"K={args.k_sectors}/{args.n_sectors} sectors")
    print(f"Orange zone:  [{-args.warn_hi:.2f}, {-args.warn_lo:.2f}] mm   "
          f"Red: < {-args.warn_hi:.2f} mm\n")

    args_dict = vars(args)
    all_rows, summary_rows = [], []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, f, args.out_dir, args_dict): f
                   for f in ply_files}
        done = 0
        for fut in as_completed(futures):
            res   = fut.result()
            done += 1
            flag  = f"  ⚠  {res['n_defects']} defects" if res["n_defects"] > 0 else ""
            mast  = f"  ~{res['n_mastics']} mastic" if res.get("n_mastics", 0) > 0 else ""
            print(f"  [{done:>2}/{len(ply_files)}] {res['label']:<28}  "
                  f"{res['n_holes']:>3} holes{mast}  "
                  f"{res['elapsed']:>5}s{flag}")
            all_rows.extend(res["rows"])
            summary_rows.append(res)

    print(f"\nTotal time: {time.time() - t_start:.1f}s")

    summary_rows.sort(key=lambda r: r["label"])
    all_rows.sort(key=lambda r: (r["surface"], r["hole_idx"]))

    if all_rows:
        csv_path = os.path.join(args.out_dir, "all_measurements.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)
        print(f"CSV  → {csv_path}  ({len(all_rows)} rows)")

    if args.plots:
        plot_path = os.path.join(args.out_dir, "summary.png")
        summary_plot(summary_rows, args.threshold,
                     args.warn_lo, args.warn_hi, plot_path,
                     critical_hi=args.critical_hi)
        print(f"Plot → {plot_path}")

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
