"""
Confronto tra aggregatori per la misura della corona:
  - MEAN : media per cella (settore × fascia) → valore peggiore per settore → media k peggiori  [attuale]
  - MIN  : minimo per settore (nessuna banda)  → media k peggiori
  - P10  : 10° percentile per settore          → media k peggiori

Output: comparison_output/aggregation_comparison.png

Usage:
  python compare_aggregation.py
  python compare_aggregation.py --workers 4 --ply-dir data/pc
"""

import argparse, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ply-dir",            default="data/pc")
    p.add_argument("--out-dir",            default="comparison_output")
    p.add_argument("--workers",            type=int,   default=4)
    p.add_argument("--hole-r-min",         type=float, default=1.0)
    p.add_argument("--hole-r-max",         type=float, default=15.0)
    p.add_argument("--feet-radius",        type=float, default=13.7)
    p.add_argument("--r-inner",            type=float, default=1.5)
    p.add_argument("--r-outer",            type=float, default=None)
    p.add_argument("--n-sectors",          type=int,   default=4)
    p.add_argument("--k-sectors",          type=int,   default=2)
    p.add_argument("--n-radial-bands",     type=int,   default=3)
    p.add_argument("--zero-probe-radius",  type=float, default=4.0)
    p.add_argument("--foot-dist-max",      type=float, default=3.0)
    p.add_argument("--feet-radius-min",    type=float, default=6.0)
    p.add_argument("--zero-nominal-thresh",type=float, default=0.10)
    p.add_argument("--zero-from-edge-min", type=float, default=13.0)
    p.add_argument("--zero-from-edge-max", type=float, default=60.0)
    p.add_argument("--green-thresh",       type=float, default=0.05)
    p.add_argument("--grid-res",           type=float, default=0.4)
    p.add_argument("--smooth-radius",      type=float, default=20.0)
    p.add_argument("--poly-degree",        type=int,   default=4)
    p.add_argument("--threshold",          type=float, default=-0.2)
    p.add_argument("--warn-lo",            type=float, default=0.14)
    p.add_argument("--warn-hi",            type=float, default=0.21)
    p.add_argument("--surf-plots",         action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out-surf-dir",       default=None,
                   help="Cartella per i plot per-superficie. Default: <out-dir>/mean_vs_min")
    return p.parse_args()


def _kworst(vals, k):
    """Mean of the k most-negative finite values in vals array."""
    v = vals[np.isfinite(vals)]
    if len(v) == 0:
        return float("nan")
    k_act = min(k, len(v))
    return float(np.sort(v)[:k_act].mean())


def crown_metrics(dists, sector_ids, band_ids, n_sectors, n_radial_bands, k_worst):
    """
    From per-point (dists, sector_ids, band_ids) compute three aggregations:
      mean_metric : current approach — mean per (s,b) cell, worst band per sector, k-worst mean
      min_metric  : min over all points per sector, k-worst mean
      p10_metric  : 10th percentile per sector, k-worst mean
    Also returns per-sector arrays for all three (for polar plots).
    """
    # ── MEAN approach ─────────────────────────────────────────────────────────
    grid = np.full((n_sectors, n_radial_bands), np.nan)
    for s in range(n_sectors):
        for b in range(n_radial_bands):
            m = (sector_ids == s) & (band_ids == b)
            if m.sum() >= 1:
                grid[s, b] = dists[m].mean()

    worst_mean = np.full(n_sectors, np.nan)
    for s in range(n_sectors):
        row = grid[s, :]
        if not np.all(np.isnan(row)):
            worst_mean[s] = np.nanmin(row)

    # ── MIN approach ──────────────────────────────────────────────────────────
    worst_min = np.full(n_sectors, np.nan)
    for s in range(n_sectors):
        m = sector_ids == s
        if m.sum() >= 1:
            worst_min[s] = dists[m].min()

    # ── P10 approach ──────────────────────────────────────────────────────────
    worst_p10 = np.full(n_sectors, np.nan)
    for s in range(n_sectors):
        m = sector_ids == s
        if m.sum() >= 1:
            worst_p10[s] = np.percentile(dists[m], 10)

    return (
        _kworst(worst_mean, k_worst),
        _kworst(worst_min,  k_worst),
        _kworst(worst_p10, k_worst),
        worst_mean, worst_min, worst_p10,
    )


def _rivet_color(val, warn_lo, warn_hi):
    if not np.isfinite(val):
        return "gray"
    if val <= -warn_hi:
        return "red"
    if val <= -warn_lo:
        return "darkorange"
    return "limegreen"


def make_surface_sidebyside(pts, holes, mastics, rows, label, a, out_path):
    """1×3 top-down map: MEAN | MIN | P10 per ogni rivetto."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    step = max(1, len(pts) // 300_000)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(
        f"Confronto aggregatori  —  {label}\n"
        f"MEAN (media per cella)  |  MIN (minimo per settore)  |  P10 (10° perc. per settore)\n"
        f"verde > {-a['warn_lo']:.2f} mm  ·  arancio [{-a['warn_hi']:.2f}, {-a['warn_lo']:.2f}]  ·  rosso < {-a['warn_hi']:.2f} mm",
        fontsize=10, fontweight="bold",
    )

    titles  = ["MEAN  (k_worst_mean)", "MIN  (k_worst_min)", "P10  (k_worst_p10)"]
    metrics = ["kw_mean", "kw_min", "kw_p10"]

    legend_elems = [
        mpatches.Patch(color="limegreen",  label=f"OK  (>{-a['warn_lo']:.2f} mm)"),
        mpatches.Patch(color="darkorange", label=f"Warn"),
        mpatches.Patch(color="red",        label=f"Difetto  (<{-a['warn_hi']:.2f} mm)"),
    ]

    for ax, title, metric in zip(axes, titles, metrics):
        ax.scatter(pts[::step, 0], pts[::step, 1],
                   c=pts[::step, 2], s=0.1, cmap="gray",
                   rasterized=True, alpha=0.35)

        for row in rows:
            c   = row["center"]
            val = row[metric]
            r   = row["hole_r"]
            col = _rivet_color(val, a["warn_lo"], a["warn_hi"])
            alp = 0.85 if row["foot_ok"] else 0.35

            ax.add_patch(plt.Circle((c[0], c[1]), r,
                                    color=col, fill=True, alpha=alp, linewidth=0))
            r_out = r + a["r_outer"]
            ax.add_patch(plt.Circle((c[0], c[1]), r_out,
                                    color=col, fill=False, linewidth=0.7,
                                    alpha=0.45, linestyle="-"))

            # Radial band rings
            r_in   = r + a["r_inner"]
            band_w = (r_out - r_in) / a["n_radial_bands"]
            for k in range(a["n_radial_bands"]):
                ax.add_patch(plt.Circle((c[0], c[1]), r_in + k * band_w,
                                        color="gray", fill=False, linewidth=0.35,
                                        alpha=0.3, linestyle="--"))

            # Sector dividers
            sector_w = 2 * np.pi / a["n_sectors"]
            for k in range(a["n_sectors"]):
                theta = -np.pi + k * sector_w
                ax.plot([c[0] + r_in * np.cos(theta), c[0] + r_out * np.cos(theta)],
                        [c[1] + r_in * np.sin(theta), c[1] + r_out * np.sin(theta)],
                        color="gray", linewidth=0.4, alpha=0.35)

            lbl = f"{val:.2f}" if np.isfinite(val) else "?"
            ax.annotate(lbl, (c[0], c[1]), fontsize=3.5,
                        ha="center", va="center", color="black")

        for m in mastics:
            mv = pts[m["verts"]]
            ax.scatter(mv[:, 0], mv[:, 1], s=0.8, c="orange",
                       alpha=0.5, linewidths=0, zorder=3)

        n_flag = sum(1 for row in rows
                     if np.isfinite(row[metric]) and row[metric] <= -a["warn_hi"])
        n_warn = sum(1 for row in rows
                     if np.isfinite(row[metric])
                     and -a["warn_hi"] < row[metric] <= -a["warn_lo"])
        ax.set_title(f"{title}  —  rossi={n_flag}  arancio={n_warn}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        ax.legend(handles=legend_elems, fontsize=6, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_one(ply_path, args_dict):
    import numpy as np, trimesh
    from scipy.spatial import cKDTree
    from virtual_comparator import find_holes, build_mastic_boundary_tree
    from virtual_comparator_v1 import (
        find_zero_point, zero_reading_at, find_valid_plane, auto_r_outer,
    )
    from virtual_comparator import signed_distances
    from deviation_map import compute_deviation

    a = args_dict
    label = os.path.splitext(os.path.basename(ply_path))[0]

    mesh   = trimesh.load(ply_path, process=False)
    pts    = np.asarray(mesh.vertices, dtype=np.float64)
    faces  = np.asarray(mesh.faces,   dtype=np.int32)
    kdtree = cKDTree(pts)

    holes, mastics = find_holes(faces, pts, r_min=a["hole_r_min"], r_max=a["hole_r_max"])
    if not holes:
        return []

    r_outer = a["r_outer"] if a["r_outer"] is not None else auto_r_outer(holes)

    deviation, _, _, _ = compute_deviation(
        pts, a["grid_res"], a["smooth_radius"], a["poly_degree"]
    )

    MASTIC_R_MAX   = 80.0
    all_centers    = [h["center"]    for h in holes]
    all_radii      = [h["radius_mm"] for h in holes]
    mastic_centers = [m["center"]    for m in mastics if m["radius_mm"] < MASTIC_R_MAX]
    mastic_radii   = [m.get("minor_r_mm", m["radius_mm"])
                      for m in mastics if m["radius_mm"] < MASTIC_R_MAX]
    mastic_bound_tree, mastic_bound_r = build_mastic_boundary_tree(mastics, pts, buffer_r=3.0)

    rows = []
    for i, hole in enumerate(holes):
        c = hole["center"].copy()
        r = hole["radius_mm"]
        other_c = [all_centers[j] for j in range(len(holes)) if j != i]
        other_r = [all_radii[j]   for j in range(len(holes)) if j != i]

        # Auto-zero
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
        zero_offset = 0.0
        if zero_pt is not None:
            zo = zero_reading_at(zero_pt, pts, kdtree,
                                 a["feet_radius"], a["zero_probe_radius"])
            if zo is not None:
                zero_offset = zo

        # Fit local plane
        plane_n, foot_pts, _, _, foot_ok = find_valid_plane(
            c, pts, kdtree, a["feet_radius"], a["foot_dist_max"], a["feet_radius_min"],
            hole_centers=other_c, hole_radii=other_r,
            mastic_centers=mastic_centers, mastic_radii=mastic_radii,
            mastic_bound_tree=mastic_bound_tree, mastic_bound_r=mastic_bound_r,
        )
        if plane_n is None:
            continue

        # Crown points
        r_in  = r + a["r_inner"]
        r_out = r + r_outer
        cand  = np.array(kdtree.query_ball_point(c, r_out), dtype=np.int32)
        if len(cand) == 0:
            continue
        d2d       = np.linalg.norm(pts[cand, :2] - c[:2], axis=1)
        mask      = (d2d >= r_in) & (d2d <= r_out)
        crown_idx = cand[mask]
        d2d_crown = d2d[mask]
        crown_pts = pts[crown_idx]

        # Exclude other holes and mastics
        valid = np.ones(len(crown_pts), dtype=bool)
        for hc, hr in zip(other_c, other_r):
            valid &= np.linalg.norm(crown_pts[:, :2] - np.asarray(hc[:2]), axis=1) > hr
        for mc, mr in zip(mastic_centers, mastic_radii):
            valid &= np.linalg.norm(crown_pts[:, :2] - np.asarray(mc[:2]), axis=1) > mr
        if mastic_bound_tree is not None:
            db, _ = mastic_bound_tree.query(crown_pts[:, :2])
            valid &= db >= mastic_bound_r
        crown_pts  = crown_pts[valid]
        d2d_crown  = d2d_crown[valid]

        n_min = max(a["n_sectors"] * a["n_radial_bands"], 5)
        if len(crown_pts) < n_min:
            continue

        dists = signed_distances(crown_pts, plane_n, foot_pts[0]) - zero_offset

        dx = crown_pts[:, 0] - c[0]
        dy = crown_pts[:, 1] - c[1]
        sector_w  = 2 * np.pi / a["n_sectors"]
        sector_ids = (np.floor((np.arctan2(dy, dx) + np.pi) / sector_w).astype(int)
                      % a["n_sectors"])
        band_w  = (r_out - r_in) / a["n_radial_bands"]
        band_ids = np.clip(
            ((d2d_crown - r_in) / band_w).astype(int), 0, a["n_radial_bands"] - 1
        )

        kw_mean, kw_min, kw_p10, sv_mean, sv_min, sv_p10 = crown_metrics(
            dists, sector_ids, band_ids,
            a["n_sectors"], a["n_radial_bands"], a["k_sectors"],
        )

        rows.append({
            "label":    label,
            "hole_idx": i + 1,
            "center":   c.tolist(),
            "hole_r":   round(r, 3),
            "n_crown":  len(crown_pts),
            "foot_ok":  foot_ok,
            "kw_mean":  kw_mean,
            "kw_min":   kw_min,
            "kw_p10":   kw_p10,
            "sv_mean":  sv_mean.tolist(),
            "sv_min":   sv_min.tolist(),
            "sv_p10":   sv_p10.tolist(),
        })

    if rows and a.get("surf_plots", True):
        surf_dir = a.get("out_surf_dir") or os.path.join(a["out_dir"], "mean_vs_min")
        os.makedirs(surf_dir, exist_ok=True)
        out_png = os.path.join(surf_dir, f"{label}_mean_vs_min.png")
        # Convert center back to numpy arrays for plotting
        plot_rows = [{**row, "center": np.array(row["center"])} for row in rows]
        r_outer_val = a["r_outer"] if a["r_outer"] is not None else auto_r_outer(holes)
        plot_args = {**a, "r_outer": r_outer_val}
        make_surface_sidebyside(pts, holes, mastics, plot_rows, label, plot_args, out_png)

    return rows


def make_plot(all_rows, threshold, out_path):
    kwm  = np.array([r["kw_mean"] for r in all_rows])
    kwn  = np.array([r["kw_min"]  for r in all_rows])
    kwp  = np.array([r["kw_p10"]  for r in all_rows])
    nc   = np.array([r["n_crown"] for r in all_rows])

    thr = threshold
    flag_mean = kwm <= thr
    flag_min  = kwn <= thr
    flag_p10  = kwp <= thr

    # agreement categories for scatter colour
    def cat(fm, fn):
        c = np.empty(len(fm), dtype=object)
        c[:] = "grey"
        c[ fm &  fn] = "red"
        c[~fm & ~fn] = "steelblue"
        c[ fm & ~fn] = "darkorange"
        c[~fm &  fn] = "purple"
        return c

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Confronto aggregatori: MEAN vs MIN vs P10 per settore\n"
        f"threshold = {thr} mm  |  n rivetti = {len(kwm)}",
        fontsize=13, fontweight="bold",
    )

    lim = (min(kwm.min(), kwn.min(), kwp.min()) - 0.05,
           max(kwm.max(), kwn.max(), kwp.max()) + 0.05)

    def scatter(ax, x, y, cols, xlabel, ylabel, title):
        ax.scatter(x, y, c=cols, s=18, alpha=0.7, linewidths=0)
        lo, hi = min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")
        ax.axvline(thr, color="red", ls=":", lw=1.0)
        ax.axhline(thr, color="red", ls=":", lw=1.0)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=8)
        from matplotlib.patches import Patch
        handles = [
            Patch(color="red",        label="entrambi flaggati"),
            Patch(color="steelblue",  label="entrambi OK"),
            Patch(color="darkorange", label=f"solo {xlabel.split()[0]}"),
            Patch(color="purple",     label=f"solo {ylabel.split()[0]}"),
        ]
        ax.legend(handles=handles, fontsize=7, loc="upper left")

    scatter(axes[0, 0], kwm, kwn, cat(flag_mean, flag_min),
            "MEAN (k_worst_mean)", "MIN (k_worst_min)", "MEAN vs MIN")
    scatter(axes[0, 1], kwm, kwp, cat(flag_mean, flag_p10),
            "MEAN (k_worst_mean)", "P10 (k_worst_p10)", "MEAN vs P10")
    scatter(axes[0, 2], kwn, kwp, cat(flag_min, flag_p10),
            "MIN (k_worst_min)", "P10 (k_worst_p10)", "MIN vs P10")

    # ── Differenze relative ───────────────────────────────────────────────────
    d_min_mean = kwn - kwm   # negativo = min più estremo di mean
    d_p10_mean = kwp - kwm

    ax = axes[1, 0]
    ax.hist(d_min_mean, bins=50, color="purple",     alpha=0.7, label="MIN − MEAN")
    ax.hist(d_p10_mean, bins=50, color="darkorange", alpha=0.7, label="P10 − MEAN")
    ax.axvline(0, color="k", ls="--", lw=0.9)
    ax.set_xlabel("differenza (mm)")
    ax.set_ylabel("conteggio rivetti")
    ax.set_title("Distribuzione delle differenze rispetto a MEAN")
    ax.legend(fontsize=8)

    med_min = np.nanmedian(d_min_mean)
    med_p10 = np.nanmedian(d_p10_mean)
    ax.text(0.97, 0.97,
            f"mediana MIN−MEAN: {med_min:+.3f} mm\nmediana P10−MEAN: {med_p10:+.3f} mm",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # ── Concordanza flagging ──────────────────────────────────────────────────
    ax = axes[1, 1]
    categories = {
        "MEAN ∩ MIN ∩ P10":        ( flag_mean &  flag_min &  flag_p10).sum(),
        "MEAN ∩ P10, not MIN":     ( flag_mean & ~flag_min &  flag_p10).sum(),
        "MEAN ∩ MIN, not P10":     ( flag_mean &  flag_min & ~flag_p10).sum(),
        "solo MEAN":               ( flag_mean & ~flag_min & ~flag_p10).sum(),
        "MIN ∩ P10, not MEAN":     (~flag_mean &  flag_min &  flag_p10).sum(),
        "solo MIN":                (~flag_mean &  flag_min & ~flag_p10).sum(),
        "solo P10":                (~flag_mean & ~flag_min &  flag_p10).sum(),
        "nessuno":                 (~flag_mean & ~flag_min & ~flag_p10).sum(),
    }
    colors_cat = ["red","orangered","tomato","darkorange","mediumpurple","purple","orchid","steelblue"]
    bars = ax.barh(list(categories.keys()), list(categories.values()),
                   color=colors_cat, alpha=0.85)
    for bar, v in zip(bars, categories.values()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=8)
    ax.set_xlabel("n rivetti")
    ax.set_title("Concordanza flagging tra i tre metodi")
    ax.set_xlim(0, max(categories.values()) * 1.15)

    # ── Bias atteso del minimo per rumore gaussiano ────────────────────────────
    ax = axes[1, 2]
    n_vals = np.array([r["n_crown"] / max(r["hole_r"], 1) for r in all_rows])
    ns = np.linspace(10, nc.max(), 200)
    sigma_noise = 0.015  # mm — stima rumore superficiale tipico
    bias_min = -sigma_noise * np.sqrt(2 * np.log(np.maximum(ns, 2)))
    bias_p10 = -sigma_noise * np.sqrt(2 * np.log(np.maximum(ns / 10, 1)))

    ax.plot(ns, bias_min, color="purple",     lw=2, label=f"bias MIN  (σ={sigma_noise} mm)")
    ax.plot(ns, bias_p10, color="darkorange", lw=2, label=f"bias P10 (σ={sigma_noise} mm)")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("n punti nel settore")
    ax.set_ylabel("bias atteso (mm)")
    ax.set_title("Bias atteso da rumore gaussiano\n(E[min] = μ − σ·√(2·ln n))")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.15, 0.02)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot → {out_path}")


def main():
    import glob
    args = parse_args()
    if args.out_surf_dir is None:
        args.out_surf_dir = os.path.join(args.out_dir, "mean_vs_min")
    args_dict = vars(args)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.surf_plots:
        os.makedirs(args.out_surf_dir, exist_ok=True)

    ply_files = sorted(glob.glob(os.path.join(args.ply_dir, "*.ply")))
    print(f"\nFound {len(ply_files)} PLY files   workers={args.workers}")

    all_rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, f, args_dict): f for f in ply_files}
        done = 0
        for fut in as_completed(futures):
            rows  = fut.result()
            done += 1
            label = os.path.splitext(os.path.basename(futures[fut]))[0]
            print(f"  [{done:>2}/{len(ply_files)}] {label:<28}  {len(rows):>3} holes")
            all_rows.extend(rows)

    print(f"\nTot. rivetti: {len(all_rows)}   ({time.time()-t0:.1f}s)")

    out_path = os.path.join(args.out_dir, "aggregation_comparison.png")
    make_plot(all_rows, args.threshold, out_path)

    # Stampa statistiche sintetiche
    kwm = np.array([r["kw_mean"] for r in all_rows])
    kwn = np.array([r["kw_min"]  for r in all_rows])
    kwp = np.array([r["kw_p10"]  for r in all_rows])
    thr = args.threshold
    print(f"\n{'Metodo':<8}  {'flagged':>7}  {'mediana':>8}  {'min':>8}")
    for name, arr in [("MEAN", kwm), ("MIN", kwn), ("P10", kwp)]:
        print(f"{name:<8}  {(arr<=thr).sum():>7}  {np.nanmedian(arr):>8.4f}  {np.nanmin(arr):>8.4f}")

    d = kwn - kwm
    print(f"\nbias mediano MIN−MEAN : {np.nanmedian(d):+.4f} mm  "
          f"(range [{d.min():.3f}, {d.max():.3f}])")
    d2 = kwp - kwm
    print(f"bias mediano P10−MEAN : {np.nanmedian(d2):+.4f} mm  "
          f"(range [{d2.min():.3f}, {d2.max():.3f}])")


if __name__ == "__main__":
    main()