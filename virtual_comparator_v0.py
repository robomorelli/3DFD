"""
Virtual Comparator for 3DFD surface defect detection.

Simulates a physical dial gauge (comparatore a 3 punte) with:
  - 3 feet on a circle of configurable radius  → define local reference plane
  - 1 central probe zone                       → measure surface deviation (mm)

Negative reading  = pull-in (surface depressed below reference plane)
Positive reading  = bump / proud rivet

Usage:
  python virtual_comparator.py --ply data/pc/Surface8_clean.ply
  python virtual_comparator.py --ply data/pc/Surface8_clean.ply \\
      --feet-radius 20 --probe-radius 6 --threshold -0.2
  python virtual_comparator.py --ply data/pc/Surface8_clean.ply \\
      --zero-xy "-100,50" --threshold -0.15
  python virtual_comparator.py --ply data/pc/Surface8_clean.ply \\
      --no-interactive
"""

import argparse
import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Virtual dial-gauge comparator on 3D surface mesh",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ply", default="data/pc/Surface8_clean.ply",
                   help="Input PLY file")
    p.add_argument("--out-dir", default="comparator_output",
                   help="Output directory")
    p.add_argument("--feet-radius", type=float, default=20.0,
                   help="Radius of foot circle (mm)")
    p.add_argument("--probe-radius", type=float, default=6.0,
                   help="Radius of central probe zone (mm)")
    p.add_argument("--threshold", type=float, default=-0.2,
                   help="Defect threshold (mm). Values below → defect flagged")
    p.add_argument("--zero-xy", type=str, default=None,
                   help="Nominal zeroing point 'x,y' in mm (optional)")
    p.add_argument("--hole-r-min", type=float, default=1.0,
                   help="Min hole radius to consider as rivet (mm)")
    p.add_argument("--hole-r-max", type=float, default=15.0,
                   help="Max hole radius to consider as rivet (mm)")
    p.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=True,
                   help="Save interactive plotly HTML")
    p.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True,
                   help="Save static matplotlib PNG")
    return p.parse_args()


# ── Geometry helpers ──────────────────────────────────────────────────────────
def local_frame(center, pts, kdtree, radius):
    """
    Estimate orthonormal frame (n, u, v) at center via PCA of local neighbourhood.
    n points roughly toward +Z (away from surface).
    """
    idxs = kdtree.query_ball_point(center, radius)
    if len(idxs) < 6:
        n = np.array([0.0, 0.0, 1.0])
    else:
        local = pts[idxs] - center
        _, _, Vt = np.linalg.svd(local, full_matrices=False)
        n = Vt[2]
        if n[2] < 0:
            n = -n
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref);  u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return n, u, v


def fit_plane_3pts(p0, p1, p2):
    """Return (normal, d) of plane through 3 points. normal·x = d."""
    n = np.cross(p1 - p0, p2 - p0)
    n /= np.linalg.norm(n)
    return n, float(np.dot(n, p0))


def signed_distances(points, plane_normal, plane_point):
    """Signed distances of points to a plane (positive = same side as normal)."""
    return (points - plane_point) @ plane_normal


# ── Hole detection ────────────────────────────────────────────────────────────
def find_holes(faces, pts, r_min=1.0, r_max=15.0):
    """Return list of hole dicts {center, radius_mm, n_verts, verts}."""
    edges = np.sort(
        np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [0,2]]], axis=0),
        axis=1,
    )
    counts = Counter(map(tuple, edges))
    boundary = [e for e, c in counts.items() if c == 1]
    if not boundary:
        return []

    adj = defaultdict(set)
    for a, b in boundary:
        adj[a].add(b); adj[b].add(a)

    visited, holes = set(), []
    for start in adj:
        if start in visited:
            continue
        comp, stack = [], [start]
        while stack:
            v = stack.pop()
            if v in visited: continue
            visited.add(v); comp.append(v)
            stack.extend(adj[v] - visited)
        hpts = pts[comp]
        ctr  = hpts.mean(axis=0)
        r    = np.sqrt(((hpts[:, :2] - ctr[:2]) ** 2).sum(axis=1)).mean()
        if r_min <= r <= r_max:
            holes.append({"center": ctr, "radius_mm": float(r),
                          "n_verts": len(comp), "verts": np.array(comp)})

    return sorted(holes, key=lambda x: -x["n_verts"])


# ── Core comparator measurement ───────────────────────────────────────────────
def measure(center, pts, kdtree, feet_radius, probe_radius):
    """
    Place virtual comparator at `center`.

    Returns dict with:
      foot_pts   : (3,3) actual foot positions on mesh
      plane_n    : plane normal
      plane_pt   : point on plane
      min_dist   : minimum signed distance in probe zone  (most depressed)
      mean_dist  : mean signed distance
      p5_dist    : 5th-percentile signed distance (robust min)
      n_probe    : number of probe-zone vertices used
    Returns None if not enough vertices found.
    """
    n_vec, u_vec, v_vec = local_frame(center, pts, kdtree, feet_radius * 1.5)

    # 3 feet at 120° intervals
    angles = np.array([0.0, 2*np.pi/3, 4*np.pi/3])
    foot_targets = center + feet_radius * (
        np.outer(np.cos(angles), u_vec) + np.outer(np.sin(angles), v_vec)
    )
    _, foot_idxs = kdtree.query(foot_targets)
    foot_pts = pts[foot_idxs]

    # Check feet are not degenerate (collinear)
    e1 = foot_pts[1] - foot_pts[0]
    e2 = foot_pts[2] - foot_pts[0]
    cross = np.cross(e1, e2)
    if np.linalg.norm(cross) < 1e-6:
        return None
    plane_n, _ = fit_plane_3pts(foot_pts[0], foot_pts[1], foot_pts[2])
    if np.dot(plane_n, n_vec) < 0:
        plane_n = -plane_n  # orient toward surface outside

    # Probe zone: all vertices within probe_radius of center
    probe_idxs = np.array(kdtree.query_ball_point(center, probe_radius), dtype=np.int32)
    if len(probe_idxs) == 0:
        return None

    probe_pts = pts[probe_idxs]
    dists     = signed_distances(probe_pts, plane_n, foot_pts[0])

    return {
        "center":    center,
        "foot_pts":  foot_pts,
        "plane_n":   plane_n,
        "plane_pt":  foot_pts[0],
        "min_dist":  float(dists.min()),
        "mean_dist": float(dists.mean()),
        "p5_dist":   float(np.percentile(dists, 5)),
        "n_probe":   len(probe_idxs),
    }


# ── Zero-point lookup ─────────────────────────────────────────────────────────
def find_surface_point(xy, pts, kdtree):
    """Given (x, y), return the nearest mesh vertex as (x, y, z)."""
    tmp = np.array([[xy[0], xy[1], 0.0]])
    _, idx = kdtree.query(tmp)
    return pts[idx[0]].copy()


# ── Plotting helpers ──────────────────────────────────────────────────────────
def make_static_plot(pts, holes, results, zero_offset, threshold, label, out_path):
    step = max(1, len(pts) // 300_000)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Virtual Comparator  —  {label}  "
                 f"(feet R={results[0]['feet_r']:.0f} mm, probe R={results[0]['probe_r']:.0f} mm)",
                 fontsize=12, fontweight="bold")

    # ── Left: top-down depth map + measurement circles ───────────────────────
    ax = axes[0]
    ax.scatter(pts[::step, 0], pts[::step, 1], c=pts[::step, 2],
               s=0.1, cmap="gray", rasterized=True, alpha=0.5)

    vals = np.array([r["corrected"] for r in results])
    vmin, vmax = min(vals.min(), threshold - 0.05), max(0.1, vals.max())
    norm  = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap  = cm.RdYlGn

    for r in results:
        c  = r["center"]
        v  = r["corrected"]
        col = cmap(norm(v))
        circle = plt.Circle((c[0], c[1]), r["hole_r"], color=col,
                             fill=True, alpha=0.7, linewidth=0)
        ax.add_patch(circle)
        feet_outer = plt.Circle((c[0], c[1]), r["feet_r"],
                                 color=col, fill=False, linewidth=0.8, alpha=0.5)
        ax.add_patch(feet_outer)
        if v <= threshold:
            ax.annotate(f"{v:.2f}", (c[0], c[1]), fontsize=5,
                        ha="center", va="center", color="black", fontweight="bold")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="deviation (mm)")
    ax.set_aspect("equal")
    ax.set_title(f"Measurements  (red = pull-in ≤ {threshold} mm)")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")

    # ── Right: histogram of corrected readings ────────────────────────────────
    ax2 = axes[1]
    ax2.hist(vals, bins=30, color="steelblue", edgecolor="white")
    ax2.axvline(threshold, color="red", ls="--", lw=1.5,
                label=f"threshold {threshold} mm")
    ax2.axvline(0, color="green", ls=":", lw=1)
    n_def = (vals <= threshold).sum()
    ax2.set_title(f"Distribution of min deviation  —  {n_def}/{len(vals)} flagged")
    ax2.set_xlabel("min deviation (mm)"); ax2.set_ylabel("count")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_interactive_plot(pts, results, zero_offset, threshold, label, out_path):
    import plotly.graph_objects as go

    step  = max(1, len(pts) // 30_000)   # keep HTML small: ~30K pts max
    pts_d = pts[::step]

    vals = np.array([r["corrected"] for r in results])

    fig = go.Figure()

    # Background point cloud
    fig.add_trace(go.Scatter3d(
        x=pts_d[:, 0], y=pts_d[:, 1], z=pts_d[:, 2],
        mode="markers",
        marker=dict(size=0.8, color=pts_d[:, 2], colorscale="Greys", opacity=0.3),
        name="surface",
        showlegend=False,
    ))

    # Measurement points coloured by deviation
    cx = [r["center"][0] for r in results]
    cy = [r["center"][1] for r in results]
    cz = [r["center"][2] for r in results]
    hover = [
        f"Hole #{i+1}<br>"
        f"min dev: {r['corrected']:.3f} mm<br>"
        f"mean dev: {r['mean_dist'] - (r['raw_zero'] if r.get('raw_zero') else 0):.3f} mm<br>"
        f"n_probe pts: {r['n_probe']}<br>"
        f"{'⚠ DEFECT' if r['corrected'] <= threshold else 'OK'}"
        for i, r in enumerate(results)
    ]
    fig.add_trace(go.Scatter3d(
        x=cx, y=cy, z=cz,
        mode="markers",
        marker=dict(
            size=6,
            color=vals,
            colorscale="RdYlGn",
            cmin=min(vals.min(), threshold - 0.05),
            cmax=max(0.1, vals.max()),
            colorbar=dict(title="dev (mm)"),
            line=dict(width=1, color="black"),
        ),
        text=hover,
        hoverinfo="text",
        name="measurements",
    ))

    # Defect markers
    def_results = [r for r in results if r["corrected"] <= threshold]
    if def_results:
        fig.add_trace(go.Scatter3d(
            x=[r["center"][0] for r in def_results],
            y=[r["center"][1] for r in def_results],
            z=[r["center"][2] + 2 for r in def_results],
            mode="markers+text",
            marker=dict(size=10, color="red", symbol="x"),
            text=[f"{r['corrected']:.2f}" for r in def_results],
            textposition="top center",
            name=f"defects ({len(def_results)})",
        ))

    fig.update_layout(
        title=f"{label} — Virtual Comparator  (feet R={results[0]['feet_r']:.0f} mm, "
              f"probe R={results[0]['probe_r']:.0f} mm, threshold {threshold} mm)",
        scene=dict(xaxis_title="X mm", yaxis_title="Y mm", zaxis_title="Z mm",
                   aspectmode="data"),
        width=1300, height=850,
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    label = os.path.splitext(os.path.basename(args.ply))[0]

    print(f"\n{'='*62}")
    print(f"  Virtual Comparator  —  {label}")
    print(f"  feet R={args.feet_radius} mm   probe R={args.probe_radius} mm   "
          f"threshold={args.threshold} mm")
    print(f"{'='*62}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1] Loading …")
    mesh = trimesh.load(args.ply, process=False)
    pts  = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    print(f"    {len(pts):,} vertices,  {len(faces):,} faces")

    # ── KD-tree ───────────────────────────────────────────────────────────────
    print("[2] Building KD-tree …")
    kdtree = cKDTree(pts)

    # ── Hole detection ────────────────────────────────────────────────────────
    print("[3] Detecting rivet holes …")
    holes = find_holes(faces, pts, r_min=args.hole_r_min, r_max=args.hole_r_max)
    print(f"    Found {len(holes)} candidate rivet holes")

    if len(holes) == 0:
        print("    No holes found — check --hole-r-min / --hole-r-max")
        return

    # ── Zero reading ──────────────────────────────────────────────────────────
    zero_offset = 0.0
    if args.zero_xy:
        print(f"\n[4] Zeroing at {args.zero_xy} …")
        xy = list(map(float, args.zero_xy.split(",")))
        z_center = find_surface_point(xy, pts, kdtree)
        print(f"    Surface point found: ({z_center[0]:.1f}, {z_center[1]:.1f}, {z_center[2]:.1f})")
        z_result = measure(z_center, pts, kdtree, args.feet_radius, args.probe_radius)
        if z_result is None:
            print("    WARNING: zero measurement failed (too few vertices). Offset = 0")
        else:
            zero_offset = z_result["min_dist"]
            print(f"    Zero reading (raw min_dist): {zero_offset:.4f} mm  → offset applied")
    else:
        print("\n[4] No zero-xy provided — raw deviations reported (no offset)")

    # ── Measure each rivet hole ───────────────────────────────────────────────
    print(f"\n[5] Measuring {len(holes)} rivet holes …")
    results = []
    skipped = 0
    for i, hole in enumerate(holes):
        center = hole["center"].copy()
        res = measure(center, pts, kdtree, args.feet_radius, args.probe_radius)
        if res is None:
            skipped += 1
            continue
        res["hole_idx"]  = i
        res["hole_r"]    = hole["radius_mm"]
        res["feet_r"]    = args.feet_radius
        res["probe_r"]   = args.probe_radius
        res["corrected"] = res["min_dist"] - zero_offset
        results.append(res)

    print(f"    Measured: {len(results)}   Skipped (no vertices): {skipped}")

    # ── Report ────────────────────────────────────────────────────────────────
    defects = [r for r in results if r["corrected"] <= args.threshold]

    print(f"\n{'─'*62}")
    print(f"  RESULTS   (zero offset = {zero_offset:.4f} mm)")
    print(f"{'─'*62}")
    print(f"  {'#':>4}  {'hole_r':>7}  {'min_dev':>9}  {'mean_dev':>9}  "
          f"{'p5_dev':>8}  {'n_pts':>6}  flag")
    for r in results:
        flag = "⚠ DEFECT" if r["corrected"] <= args.threshold else ""
        print(f"  {r['hole_idx']+1:>4}  {r['hole_r']:>7.2f}  "
              f"{r['corrected']:>9.4f}  "
              f"{r['mean_dist']-zero_offset:>9.4f}  "
              f"{r['p5_dist']-zero_offset:>8.4f}  "
              f"{r['n_probe']:>6}  {flag}")

    print(f"\n  Total holes measured : {len(results)}")
    print(f"  Defects flagged      : {len(defects)}  "
          f"({100*len(defects)/len(results):.1f}%)")
    if defects:
        worst = min(defects, key=lambda r: r["corrected"])
        c = worst["center"]
        print(f"  Worst pull-in        : {worst['corrected']:.4f} mm  "
              f"at ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})")

    if not results:
        print("  No results to plot.")
        return

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plots:
        print(f"\n[6] Static plot …")
        out_png = os.path.join(args.out_dir, f"{label}_comparator.png")
        make_static_plot(pts, holes, results, zero_offset, args.threshold, label, out_png)
        print(f"    Saved → {out_png}")

    if args.interactive:
        print(f"[7] Interactive plot …")
        out_html = os.path.join(args.out_dir, f"{label}_comparator.html")
        make_interactive_plot(pts, results, zero_offset, args.threshold, label, out_html)
        print(f"    Saved → {out_html}")

    print(f"\n  Done. Output in: {args.out_dir}/\n")


if __name__ == "__main__":
    main()