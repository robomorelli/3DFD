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
    p.add_argument("--out-dir", default="comparator_output_v1",
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


# ── Hole / mastic detection ───────────────────────────────────────────────────
def _aspect_ratio(pts_xy):
    """PCA aspect ratio of a 2D point set: eigenvalue_1 / eigenvalue_2."""
    centered = pts_xy - pts_xy.mean(axis=0)
    if len(centered) < 3:
        return 1.0
    cov = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    return float(eigvals[0] / eigvals[1]) if eigvals[1] > 1e-9 else 999.0


def find_holes(faces, pts, r_min=3.0, r_max=15.0,
               mastic_aspect_min=4.0,
               mastic_n_verts_min=150):
    """
    Classify all boundary loops into:
      - rivets  : roughly circular, r_min ≤ radius ≤ r_max, aspect_ratio < mastic_aspect_min
      - mastic  : elongated rectangular strip (aspect_ratio ≥ mastic_aspect_min,
                  n_verts ≥ mastic_n_verts_min) — no upper bound on aspect_ratio
      - other   : everything else (noise, tiny holes, round panel perimeter)

    Returns (rivets, mastics) — each a list of dicts:
      {center, radius_mm, n_verts, verts, aspect_ratio}
    """
    edges = np.sort(
        np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [0,2]]], axis=0),
        axis=1,
    )
    counts = Counter(map(tuple, edges))
    boundary = [e for e, c in counts.items() if c == 1]
    if not boundary:
        return [], []

    adj = defaultdict(set)
    for a, b in boundary:
        adj[a].add(b); adj[b].add(a)

    visited, rivets, mastics = set(), [], []
    for start in adj:
        if start in visited:
            continue
        comp, stack = [], [start]
        while stack:
            v = stack.pop()
            if v in visited: continue
            visited.add(v); comp.append(v)
            stack.extend(adj[v] - visited)

        hpts    = pts[comp]
        ctr     = hpts.mean(axis=0)
        r       = np.sqrt(((hpts[:, :2] - ctr[:2]) ** 2).sum(axis=1)).mean()
        ar      = _aspect_ratio(hpts[:, :2])
        # Minor PCA half-axis = physical half-width of the strip (used for exclusion)
        centered2d = hpts[:, :2] - ctr[:2]
        eigvals = np.sort(np.linalg.eigvalsh(np.cov(centered2d.T)))
        minor_r = float(np.sqrt(max(eigvals[0], 0))) + 1.0  # half-width + 1mm buffer
        entry   = {"center": ctr, "radius_mm": float(r),
                   "minor_r_mm": minor_r,
                   "n_verts": len(comp), "verts": np.array(comp),
                   "aspect_ratio": float(ar)}

        if r_min <= r <= r_max and ar < mastic_aspect_min:
            rivets.append(entry)
        elif ar >= mastic_aspect_min and len(comp) >= mastic_n_verts_min:
            mastics.append(entry)
        # else: tiny noise loop → ignored

    rivets.sort(key=lambda x: -x["n_verts"])
    mastics.sort(key=lambda x: -x["n_verts"])
    return rivets, mastics


def build_mastic_boundary_tree(mastics, pts, buffer_r=3.0):
    """
    Build a 2D KDTree from the boundary vertices of all mastics.
    Returns (tree, buffer_r) or (None, buffer_r) if no mastics.

    Use tree.query(xy)[0] < buffer_r to test proximity to any mastic boundary.
    This correctly handles elongated mastics regardless of their orientation.
    """
    from scipy.spatial import cKDTree as _KDTree
    if not mastics:
        return None, buffer_r
    all_verts = np.concatenate([pts[m["verts"]][:, :2] for m in mastics], axis=0)
    return _KDTree(all_verts), buffer_r


def find_mastics_from_deviation(pts, deviation, grid_res=0.4,
                                proud_thresh=0.15,
                                mastic_aspect_min=4.0,
                                mastic_area_min_mm2=50.0):
    """
    Find elongated proud regions in the deviation map — mastic beads that
    appear as raised rectangular strips above the panel surface (NOT holes).

    Returns list of dicts compatible with find_holes() mastics:
      {center, radius_mm, n_verts, verts, aspect_ratio}
    where radius_mm is the max extent from centroid (used as exclusion radius).
    """
    from scipy.ndimage import label as nd_label

    x, y = pts[:, 0], pts[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    nx = int(np.ceil((x_max - x_min) / grid_res)) + 1
    ny = int(np.ceil((y_max - y_min) / grid_res)) + 1
    xi = x_min + np.arange(nx) * grid_res
    yi = y_min + np.arange(ny) * grid_res

    ix = np.clip(((x - x_min) / grid_res).astype(int), 0, nx - 1)
    iy = np.clip(((y - y_min) / grid_res).astype(int), 0, ny - 1)

    # Build mean deviation grid
    grid_sum   = np.zeros((ny, nx), dtype=np.float64)
    grid_count = np.zeros((ny, nx), dtype=np.int32)
    np.add.at(grid_sum,   (iy, ix), deviation)
    np.add.at(grid_count, (iy, ix), 1)
    dev_grid = np.where(grid_count > 0, grid_sum / grid_count, np.nan)

    # Threshold: cells clearly proud above reference
    proud_mask = (dev_grid >= proud_thresh) & (~np.isnan(dev_grid))

    # 8-connected components
    labeled, n_comp = nd_label(proud_mask, structure=np.ones((3, 3), dtype=int))

    mastics = []
    for comp_id in range(1, n_comp + 1):
        comp_mask = labeled == comp_id
        area_mm2  = comp_mask.sum() * grid_res ** 2
        if area_mm2 < mastic_area_min_mm2:
            continue

        rows, cols = np.where(comp_mask)
        pts2d = np.column_stack([xi[cols], yi[rows]])

        ar = _aspect_ratio(pts2d)
        if ar < mastic_aspect_min:
            continue

        # Vertex indices in this component
        verts = np.where(labeled[iy, ix] == comp_id)[0]

        ctr_xy = pts2d.mean(axis=0)
        ctr_z  = float(pts[verts, 2].mean()) if len(verts) > 0 else 0.0
        center = np.array([ctr_xy[0], ctr_xy[1], ctr_z])

        # Use max distance from centroid as exclusion radius
        r = float(np.sqrt(((pts2d - ctr_xy) ** 2).sum(axis=1)).max())

        mastics.append({
            "center":       center,
            "radius_mm":    r,
            "n_verts":      len(verts),
            "verts":        verts,
            "aspect_ratio": float(ar),
        })
    return mastics


# ── Core comparator measurement ───────────────────────────────────────────────
def measure(center, pts, kdtree, feet_radius, boundary_verts=None,
            probe_radius=6.0, hole_radius=0.0):
    """
    Place virtual comparator at `center`.

    feet_radius : distance from the HOLE BOUNDARY to each foot (not from center).
                  Effective radius from center = hole_radius + feet_radius.
    hole_radius : radius of the rivet hole (mm). 0 for zeroing on flat surface.

    Probe zone (in priority order):
      - boundary_verts: indices of the hole boundary ring (rivet measurement)
      - fallback: all vertices within probe_radius of center (zeroing on flat surface)

    Returns dict with:
      foot_pts        : (3,3) actual foot positions on mesh
      plane_n         : reference plane normal
      boundary_mean   : mean signed distance of probe points  ← primary metric
      boundary_min    : minimum signed distance (worst point)
      boundary_p5     : 5th-percentile signed distance
      n_probe         : number of probe points used
    Returns None if not enough vertices found.
    """
    effective_r = hole_radius + feet_radius
    n_vec, u_vec, v_vec = local_frame(center, pts, kdtree, effective_r * 1.5)

    # 3 feet at 120° intervals on the local tangent plane, offset from hole boundary
    angles = np.array([0.0, 2*np.pi/3, 4*np.pi/3])
    foot_targets = center + effective_r * (
        np.outer(np.cos(angles), u_vec) + np.outer(np.sin(angles), v_vec)
    )
    _, foot_idxs = kdtree.query(foot_targets)
    foot_pts = pts[foot_idxs]

    # Check feet are not degenerate (collinear)
    e1 = foot_pts[1] - foot_pts[0]
    e2 = foot_pts[2] - foot_pts[0]
    if np.linalg.norm(np.cross(e1, e2)) < 1e-6:
        return None
    plane_n, _ = fit_plane_3pts(foot_pts[0], foot_pts[1], foot_pts[2])
    if np.dot(plane_n, n_vec) < 0:
        plane_n = -plane_n

    # Probe points: boundary ring if available, else radius fallback (for zero point)
    if boundary_verts is not None and len(boundary_verts) > 0:
        probe_pts = pts[boundary_verts]
    else:
        probe_idxs = np.array(kdtree.query_ball_point(center, probe_radius), dtype=np.int32)
        if len(probe_idxs) == 0:
            return None
        probe_pts = pts[probe_idxs]

    dists = signed_distances(probe_pts, plane_n, foot_pts[0])

    return {
        "center":        center,
        "foot_pts":      foot_pts,
        "plane_n":       plane_n,
        "boundary_mean": float(dists.mean()),
        "boundary_min":  float(dists.min()),
        "boundary_p5":   float(np.percentile(dists, 5)),
        "n_probe":       len(probe_pts),
    }


# ── Zero-point lookup ─────────────────────────────────────────────────────────
def find_surface_point(xy, pts, kdtree):
    """Given (x, y), return the nearest mesh vertex as (x, y, z)."""
    tmp = np.array([[xy[0], xy[1], 0.0]])
    _, idx = kdtree.query(tmp)
    return pts[idx[0]].copy()


# ── Plotting helpers ──────────────────────────────────────────────────────────
def make_static_plot(pts, holes, mastics, results, zero_offset, threshold, label, out_path):
    step = max(1, len(pts) // 300_000)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Virtual Comparator  —  {label}  "
                 f"(feet R={results[0]['feet_r']:.0f} mm)",
                 fontsize=12, fontweight="bold")

    # ── Left: top-down depth map + measurement circles ───────────────────────
    ax = axes[0]
    ax.scatter(pts[::step, 0], pts[::step, 1], c=pts[::step, 2],
               s=0.1, cmap="gray", rasterized=True, alpha=0.5)

    vals = np.array([r["corrected"] for r in results])
    vabs = max(abs(vals.min()), abs(vals.max()), 0.05)
    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)
    cmap = cm.RdYlGn

    for r in results:
        c   = r["center"]
        v   = r["corrected"]
        col = cmap(norm(v))
        circle = plt.Circle((c[0], c[1]), r["hole_r"], color=col,
                             fill=True, alpha=0.8, linewidth=0)
        ax.add_patch(circle)
        feet_outer = plt.Circle((c[0], c[1]), r["feet_r"],
                                 color=col, fill=False, linewidth=0.6, alpha=0.4)
        ax.add_patch(feet_outer)
        # label every rivet with its mean deviation
        ax.annotate(f"{v:.2f}", (c[0], c[1]), fontsize=4,
                    ha="center", va="center", color="black")

    # overlay mastic zones as orange scatter
    for m in mastics:
        mverts = pts[m["verts"]]
        ax.scatter(mverts[:, 0], mverts[:, 1], s=1, c="orange",
                   alpha=0.6, linewidths=0, zorder=3)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="boundary mean deviation (mm)")
    ax.set_aspect("equal")
    ax.set_title("Boundary mean deviation per rivet  (green=proud · red=pull-in · orange=mastic)")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")

    # ── Right: histogram of all readings ─────────────────────────────────────
    ax2 = axes[1]
    ax2.hist(vals, bins=40, color="steelblue", edgecolor="white")
    ax2.axvline(threshold, color="red",   ls="--", lw=1.5, label=f"threshold {threshold} mm")
    ax2.axvline(0,         color="green", ls=":",  lw=1.0, label="zero")
    ax2.set_title(f"Distribution of boundary mean deviation  (n={len(vals)} rivets)")
    ax2.set_xlabel("boundary mean (mm)"); ax2.set_ylabel("count")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_interactive_plot(pts, holes, mastics, results, zero_offset, threshold, label, out_path):
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
        f"Hole #{r['hole_idx']+1}<br>"
        f"bnd_mean: {r['corrected']:.3f} mm<br>"
        f"bnd_min:  {r['boundary_min']:.3f} mm<br>"
        f"bnd_p5:   {r['boundary_p5']:.3f} mm<br>"
        f"n_boundary: {r['n_probe']}<br>"
        f"hole_r: {r['hole_r']:.2f} mm"
        for r in results
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

    # Value labels on all rivets
    fig.add_trace(go.Scatter3d(
        x=[r["center"][0] for r in results],
        y=[r["center"][1] for r in results],
        z=[r["center"][2] + 1 for r in results],
        mode="text",
        text=[f"{r['corrected']:.2f}" for r in results],
        textfont=dict(size=8, color="black"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Mastic zones in orange
    if mastics:
        mastic_pts = np.concatenate([pts[m["verts"]] for m in mastics], axis=0)
        step_m = max(1, len(mastic_pts) // 5000)
        fig.add_trace(go.Scatter3d(
            x=mastic_pts[::step_m, 0], y=mastic_pts[::step_m, 1], z=mastic_pts[::step_m, 2],
            mode="markers",
            marker=dict(size=2, color="orange", opacity=0.8),
            name=f"mastic ({len(mastics)})",
        ))

    fig.update_layout(
        title=f"{label} — Virtual Comparator  (feet R={results[0]['feet_r']:.0f} mm, "
              f"threshold {threshold} mm)",
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

    # ── Hole / mastic detection ───────────────────────────────────────────────
    print("[3] Detecting rivet holes and mastic …")
    holes, mastics = find_holes(faces, pts, r_min=args.hole_r_min, r_max=args.hole_r_max)
    print(f"    Rivet holes : {len(holes)}")
    print(f"    Mastic zones: {len(mastics)}")

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
        z_result = measure(z_center, pts, kdtree, args.feet_radius,
                           boundary_verts=None, probe_radius=args.probe_radius)
        if z_result is None:
            print("    WARNING: zero measurement failed (too few vertices). Offset = 0")
        else:
            zero_offset = z_result["boundary_mean"]
            print(f"    Zero reading (boundary_mean): {zero_offset:.4f} mm  → offset applied")
    else:
        print("\n[4] No zero-xy provided — raw deviations reported (no offset)")

    # ── Measure each rivet hole ───────────────────────────────────────────────
    print(f"\n[5] Measuring {len(holes)} rivet holes …")
    results = []
    skipped = 0
    for i, hole in enumerate(holes):
        res = measure(hole["center"].copy(), pts, kdtree, args.feet_radius,
                      boundary_verts=hole["verts"], hole_radius=hole["radius_mm"])
        if res is None:
            skipped += 1
            continue
        res["hole_idx"]  = i
        res["hole_r"]    = hole["radius_mm"]
        res["feet_r"]    = args.feet_radius
        res["corrected"] = res["boundary_mean"] - zero_offset
        results.append(res)

    print(f"    Measured: {len(results)}   Skipped (no vertices): {skipped}")

    # ── Report ────────────────────────────────────────────────────────────────
    defects = [r for r in results if r["corrected"] <= args.threshold]

    print(f"\n{'─'*62}")
    print(f"  RESULTS   (zero offset = {zero_offset:.4f} mm)  "
          f"metric = mean of boundary ring")
    print(f"{'─'*62}")
    print(f"  {'#':>4}  {'hole_r':>7}  {'bnd_mean':>9}  {'bnd_min':>9}  "
          f"{'bnd_p5':>8}  {'n_bnd':>6}  flag")
    for r in results:
        flag = "⚠ DEFECT" if r["corrected"] <= args.threshold else ""
        print(f"  {r['hole_idx']+1:>4}  {r['hole_r']:>7.2f}  "
              f"{r['corrected']:>9.4f}  "
              f"{r['boundary_min']-zero_offset:>9.4f}  "
              f"{r['boundary_p5']-zero_offset:>8.4f}  "
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
        make_static_plot(pts, holes, mastics, results, zero_offset, args.threshold, label, out_png)
        print(f"    Saved → {out_png}")

    if args.interactive:
        print(f"[7] Interactive plot …")
        out_html = os.path.join(args.out_dir, f"{label}_comparator.html")
        make_interactive_plot(pts, holes, mastics, results, zero_offset, args.threshold, label, out_html)
        print(f"    Saved → {out_html}")

    print(f"\n  Done. Output in: {args.out_dir}/\n")


if __name__ == "__main__":
    main()