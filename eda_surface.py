"""
EDA script for 3DFD surface inspection.

Usage:
  python eda_surface.py                          # uses default PLY
  python eda_surface.py --ply data/pc/Surface8_clean.ply
  python eda_surface.py --no-interactive         # skip plotly HTML output
  python eda_surface.py --no-plots               # stats only, no figures
  python eda_surface.py --voxel-size 1.0         # coarser downsample for display
"""

import argparse
import os
import sys
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter, defaultdict

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="EDA for 3DFD PLY surfaces")
    p.add_argument("--ply", default="data/pc/Surface8_clean.ply",
                   help="Path to PLY file (default: %(default)s)")
    p.add_argument("--out-dir", default="eda_output",
                   help="Output directory for figures (default: %(default)s)")
    p.add_argument("--voxel-size", type=float, default=0.5,
                   help="Voxel size mm for point cloud downsampling in display (default: %(default)s)")
    p.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=True,
                   help="Save interactive plotly HTML (default: enabled)")
    p.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True,
                   help="Save static matplotlib PNG (default: enabled)")
    p.add_argument("--max-display-pts", type=int, default=200_000,
                   help="Max points shown in 3D scatter plot (default: %(default)s)")
    return p.parse_args()

# ── Helpers ───────────────────────────────────────────────────────────────────
def find_holes(faces, pts):
    """Return list of hole dicts {n_verts, center, radius_mm, verts}."""
    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], axis=0),
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
            if v in visited:
                continue
            visited.add(v); comp.append(v)
            stack.extend(adj[v] - visited)
        hpts = pts[comp]
        ctr  = hpts.mean(axis=0)
        r    = np.sqrt(((hpts[:, :2] - ctr[:2]) ** 2).sum(axis=1)).mean()
        holes.append({"n_verts": len(comp), "center": ctr, "radius_mm": r, "verts": comp})

    return sorted(holes, key=lambda x: -x["n_verts"])


def compute_face_normals(pts, faces):
    v0, v1, v2 = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return n / norm


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    label = os.path.splitext(os.path.basename(args.ply))[0]

    print(f"\n{'='*62}")
    print(f"  EDA  —  {label}")
    print(f"{'='*62}")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[1] Loading PLY …")
    mesh = trimesh.load(args.ply, process=False)
    is_mesh = isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0

    pts   = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces,    dtype=np.int32) if is_mesh else None

    print(f"    Type     : {'Mesh (vertices + faces)' if is_mesh else 'Point cloud only'}")
    print(f"    Vertices : {len(pts):,}")
    if is_mesh:
        print(f"    Faces    : {len(faces):,}")

    # ── 2. Geometric stats ────────────────────────────────────────────────────
    print("\n[2] Geometry …")
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    dims   = mx - mn
    print(f"    X  [{mn[0]:8.2f}, {mx[0]:8.2f}]  →  {dims[0]:.1f} mm")
    print(f"    Y  [{mn[1]:8.2f}, {mx[1]:8.2f}]  →  {dims[1]:.1f} mm")
    print(f"    Z  [{mn[2]:8.2f}, {mx[2]:8.2f}]  →  {dims[2]:.1f} mm")
    density = len(pts) / (dims[0] * dims[1])
    print(f"    ~density : {density:.2f} pts/mm²  (avg spacing ≈ {1/density**0.5:.2f} mm)")

    z = pts[:, 2]
    print(f"    Z mean   : {z.mean():.4f} mm   std: {z.std():.4f} mm")
    print(f"    Z p1/p99 : {np.percentile(z,1):.4f} / {np.percentile(z,99):.4f} mm")

    # ── 3. Face normals ───────────────────────────────────────────────────────
    if is_mesh:
        print("\n[3] Face normals …")
        fnormals = compute_face_normals(pts, faces)
        nz = np.abs(fnormals[:, 2])
        print(f"    |nz| mean     : {nz.mean():.4f}  (1 = flat)")
        print(f"    |nz| < 0.9    : {(nz < 0.9).sum():,} faces ({100*(nz<0.9).mean():.1f}%) ← curved / holes")

    # ── 4. Hole detection ─────────────────────────────────────────────────────
    holes = []
    if is_mesh:
        print("\n[4] Hole detection (boundary edges) …")
        holes = find_holes(faces, pts)
        all_bverts = np.unique([v for h in holes for v in h["verts"]]) if holes else np.array([])
        print(f"    Boundary edges groups : {len(holes)}")
        print(f"    Boundary vertices     : {len(all_bverts):,}")

        rivet_holes = [h for h in holes if 1.0 < h["radius_mm"] < 15.0]
        print(f"\n    Candidate rivet holes (r 1–15 mm): {len(rivet_holes)}")
        print(f"    {'#':>4}  {'verts':>6}  {'r(mm)':>7}  center (x, y, z)")
        for i, h in enumerate(rivet_holes[:30]):
            c = h["center"]
            print(f"    {i+1:>4}  {h['n_verts']:>6}  {h['radius_mm']:>7.2f}  "
                  f"({c[0]:7.1f}, {c[1]:7.1f}, {c[2]:7.1f})")

    # ── 5. Face area stats ────────────────────────────────────────────────────
    areas = None
    if is_mesh:
        print("\n[5] Face areas …")
        v0, v1, v2 = pts[faces[:,0]], pts[faces[:,1]], pts[faces[:,2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1-v0, v2-v0), axis=1)
        print(f"    mean   : {areas.mean():.4f} mm²")
        print(f"    median : {np.median(areas):.4f} mm²")
        print(f"    p99    : {np.percentile(areas,99):.4f} mm²")

    # ── 6. Static plots ───────────────────────────────────────────────────────
    if args.plots:
        print("\n[6] Static plots …")
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f"EDA  —  {label}", fontsize=13, fontweight="bold")
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        # 6a  Top-down depth map
        ax1 = fig.add_subplot(gs[0, :2])
        step = max(1, len(pts) // 300_000)
        sc = ax1.scatter(pts[::step, 0], pts[::step, 1], c=pts[::step, 2],
                         s=0.15, cmap="RdYlGn", rasterized=True)
        plt.colorbar(sc, ax=ax1, label="Z (mm)")
        if holes:
            hx = [h["center"][0] for h in holes]
            hy = [h["center"][1] for h in holes]
            ax1.scatter(hx, hy, c="blue", s=25, marker="x", zorder=5,
                        label=f"holes ({len(holes)})")
            ax1.legend(fontsize=8)
        ax1.set_title("Top-down depth map  (colour = Z)")
        ax1.set_xlabel("X (mm)"); ax1.set_ylabel("Y (mm)")
        ax1.set_aspect("equal")

        # 6b  Z histogram
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(z, bins=200, color="steelblue", edgecolor="none")
        ax2.axvline(z.mean(), color="red",    ls="--", lw=1.2, label=f"mean {z.mean():.3f}")
        ax2.axvline(np.percentile(z, 1),  color="orange", ls=":", lw=1, label="p1/p99")
        ax2.axvline(np.percentile(z, 99), color="orange", ls=":", lw=1)
        ax2.set_title("Z distribution"); ax2.set_xlabel("Z (mm)"); ax2.set_ylabel("count")
        ax2.legend(fontsize=7)

        # 6c  XZ profile slice
        ax3 = fig.add_subplot(gs[1, :])
        y_med = float(np.median(pts[:, 1]))
        tol   = dims[1] * 0.01
        mask  = np.abs(pts[:, 1] - y_med) < tol
        ax3.scatter(pts[mask, 0], pts[mask, 2], s=0.5, c="steelblue", rasterized=True)
        ax3.set_title(f"XZ profile slice  Y ≈ {y_med:.1f} mm  ±{tol:.1f} mm")
        ax3.set_xlabel("X (mm)"); ax3.set_ylabel("Z (mm)")

        # 6d  Face normal nz distribution
        ax4 = fig.add_subplot(gs[2, 0])
        if is_mesh:
            ax4.hist(fnormals[:, 2], bins=100, color="coral", edgecolor="none")
        ax4.set_title("Face normal Z component"); ax4.set_xlabel("nz"); ax4.set_ylabel("count")

        # 6e  Hole radius distribution
        ax5 = fig.add_subplot(gs[2, 1])
        if holes:
            radii = [h["radius_mm"] for h in holes]
            ax5.hist(radii, bins=40, color="mediumpurple", edgecolor="none")
            ax5.axvline(1.0,  color="green", ls="--", lw=1, label="1 mm")
            ax5.axvline(15.0, color="red",   ls="--", lw=1, label="15 mm")
            ax5.set_title(f"Hole radius distribution  (n={len(holes)})")
            ax5.set_xlabel("radius (mm)"); ax5.set_ylabel("count")
            ax5.legend(fontsize=7)
        else:
            ax5.text(0.5, 0.5, "No holes detected", ha="center", va="center",
                     transform=ax5.transAxes)
            ax5.set_title("Hole radius distribution")

        # 6f  Face area distribution
        ax6 = fig.add_subplot(gs[2, 2])
        if areas is not None:
            clip = np.percentile(areas, 99)
            ax6.hist(areas[areas < clip], bins=100, color="teal", edgecolor="none")
            ax6.set_title(f"Face area (p99={clip:.3f} mm²)")
            ax6.set_xlabel("mm²"); ax6.set_ylabel("count")
        else:
            ax6.text(0.5, 0.5, "No mesh faces", ha="center", va="center",
                     transform=ax6.transAxes)
            ax6.set_title("Face area distribution")

        out_png = os.path.join(args.out_dir, f"{label}_eda.png")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved → {out_png}")

    # ── 7. Interactive plotly ─────────────────────────────────────────────────
    if args.interactive:
        print("\n[7] Interactive plots (plotly) …")
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Downsample for display
        step  = max(1, len(pts) // args.max_display_pts)
        pts_d = pts[::step]

        # 7a  Point cloud coloured by Z
        fig_pc = go.Figure(data=[go.Scatter3d(
            x=pts_d[:, 0], y=pts_d[:, 1], z=pts_d[:, 2],
            mode="markers",
            marker=dict(size=1, color=pts_d[:, 2], colorscale="RdYlGn",
                        colorbar=dict(title="Z mm"), opacity=0.8),
            name="point cloud",
        )])
        # overlay hole centres
        if holes:
            hx = [h["center"][0] for h in holes]
            hy = [h["center"][1] for h in holes]
            hz = [h["center"][2] for h in holes]
            fig_pc.add_trace(go.Scatter3d(
                x=hx, y=hy, z=hz, mode="markers",
                marker=dict(size=4, color="blue", symbol="x"),
                name=f"holes ({len(holes)})",
            ))
        fig_pc.update_layout(
            title=f"{label} — Point Cloud (1/{step} pts, colour=Z)",
            scene=dict(xaxis_title="X mm", yaxis_title="Y mm", zaxis_title="Z mm",
                       aspectmode="data"),
            width=1200, height=800,
        )
        out_pc = os.path.join(args.out_dir, f"{label}_pointcloud.html")
        fig_pc.write_html(out_pc)
        print(f"    Saved → {out_pc}")

        # 7b  Mesh coloured by Z  (subsample faces for speed)
        if is_mesh:
            face_step = max(1, len(faces) // 500_000)
            f_sub = faces[::face_step]
            used  = np.unique(f_sub)
            remap = np.zeros(len(pts), dtype=np.int32)
            remap[used] = np.arange(len(used))
            pts_m  = pts[used]
            f_remap = remap[f_sub]

            fig_mesh = go.Figure(data=[go.Mesh3d(
                x=pts_m[:, 0], y=pts_m[:, 1], z=pts_m[:, 2],
                i=f_remap[:, 0], j=f_remap[:, 1], k=f_remap[:, 2],
                intensity=pts_m[:, 2], colorscale="RdYlGn",
                colorbar=dict(title="Z mm"),
                name="mesh",
            )])
            if holes:
                fig_mesh.add_trace(go.Scatter3d(
                    x=hx, y=hy, z=hz, mode="markers",
                    marker=dict(size=5, color="blue", symbol="x"),
                    name=f"holes ({len(holes)})",
                ))
            fig_mesh.update_layout(
                title=f"{label} — Mesh (1/{face_step} faces, colour=Z)",
                scene=dict(xaxis_title="X mm", yaxis_title="Y mm", zaxis_title="Z mm",
                           aspectmode="data"),
                width=1200, height=800,
            )
            out_mesh = os.path.join(args.out_dir, f"{label}_mesh.html")
            fig_mesh.write_html(out_mesh)
            print(f"    Saved → {out_mesh}")

    print(f"\n  Done. Output in: {args.out_dir}/\n")


if __name__ == "__main__":
    main()