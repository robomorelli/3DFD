"""
Local deviation map for 3DFD surfaces.

For each vertex computes: deviation = Z - Z_reference
where Z_reference is a Gaussian-smoothed version of the surface Z grid.

Pull-ins appear as blue halos around rivet holes.
Dents appear as blue patches anywhere on the surface.
Proud features appear green/yellow.

Usage:
  python deviation_map.py
  python deviation_map.py --ply data/pc/Surface8_clean.ply
  python deviation_map.py --ply data/pc/Surface8_clean.ply --smooth-radius 20 --grid-res 0.4
  python deviation_map.py --ply data/pc/Surface8_clean.ply --no-interactive
"""

import argparse
import os
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Local deviation map — colours each vertex by departure from smooth reference surface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ply",           default="data/pc/Surface8_clean.ply")
    p.add_argument("--out-dir",       default="deviation_output")
    p.add_argument("--grid-res",      type=float, default=0.4,
                   help="2D grid resolution (mm)")
    p.add_argument("--smooth-radius", type=float, default=20.0,
                   help="Gaussian smoothing radius applied ON TOP of polynomial fit (mm)")
    p.add_argument("--poly-degree",   type=int,   default=4,
                   help="Degree of polynomial fit for global shape removal (0 = disabled)")
    p.add_argument("--clip-mm",       type=float, default=1.5,
                   help="Clip deviation colourscale to ±clip_mm")
    p.add_argument("--interactive",   action=argparse.BooleanOptionalAction, default=True,
                   help="Save interactive plotly HTML")
    p.add_argument("--plots",         action=argparse.BooleanOptionalAction, default=True,
                   help="Save static matplotlib PNG")
    p.add_argument("--max-html-pts",  type=int, default=300_000,
                   help="Max points in HTML scatter (downsampled for file size)")
    return p.parse_args()


# ── Core: deviation map ───────────────────────────────────────────────────────
def poly_design_matrix(x, y, degree):
    """Build polynomial feature matrix for z = f(x,y) up to given degree."""
    cols = []
    for d in range(degree + 1):
        for k in range(d + 1):
            cols.append((x ** (d - k)) * (y ** k))
    return np.column_stack(cols)


def compute_deviation(pts, grid_res, smooth_radius, poly_degree=4):
    """
    Returns deviation array (same length as pts).
    Positive = surface proud above reference.
    Negative = pull-in / dent below reference.

    Pipeline:
      1. Fit global polynomial z = f(x,y) of degree poly_degree → remove panel curvature
      2. Project residuals onto 2D grid
      3. Gaussian smooth residual grid (radius smooth_radius) → residual low-freq
      4. Final deviation = poly_residual - gaussian_residual_lowfreq
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # ── Step 1: polynomial fit on a subsample ────────────────────────────────
    # Normalise for numerical stability
    x_mu, x_s = x.mean(), x.std()
    y_mu, y_s = y.mean(), y.std()
    xn = (x - x_mu) / x_s
    yn = (y - y_mu) / y_s

    step_fit = max(1, len(pts) // 80_000)
    A_sub = poly_design_matrix(xn[::step_fit], yn[::step_fit], poly_degree)
    coeffs, _, _, _ = np.linalg.lstsq(A_sub, z[::step_fit], rcond=None)

    A_full = poly_design_matrix(xn, yn, poly_degree)
    z_poly = A_full @ coeffs
    residual = z - z_poly   # large-scale curvature removed

    # ── Step 2: project residuals onto 2D grid ───────────────────────────────
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    nx = int(np.ceil((x_max - x_min) / grid_res)) + 1
    ny = int(np.ceil((y_max - y_min) / grid_res)) + 1
    xi = x_min + np.arange(nx) * grid_res
    yi = y_min + np.arange(ny) * grid_res

    ix = np.clip(((x - x_min) / grid_res).astype(int), 0, nx - 1)
    iy = np.clip(((y - y_min) / grid_res).astype(int), 0, ny - 1)

    grid_sum   = np.zeros((ny, nx), dtype=np.float64)
    grid_count = np.zeros((ny, nx), dtype=np.int32)
    np.add.at(grid_sum,   (iy, ix), residual)
    np.add.at(grid_count, (iy, ix), 1)

    grid_r = np.where(grid_count > 0, grid_sum / grid_count, np.nan)

    # ── Step 3: fill NaN holes and Gaussian smooth ───────────────────────────
    from scipy.ndimage import distance_transform_edt
    nan_mask = np.isnan(grid_r)
    if nan_mask.any():
        _, nearest_idx = distance_transform_edt(nan_mask, return_indices=True)
        grid_r = grid_r[nearest_idx[0], nearest_idx[1]]

    sigma_px  = smooth_radius / grid_res
    ref_grid  = gaussian_filter(grid_r, sigma=sigma_px)

    # ── Step 4: interpolate and subtract low-freq residual ───────────────────
    interp = RegularGridInterpolator(
        (yi, xi), ref_grid,
        method="linear", bounds_error=False, fill_value=None,
    )
    ref_r = interp(np.column_stack([y, x]))
    deviation = residual - ref_r

    return deviation, ref_grid, xi, yi


# ── Static plot ───────────────────────────────────────────────────────────────
def make_static_plot(pts, deviation, smooth_radius, clip_mm, label, out_path):
    x, y = pts[:, 0], pts[:, 1]

    norm = mcolors.TwoSlopeNorm(vmin=-clip_mm, vcenter=0.0, vmax=clip_mm)
    cmap = cm.RdYlGn

    step = max(1, len(pts) // 500_000)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        f"Local deviation map  —  {label}\n"
        f"(smooth radius = {smooth_radius} mm  |  green = nominal · blue = pull-in · red = proud)",
        fontsize=11, fontweight="bold",
    )

    # ── Left: top-down map coloured by deviation ──────────────────────────────
    ax = axes[0]
    sc = ax.scatter(x[::step], y[::step], c=deviation[::step],
                    s=0.15, cmap=cmap, norm=norm, rasterized=True)
    plt.colorbar(sc, ax=ax, label="deviation (mm)")
    ax.set_aspect("equal")
    ax.set_title("Deviation from smooth reference surface")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")

    # ── Right: deviation histogram ────────────────────────────────────────────
    ax2 = axes[1]
    clipped = np.clip(deviation, -clip_mm * 2, clip_mm * 2)
    ax2.hist(clipped, bins=200, color="steelblue", edgecolor="none")
    ax2.axvline(0,        color="green", ls=":",  lw=1.2, label="zero")
    ax2.axvline(-clip_mm, color="blue",  ls="--", lw=1.0, label=f"−{clip_mm} mm")
    ax2.axvline( clip_mm, color="red",   ls="--", lw=1.0, label=f"+{clip_mm} mm")
    ax2.set_title("Deviation distribution")
    ax2.set_xlabel("deviation (mm)"); ax2.set_ylabel("count")
    ax2.legend(fontsize=8)

    neg_frac = 100 * (deviation < -0.1).mean()
    pos_frac = 100 * (deviation >  0.1).mean()
    ax2.text(0.02, 0.97,
             f"< −0.1 mm: {neg_frac:.1f}%\n> +0.1 mm: {pos_frac:.1f}%",
             transform=ax2.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Interactive plot ──────────────────────────────────────────────────────────
def make_interactive_plot(pts, deviation, smooth_radius, clip_mm, label, max_pts, out_path):
    import plotly.graph_objects as go

    step  = max(1, len(pts) // max_pts)
    pts_d = pts[::step]
    dev_d = deviation[::step]

    fig = go.Figure(go.Scatter3d(
        x=pts_d[:, 0], y=pts_d[:, 1], z=pts_d[:, 2],
        mode="markers",
        marker=dict(
            size=1,
            color=dev_d,
            colorscale="RdYlGn",
            cmin=-clip_mm,
            cmax= clip_mm,
            colorbar=dict(title="deviation (mm)"),
            opacity=0.9,
        ),
        hovertemplate="x:%{x:.1f} y:%{y:.1f} z:%{z:.1f}<br>dev:%{marker.color:.3f} mm<extra></extra>",
    ))

    fig.update_layout(
        title=f"{label} — Deviation map  (smooth R={smooth_radius} mm)",
        scene=dict(
            xaxis_title="X mm", yaxis_title="Y mm", zaxis_title="Z mm",
            aspectmode="data",
        ),
        width=1300, height=850,
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    label = os.path.splitext(os.path.basename(args.ply))[0]

    print(f"\n{'='*60}")
    print(f"  Deviation map  —  {label}")
    print(f"  grid_res={args.grid_res} mm   smooth_radius={args.smooth_radius} mm")
    print(f"{'='*60}")

    print("\n[1] Loading …")
    mesh = trimesh.load(args.ply, process=False)
    pts  = np.asarray(mesh.vertices, dtype=np.float64)
    print(f"    {len(pts):,} vertices")

    print("[2] Computing deviation map …")
    deviation, ref_grid, xi, yi = compute_deviation(pts, args.grid_res, args.smooth_radius, args.poly_degree)
    print(f"    deviation range: [{deviation.min():.3f}, {deviation.max():.3f}] mm")
    print(f"    < −0.1 mm : {100*(deviation < -0.1).mean():.1f}%  of vertices")
    print(f"    < −0.2 mm : {100*(deviation < -0.2).mean():.1f}%")

    if args.plots:
        print("[3] Static plot …")
        out_png = os.path.join(args.out_dir, f"{label}_deviation.png")
        make_static_plot(pts, deviation, args.smooth_radius, args.clip_mm, label, out_png)
        print(f"    Saved → {out_png}")

    if args.interactive:
        print("[4] Interactive HTML …")
        out_html = os.path.join(args.out_dir, f"{label}_deviation.html")
        make_interactive_plot(pts, deviation, args.smooth_radius, args.clip_mm,
                              label, args.max_html_pts, out_html)
        print(f"    Saved → {out_html}")

    print(f"\n  Done. Output in: {args.out_dir}/\n")


if __name__ == "__main__":
    main()
