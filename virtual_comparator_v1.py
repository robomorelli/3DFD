"""
Virtual Comparator v1 for 3DFD surface defect detection.

Improvements over v0:
  1. AUTO-ZEROING: for each rivet, finds the nearest nominal (near-zero,
     yellow on the deviation map) zone within ~4 cm — no manual --zero-xy.
  2. CROWN PROBE ZONE: measures an annular band [r_inner..r_outer] mm
     outside the hole boundary, where the pull-in physically lives,
     instead of using only the boundary ring itself.
  3. SECTOR ANALYSIS: divides the crown into N angular sectors and
     reports the mean of the K worst sectors — robust to partial
     (half-corona) pull-in that doesn't extend all the way around.
  4. PHYSICAL FEET GEOMETRY: uses the real circumradius of the
     equilateral triangle foot pattern (default 13.7 mm) — same value
     for both zero and rivet measurement, exactly as the real tool.

Physical comparator geometry:
  Equilateral triangle side ≈ 23 mm → circumradius (center→vertex) = 23/√3 ≈ 13.3 mm.
  The central probe (sonda) sits at the centroid, ≈ 13.7 mm from each foot.
  --feet-radius should match your real instrument.

Primary defect metric: k_worst_mean
  = mean signed distance of the c'è high.png che riporta stime vicino al mastice esagerafalse_positive_masticeK worst angular sectors in the crown,
    after subtracting the local zero reading.
  Negative = pull-in.  Positive = proud.

Usage:
      python virtual_comparator_v1.py --ply data/pc/Surface8_clean.ply
  python virtual_comparator_v1.py --ply data/pc/Surface8_clean.ply \\
      --r-inner 1.5 --r-outer 7 --n-sectors 8 --k-sectors 4 \\
      --threshold -0.2
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from deviation_map import compute_deviation
from virtual_comparator import find_holes, fit_plane_3pts, local_frame, signed_distances


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Virtual comparator v1 — auto-zero + crown sector analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ply", default="data/pc/Surface8_clean.ply",
                   help="Input PLY file")
    p.add_argument("--out-dir", default="comparator_output_v1",
                   help="Output directory")

    # Rivet detection
    p.add_argument("--hole-r-min", type=float, default=1.0,
                   help="Min rivet hole radius (mm)")
    p.add_argument("--hole-r-max", type=float, default=15.0,
                   help="Max rivet hole radius (mm)")

    # Physical comparator geometry (same for zero and rivet measurement)
    p.add_argument("--feet-radius", type=float, default=13.7,
                   help="Circumradius of the 3-foot equilateral triangle (mm). "
                        "Real instrument: side=23mm → 23/√3≈13.3mm; use your measured value.")

    # Crown geometry
    p.add_argument("--r-inner", type=float, default=1.5,
                   help="Crown inner offset from hole edge (mm)")
    p.add_argument("--r-outer", type=float, default=None,
                   help="Crown outer offset from hole edge (mm). "
                        "Default: auto = half the mean edge-to-edge distance between rivets.")

    # Sector analysis
    p.add_argument("--n-sectors",      type=int, default=4,
                   help="Number of angular sectors (90° each for default 4)")
    p.add_argument("--k-sectors",      type=int, default=2,
                   help="Number of worst sectors to average (default 2 = half-corona)")
    p.add_argument("--n-radial-bands", type=int, default=3,
                   help="Radial subdivisions per sector (inner→outer). "
                        "k_worst_mean uses per-sector worst-band, not full-sector average.")

    # Auto-zero
    p.add_argument("--zero-probe-radius", type=float, default=4.0,
                   help="Probe disc radius at zero point (mm)")
    p.add_argument("--green-thresh", type=float, default=0.05,
                   help="Max |deviation| for reporting green vertex count (mm)")
    p.add_argument("--zero-nominal-thresh", type=float, default=0.10,
                   help="Max |deviation| to qualify as nominal (yellow) for zeroing (mm). "
                        "If no nominal candidate found, falls back to least-red in the band.")
    p.add_argument("--zero-from-edge-min", type=float, default=13.0,
                   help="Min distance from hole edge to zero center (mm)")
    p.add_argument("--zero-from-edge-max", type=float, default=60.0,
                   help="Max distance from hole edge to zero center (mm) — NTA 70901 allows up to 5-6 cm")

    # Deviation map (for green-zone detection)
    p.add_argument("--grid-res",      type=float, default=0.4,  help="Grid resolution (mm)")
    p.add_argument("--smooth-radius", type=float, default=20.0, help="Gaussian smooth radius (mm)")
    p.add_argument("--poly-degree",   type=int,   default=4,    help="Polynomial degree for curvature removal")

    # Detection
    p.add_argument("--threshold", type=float, default=-0.2,
                   help="Defect threshold on k_worst_mean (mm). Values below are flagged.")

    # Colour zones
    p.add_argument("--warn-lo", type=float, default=0.14,
                   help="Orange zone start |pull-in| (mm) — below this is green")
    p.add_argument("--warn-hi", type=float, default=0.21,
                   help="Red start |pull-in| (mm) — above this is red, below is orange")
    p.add_argument("--critical-hi", type=float, default=0.60,
                   help="Black (critical) start |pull-in| (mm) — above this is black")

    # Foot validity
    p.add_argument("--foot-dist-max", type=float, default=3.0,
                   help="Max allowed distance (mm) between a foot target and the "
                        "nearest mesh point. If exceeded the foot is off-panel. "
                        "The script retries with smaller radii down to --feet-radius-min.")
    p.add_argument("--feet-radius-min", type=float, default=6.0,
                   help="Minimum feet radius tried when full radius puts feet off-panel (mm)")

    # Measurement mode
    p.add_argument("--measure-mode", choices=["plane", "deviation", "local-poly", "per-point"],
                   default="plane",
                   help="plane: signed distance to 3-foot local plane (default). "
                        "deviation: deviation-map residual (global curvature removed). "
                        "local-poly: polynomial fit to the nominal ring outside the crown. "
                        "per-point: one comparator placement per crown point — feet land "
                        "around that point, new plane for each, then subtract zero offset.")
    p.add_argument("--local-poly-fit-radius", type=float, default=30.0,
                   help="[local-poly] radius (mm) of the local fitting region around the rivet")
    p.add_argument("--local-poly-degree", type=int, default=2,
                   help="[local-poly] degree of the 2-D polynomial fit (default 2 = quadratic)")
    p.add_argument("--local-poly-method", choices=["exclude", "robust"], default="exclude",
                   help="[local-poly] exclude: fit only outside crown+1mm (needs r_outer). "
                        "robust: fit all points in patch, iteratively reject negative outliers "
                        "(pull-in) — no r_outer dependency.")
    p.add_argument("--max-crown-dev-range", type=float, default=None,
                   help="[plane] skip rivets whose crown deviation-map range exceeds this "
                        "value (mm). Filters measurements biased by local surface curvature. "
                        "Suggested: 0.15 mm. Default: disabled.")

    # Output
    p.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=True,
                   help="Save interactive Plotly HTML")
    p.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True,
                   help="Save static matplotlib PNG")
    return p.parse_args()


# ── Geometry helpers ──────────────────────────────────────────────────────────
def compute_plane_at_point(center, pts, kdtree, feet_radius,
                           hole_centers=None, hole_radii=None,
                           mastic_centers=None, mastic_radii=None,
                           mastic_bound_tree=None, mastic_bound_r=3.0):
    """
    Place 3 feet at feet_radius from center on the local tangent plane.
    Returns (plane_n, foot_pts, foot_max_dist) or (None, None, inf).

    foot_max_dist: largest gap between any foot target and the nearest mesh
                   vertex — large values mean the foot landed off-panel.
    Returns inf also when any foot target falls inside a rivet hole or mastic,
    so find_valid_plane retries with a smaller radius.
    """
    n_vec, u_vec, v_vec = local_frame(center, pts, kdtree, feet_radius * 1.5)
    angles  = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    targets = center + feet_radius * (
        np.outer(np.cos(angles), u_vec) + np.outer(np.sin(angles), v_vec)
    )

    # Reject if any foot target falls inside another rivet hole
    if hole_centers is not None:
        for hc, hr in zip(hole_centers, hole_radii):
            if any(np.linalg.norm(t[:2] - np.asarray(hc[:2])) < hr for t in targets):
                return None, None, float("inf")

    # Reject if any foot target falls on a mastic region (centroid-based fast check)
    if mastic_centers is not None:
        for mc, mr in zip(mastic_centers, mastic_radii):
            if any(np.linalg.norm(t[:2] - np.asarray(mc[:2])) < mr for t in targets):
                return None, None, float("inf")

    # Reject if any foot target is within buffer of a mastic boundary vertex
    # (handles elongated mastics where centroid-based check misses the ends)
    if mastic_bound_tree is not None:
        d_boundary, _ = mastic_bound_tree.query(targets[:, :2])
        if (d_boundary < mastic_bound_r).any():
            return None, None, float("inf")

    dists, foot_idxs = kdtree.query(targets)
    foot_pts      = pts[foot_idxs]
    foot_max_dist = float(dists.max())

    e1, e2 = foot_pts[1] - foot_pts[0], foot_pts[2] - foot_pts[0]
    if np.linalg.norm(np.cross(e1, e2)) < 1e-6:
        return None, None, float("inf")

    plane_n, _ = fit_plane_3pts(foot_pts[0], foot_pts[1], foot_pts[2])
    if np.dot(plane_n, n_vec) < 0:
        plane_n = -plane_n
    return plane_n, foot_pts, foot_max_dist


def find_valid_plane(center, pts, kdtree, feet_radius,
                     foot_dist_max, feet_radius_min,
                     hole_centers=None, hole_radii=None,
                     mastic_centers=None, mastic_radii=None,
                     mastic_bound_tree=None, mastic_bound_r=3.0):
    """
    Try compute_plane_at_point with decreasing feet_radius until all feet
    are within foot_dist_max of the mesh, none inside a rivet hole, and none
    on a mastic region.  Retries with smaller radius down to feet_radius_min.

    Returns (plane_n, foot_pts, actual_feet_radius, foot_max_dist, foot_ok).
    foot_ok=False means even the smallest radius had a foot off-panel/on-hole/on-mastic.
    """
    step = (feet_radius - feet_radius_min) / 4
    r = feet_radius
    while r >= feet_radius_min - 1e-6:
        plane_n, foot_pts, fmd = compute_plane_at_point(
            center, pts, kdtree, r,
            hole_centers=hole_centers, hole_radii=hole_radii,
            mastic_centers=mastic_centers, mastic_radii=mastic_radii,
            mastic_bound_tree=mastic_bound_tree, mastic_bound_r=mastic_bound_r,
        )
        if plane_n is None:
            r -= max(step, 0.5)
            continue
        if fmd <= foot_dist_max:
            return plane_n, foot_pts, r, fmd, True
        r -= max(step, 0.5)

    # Last attempt at minimum radius — return anyway but flag as invalid
    plane_n, foot_pts, fmd = compute_plane_at_point(
        center, pts, kdtree, feet_radius_min,
        hole_centers=hole_centers, hole_radii=hole_radii,
        mastic_centers=mastic_centers, mastic_radii=mastic_radii,
        mastic_bound_tree=mastic_bound_tree, mastic_bound_r=mastic_bound_r,
    )
    ok = (plane_n is not None) and (fmd <= foot_dist_max)
    return plane_n, foot_pts, feet_radius_min, fmd, ok


# ── Auto r_outer from rivet spacing ──────────────────────────────────────────
def auto_r_outer(holes):
    """
    r_outer = half the mean edge-to-edge distance between nearest-neighbour rivets.

    For each rivet the nearest other rivet is found; the mean of those
    center-to-center distances minus 2×mean_hole_radius gives the mean
    edge-to-edge gap.  r_outer is half that gap — the midpoint between adjacent
    hole edges.  Falls back to 7.0 mm if fewer than 2 rivets are present.
    """
    if len(holes) < 2:
        return 7.0
    centers = np.array([h["center"][:2] for h in holes], dtype=np.float64)
    tree    = cKDTree(centers)
    nn_d, _ = tree.query(centers, k=2)          # k=2: [self, nearest]
    d_avg   = float(nn_d[:, 1].mean())
    r_avg   = float(np.mean([h["radius_mm"] for h in holes]))
    r_out   = (d_avg - 2.0 * r_avg) / 2.0
    return max(r_out, 2.0)                       # floor at 2 mm


# ── Auto-zero point search ────────────────────────────────────────────────────
def find_zero_point(rivet_center, hole_radius, pts, kdtree, deviation,
                    nominal_thresh=0.10, from_edge_min=13.0, from_edge_max=60.0,
                    feet_radius=13.7,
                    other_centers=None, other_radii=None,
                    mastic_centers=None, mastic_radii=None,
                    mastic_bound_tree=None, mastic_bound_r=3.0,
                    zero_search="free", crown_buffer=0.0):
    """
    Find the nearest nominal (yellow, near-zero) vertex for zeroing.

    Distances measured from hole edge:
      d_min = hole_radius + from_edge_min
      d_max = hole_radius + from_edge_max   (up to 6 cm per NTA 70901)

    Selection:
      1. Prefer candidates with |deviation| < nominal_thresh (yellow zone).
         Among those, take the nearest.
      2. If no nominal candidate exists, take the one with the highest
         deviation (least red) among all non-excluded candidates.

    Excludes zones covered by other rivet holes or mastic.

    Returns a (3,) point array or None if no candidates remain.
    """
    d_min = hole_radius + from_edge_min
    d_max = hole_radius + from_edge_max

    idxs = np.array(kdtree.query_ball_point(rivet_center, d_max), dtype=np.int32)
    if len(idxs) == 0:
        return None

    d2d  = np.linalg.norm(pts[idxs, :2] - rivet_center[:2], axis=1)
    mask = d2d >= d_min
    idxs, d2d = idxs[mask], d2d[mask]
    if len(idxs) == 0:
        return None

    # Exclude neighbouring rivet holes (and optionally their crown zone).
    # free:    exclude only the hole body     → d > hole_r + probe_r
    # bounded: also exclude the crown zone   → d > hole_r + crown_buffer + probe_r
    probe_r = 4.0   # default probe disc radius
    if other_centers is not None and len(other_centers) > 0:
        keep = np.ones(len(idxs), dtype=bool)
        extra = crown_buffer if zero_search == "bounded" else 0.0
        for oc, or_ in zip(other_centers, other_radii):
            if np.linalg.norm(np.asarray(oc[:2]) - rivet_center[:2]) < 1e-3:
                continue
            d_oc = np.linalg.norm(pts[idxs, :2] - np.asarray(oc[:2]), axis=1)
            keep &= d_oc > (or_ + extra + probe_r)
        idxs, d2d = idxs[keep], d2d[keep]

    # Mastic exclusion is stricter: avoid comparator landing on mastic material
    if mastic_centers is not None and len(mastic_centers) > 0:
        keep = np.ones(len(idxs), dtype=bool)
        for mc, mr in zip(mastic_centers, mastic_radii):
            d_mc = np.linalg.norm(pts[idxs, :2] - np.asarray(mc[:2]), axis=1)
            keep &= d_mc > (mr + feet_radius + 2.0)
        idxs, d2d = idxs[keep], d2d[keep]

    # Boundary tree: keep away from mastic boundary by feet_radius + buffer
    # (handles elongated mastics where centroid-based check misses the ends)
    if mastic_bound_tree is not None and len(idxs) > 0:
        d_boundary, _ = mastic_bound_tree.query(pts[idxs, :2])
        keep = d_boundary >= (mastic_bound_r + feet_radius + 2.0)
        idxs, d2d = idxs[keep], d2d[keep]

    if len(idxs) == 0:
        return None

    # Prefer nominal (yellow, near-zero) — take nearest among them
    nominal = np.abs(deviation[idxs]) < nominal_thresh
    if nominal.any():
        sub_idxs = idxs[nominal]
        sub_d2d  = d2d[nominal]
        return pts[sub_idxs[np.argmin(sub_d2d)]].copy()

    # Fallback: no nominal zone found — take least red (highest deviation)
    return pts[idxs[np.argmax(deviation[idxs])]].copy()


def zero_reading_at(zero_center, pts, kdtree, feet_radius, probe_radius,
                    foot_dist_max=3.0, feet_radius_min=6.0):
    """
    Place the comparator at zero_center; return the mean signed distance of
    the probe disc to the 3-foot plane.  Returns None on failure.
    Also returns None if feet are off-panel (foot_dist > foot_dist_max).
    """
    plane_n, foot_pts, _, fmd, foot_ok = find_valid_plane(
        zero_center, pts, kdtree, feet_radius, foot_dist_max, feet_radius_min)
    if plane_n is None or not foot_ok:
        return None
    probe_idxs = np.array(kdtree.query_ball_point(zero_center, probe_radius), dtype=np.int32)
    if len(probe_idxs) == 0:
        return None
    dists = signed_distances(pts[probe_idxs], plane_n, foot_pts[0])
    return float(dists.mean())


def deviation_zero_reading(zero_center, pts, kdtree, deviation, probe_radius):
    """Median deviation-map value in a disc of probe_radius around zero_center."""
    idxs = np.array(kdtree.query_ball_point(zero_center, probe_radius), dtype=np.int32)
    if len(idxs) == 0:
        return None
    return float(np.median(deviation[idxs]))


# ── Local polynomial helpers ──────────────────────────────────────────────────
def _poly2d_basis(x, y, degree):
    """Vandermonde matrix for a 2-D polynomial up to `degree`."""
    cols = []
    for d in range(degree + 1):
        for i in range(d + 1):
            cols.append(x ** (d - i) * y ** i)
    return np.column_stack(cols)


def _robust_poly_fit(x, y, z, degree, n_iter=4, reject_sigma=1.5):
    """
    Iterative negative-outlier rejection for 2-D polynomial fit.

    Pull-in depressions are strong NEGATIVE residuals.  Each iteration:
      1. Fit OLS on current inlier set.
      2. Compute residuals; estimate scatter from the positive half (robust σ).
      3. Reject points with residual < -reject_sigma * σ.
    Returns coefficients of the final fit on the cleaned inlier set.
    """
    A = _poly2d_basis(x, y, degree)
    keep = np.ones(len(z), dtype=bool)
    min_pts = A.shape[1] * 2
    for _ in range(n_iter):
        Ak, zk = A[keep], z[keep]
        coeffs, _, _, _ = np.linalg.lstsq(Ak, zk, rcond=None)
        resid = z - A @ coeffs
        # Estimate σ from positive residuals only (unaffected by pull-in)
        pos = resid[resid > 0]
        sigma = float(pos.std()) if len(pos) > 3 else float(np.std(resid))
        new_keep = resid > -reject_sigma * sigma
        if new_keep.sum() < min_pts or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    coeffs, _, _, _ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
    return coeffs


# ── Crown measurement with sector analysis ────────────────────────────────────
def measure_crown_v1(rivet_center, hole_radius, pts, kdtree,
                     feet_radius, r_inner_mm, r_outer_mm,
                     n_sectors, k_worst, n_radial_bands=3,
                     zero_offset=0.0,
                     foot_dist_max=3.0, feet_radius_min=6.0,
                     other_hole_centers=None, other_hole_radii=None,
                     mastic_centers=None, mastic_radii=None,
                     mastic_bound_tree=None, mastic_bound_r=3.0,
                     min_sector_coverage=0.3,
                     deviation_arr=None, measure_mode="plane",
                     local_poly_fit_radius=30.0, local_poly_degree=2,
                     local_poly_method="exclude"):
    """
    Measure the annular crown [hole_r+r_inner, hole_r+r_outer] around a rivet.

    Crown is divided into a 2D grid: n_sectors angular × n_radial_bands radial.
    For each sector the worst (most negative) radial band is identified.
    k_worst_mean = mean of the k worst per-sector worst-band values.

    Coherence: a geometrically consistent pull-in ring shows the same worst
    radial band across all sectors.  coherence = fraction of sectors agreeing
    on the consensus_band.

    Returns a dict with all metrics, or None if too few crown points.

    --- Known limitation of measure_mode="plane" ---
    If a foot lands inside the pull-in zone of THIS rivet (depressed by δ),
    the 3-foot plane tilts toward that foot and the measurement at any probe
    point shifts by approximately δ/3 (appears falsely proud).  In practice
    the feet sit ~13-20 mm from the rivet centre, where pull-in depression is
    usually < 0.05 mm, so the bias is < 0.02 mm — negligible for normal rivets.
    It becomes relevant for very large deformation zones (A > 15 mm radius) or
    in dense arrays where a foot lands in a neighbouring rivet's depression.
    """
    plane_n, foot_pts, actual_feet_r, foot_max_dist, foot_ok = find_valid_plane(
        rivet_center, pts, kdtree, feet_radius, foot_dist_max, feet_radius_min,
        hole_centers=other_hole_centers, hole_radii=other_hole_radii,
        mastic_centers=mastic_centers, mastic_radii=mastic_radii,
        mastic_bound_tree=mastic_bound_tree, mastic_bound_r=mastic_bound_r,
    )
    if plane_n is None:
        return None

    r_in  = hole_radius + r_inner_mm
    r_out = hole_radius + r_outer_mm

    cand = np.array(kdtree.query_ball_point(rivet_center, r_out), dtype=np.int32)
    if len(cand) == 0:
        return None

    d2d        = np.linalg.norm(pts[cand, :2] - rivet_center[:2], axis=1)
    crown_mask = (d2d >= r_in) & (d2d <= r_out)
    crown_idx  = cand[crown_mask]
    crown_pts  = pts[crown_idx]
    d2d_crown  = d2d[crown_mask]

    # Exclude crown points that fall over another rivet hole or mastic:
    # those vertices sit on the edge of a void or on raised mastic material
    # and would corrupt the average.
    valid_crown = np.ones(len(crown_pts), dtype=bool)
    if other_hole_centers is not None:
        for hc, hr in zip(other_hole_centers, other_hole_radii):
            d_h = np.linalg.norm(crown_pts[:, :2] - np.asarray(hc[:2]), axis=1)
            valid_crown &= d_h > hr
    if mastic_centers is not None:
        for mc, mr in zip(mastic_centers, mastic_radii):
            d_m = np.linalg.norm(crown_pts[:, :2] - np.asarray(mc[:2]), axis=1)
            valid_crown &= d_m > mr
    # Boundary tree: exclude crown points within buffer of any mastic boundary vertex
    # (handles elongated mastics where the centroid-based check misses the ends)
    if mastic_bound_tree is not None:
        d_boundary, _ = mastic_bound_tree.query(crown_pts[:, :2])
        valid_crown &= d_boundary >= mastic_bound_r
    crown_idx  = crown_idx[valid_crown]
    crown_pts  = crown_pts[valid_crown]
    d2d_crown  = d2d_crown[valid_crown]

    if len(crown_pts) < max(n_sectors * n_radial_bands, 5):
        return None

    # Crown deviations — three modes:
    #   plane:      signed distance to 3-foot local plane (comparator simulation)
    #   deviation:  deviation-map residual per vertex (global curvature removed)
    #   local-poly: residual from polynomial fit to the nominal ring outside the
    #               crown; excludes pull-in points from the fit so the reference
    #               surface is the true nominal, not a biased average.
    if measure_mode == "per-point":
        # For each crown point: place the comparator centred on that point,
        # let the feet land on the mesh, build a local plane, measure the
        # signed distance of the probe (the crown point itself) to that plane,
        # then subtract zero_offset.
        all_hc = list(other_hole_centers or []) + [rivet_center]
        all_hr = list(other_hole_radii   or []) + [hole_radius]
        raw = np.full(len(crown_pts), np.nan)
        for k, cp in enumerate(crown_pts):
            pn, fp, _, _, fok = find_valid_plane(
                cp, pts, kdtree, feet_radius, foot_dist_max, feet_radius_min,
                hole_centers=all_hc, hole_radii=all_hr,
                mastic_centers=mastic_centers, mastic_radii=mastic_radii,
                mastic_bound_tree=mastic_bound_tree, mastic_bound_r=mastic_bound_r,
            )
            if pn is not None and fok:
                raw[k] = float((cp - fp[0]) @ pn)
        valid = np.isfinite(raw)
        crown_idx  = crown_idx[valid]
        crown_pts  = crown_pts[valid]
        d2d_crown  = d2d_crown[valid]
        dists      = raw[valid] - zero_offset
        if len(crown_pts) < max(n_sectors * n_radial_bands, 5):
            return None

    elif measure_mode == "deviation" and deviation_arr is not None:
        dists = deviation_arr[crown_idx] - zero_offset

    elif measure_mode == "local-poly":
        cx, cy = rivet_center[0], rivet_center[1]

        # Collect all points within the local fitting radius
        fit_cand = np.array(
            kdtree.query_ball_point(rivet_center, local_poly_fit_radius),
            dtype=np.int32,
        )
        fit_d2d = np.linalg.norm(pts[fit_cand, :2] - rivet_center[:2], axis=1)

        if local_poly_method == "exclude":
            # Keep only the nominal zone: outside own crown + 1 mm margin.
            # Also exclude neighbouring rivets' crown zones (not just their holes)
            # to avoid the nominal ring being contaminated by adjacent pull-ins.
            r_out  = hole_radius + r_outer_mm
            fit_ok = fit_d2d >= r_out + 1.0
            fit_idx = fit_cand[fit_ok]
            if other_hole_centers is not None:
                for hc, hr in zip(other_hole_centers, other_hole_radii):
                    d_h = np.linalg.norm(pts[fit_idx, :2] - np.asarray(hc[:2]), axis=1)
                    fit_idx = fit_idx[d_h > hr]
        else:  # "robust" — use all points, iteratively reject negative outliers
            fit_ok = fit_d2d >= hole_radius
            fit_idx = fit_cand[fit_ok]
            if other_hole_centers is not None:
                for hc, hr in zip(other_hole_centers, other_hole_radii):
                    d_h = np.linalg.norm(pts[fit_idx, :2] - np.asarray(hc[:2]), axis=1)
                    fit_idx = fit_idx[d_h > hr]   # robust handles pull-in by itself

        min_pts = (local_poly_degree + 1) * (local_poly_degree + 2) // 2
        if len(fit_idx) >= min_pts * 3:
            xf = pts[fit_idx, 0] - cx
            yf = pts[fit_idx, 1] - cy
            zf = pts[fit_idx, 2]

            if local_poly_method == "robust":
                coeffs = _robust_poly_fit(xf, yf, zf, local_poly_degree)
            else:
                coeffs, _, _, _ = np.linalg.lstsq(
                    _poly2d_basis(xf, yf, local_poly_degree), zf, rcond=None)

            xc = crown_pts[:, 0] - cx
            yc = crown_pts[:, 1] - cy
            z_ref = _poly2d_basis(xc, yc, local_poly_degree) @ coeffs
            dists = crown_pts[:, 2] - z_ref
        else:
            # Not enough fit points — fall back to plane mode
            dists = signed_distances(crown_pts, plane_n, foot_pts[0]) - zero_offset

    else:
        dists = signed_distances(crown_pts, plane_n, foot_pts[0]) - zero_offset

    # Angular sector index for each crown point
    dx = crown_pts[:, 0] - rivet_center[0]
    dy = crown_pts[:, 1] - rivet_center[1]
    angles_pts = np.arctan2(dy, dx)
    sector_w   = 2 * np.pi / n_sectors
    sector_ids = (np.floor((angles_pts + np.pi) / sector_w).astype(int) % n_sectors)

    # Radial band index for each crown point
    band_w  = (r_out - r_in) / n_radial_bands
    band_ids = np.clip(
        ((d2d_crown - r_in) / band_w).astype(int), 0, n_radial_bands - 1
    )

    # Per-sector point counts — detect void/boundary sectors
    pts_per_sector   = np.array([(sector_ids == s).sum() for s in range(n_sectors)])
    expected_per_sec = len(crown_pts) / n_sectors
    # A sector with fewer than min_sector_coverage × expected points is partially
    # in the void (panel edge) and would give spurious extreme readings.
    sector_ok = pts_per_sector >= min_sector_coverage * expected_per_sec

    # 2D grid: grid[sector, band] = mean deviation (nan if no points or void sector)
    grid = np.full((n_sectors, n_radial_bands), np.nan)
    for s in range(n_sectors):
        if not sector_ok[s]:
            continue
        for b in range(n_radial_bands):
            m = (sector_ids == s) & (band_ids == b)
            if m.sum() >= 1:
                grid[s, b] = float(dists[m].mean())

    # Per sector: worst (most negative) radial band
    worst_val  = np.full(n_sectors, np.nan)
    worst_band = np.full(n_sectors, -1, dtype=int)
    for s in range(n_sectors):
        row = grid[s, :]
        if not np.all(np.isnan(row)):
            b = int(np.nanargmin(row))
            worst_band[s] = b
            worst_val[s]  = row[b]

    # k_worst_mean: mean of k most-depressed per-sector worst-band values
    valid   = ~np.isnan(worst_val)
    n_pop   = int(valid.sum())
    k_act   = min(k_worst, n_pop)
    k_worst_mean = float(np.sort(worst_val[valid])[:k_act].mean()) if k_act > 0 else float("nan")

    # Coherence: fraction of sectors sharing the same worst radial band
    valid_bands = worst_band[valid]
    if len(valid_bands) > 0:
        from collections import Counter
        consensus_band, cnt = Counter(valid_bands.tolist()).most_common(1)[0]
        coherence      = cnt / n_pop
        coherent_mean  = float(np.nanmean(grid[:, consensus_band]))
    else:
        consensus_band = -1
        coherence      = 0.0
        coherent_mean  = float("nan")

    # Sector averages over full radial width (kept for histogram/plot)
    s_means  = np.nanmean(grid, axis=1)
    s_counts = np.array([int((sector_ids == s).sum()) for s in range(n_sectors)])
    populated = ~np.isnan(s_means)

    # Curvature quality flag (plane mode): if the deviation map range across
    # the crown is large, the 3-foot plane is likely biased by local surface
    # tilt — the measurement may be unreliable.
    if deviation_arr is not None and measure_mode == "plane":
        dev_crown = deviation_arr[crown_idx]
        crown_dev_range = float(np.nanmax(dev_crown) - np.nanmin(dev_crown))
        crown_dev_std   = float(np.nanstd(dev_crown))
    else:
        crown_dev_range = float("nan")
        crown_dev_std   = float("nan")

    return {
        "center":            rivet_center,
        "hole_r":            float(hole_radius),
        "feet_r":            float(actual_feet_r),
        "foot_max_dist":     float(foot_max_dist),
        "foot_ok":           bool(foot_ok),
        "n_crown":           int(len(crown_pts)),
        "crown_mean":        float(dists.mean()),
        "crown_p10":         float(np.percentile(dists, 10)),
        "crown_min":         float(dists.min()),
        "k_worst_mean":      k_worst_mean,
        "worst_sector_mean": float(np.nanmin(worst_val)) if n_pop > 0 else float("nan"),
        "sector_means":      [float(v) if not np.isnan(v) else None for v in s_means],
        "sector_counts":     s_counts.tolist(),
        "n_sectors_pop":     n_pop,
        "n_sectors_void":    int((~sector_ok).sum()),
        "n_sectors_below":   int((worst_val[valid] < 0).sum()),
        "consensus_band":    int(consensus_band),
        "coherence":         round(float(coherence), 3),
        "coherent_mean":     coherent_mean,
        "sector_grid":       grid.tolist(),
        "zero_offset":       float(zero_offset),
        "crown_dev_range":   round(crown_dev_range, 4),
        "crown_dev_std":     round(crown_dev_std, 4),
    }


# ── Rivet colour helper ───────────────────────────────────────────────────────
def _rivet_col(r, args):
    """4-zone colour: limegreen / darkorange / red / black (critical)."""
    v = round(r["k_worst_mean"], 2)   # match label precision (:.2f) so colour = what label shows
    if not r.get("foot_ok", True) or not np.isfinite(v):
        return "gray"
    critical = getattr(args, "critical_hi", 0.60)
    if v <= -critical:
        return "black"
    elif v <= -args.warn_hi:
        return "red"
    elif v <= -args.warn_lo:
        return "darkorange"
    else:
        return "limegreen"


# ── Static plot ───────────────────────────────────────────────────────────────
def make_static_plot(pts, holes, mastics, results, threshold, label, out_path, args):
    step = max(1, len(pts) // 300_000)
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.suptitle(
        f"Virtual Comparator v1  —  {label}\n"
        f"Crown [{args.r_inner:.1f}..{args.r_outer:.1f}] mm  |  "
        f"feet R={args.feet_radius:.1f} mm  |  "
        f"K={args.k_sectors}/{args.n_sectors} sectors",
        fontsize=11, fontweight="bold",
    )

    ax.scatter(pts[::step, 0], pts[::step, 1], c=pts[::step, 2],
               s=0.1, cmap="gray", rasterized=True, alpha=0.4)

    for r in results:
        c       = r["center"]
        v       = r["k_worst_mean"]
        foot_ok = r.get("foot_ok", True)
        col   = _rivet_col(r, args)
        alpha = 0.85 if foot_ok else 0.35
        ls    = "-" if foot_ok else "--"

        ax.add_patch(plt.Circle((c[0], c[1]), r["hole_r"],
                                color=col, fill=True, alpha=alpha, linewidth=0))
        ax.add_patch(plt.Circle((c[0], c[1]), r["hole_r"] + args.r_outer,
                                color=col, fill=False, linewidth=0.8,
                                alpha=0.5, linestyle=ls))

        # Crown grid: radial band rings + sector dividers
        r_in  = r["hole_r"] + args.r_inner
        r_out = r["hole_r"] + args.r_outer
        band_w   = (r_out - r_in) / args.n_radial_bands
        sector_w = 2 * np.pi / args.n_sectors
        for k in range(args.n_radial_bands):
            ax.add_patch(plt.Circle((c[0], c[1]), r_in + k * band_w,
                                    color="gray", fill=False, linewidth=0.4,
                                    alpha=0.35, linestyle="--", zorder=2))
        for k in range(args.n_sectors):
            theta = -np.pi + k * sector_w
            ct, st = np.cos(theta), np.sin(theta)
            ax.plot([c[0] + r_in * ct, c[0] + r_out * ct],
                    [c[1] + r_in * st, c[1] + r_out * st],
                    color="gray", linewidth=0.5, alpha=0.4, zorder=2)

        coh     = r.get("coherence", 1.0)
        coh_str = f" c{coh:.0%}" if coh < 1.0 else ""
        lbl = f"{v:.2f}{coh_str}" if foot_ok else f"({v:.2f})??"
        ax.annotate(lbl, (c[0], c[1]), fontsize=4,
                    ha="center", va="center",
                    color="black" if foot_ok else "dimgray")

        if r.get("zero_center") is not None:
            zc = r["zero_center"]
            ax.plot(zc[0], zc[1], marker="+", ms=5, color="cyan",
                    lw=0, markeredgewidth=1.2, zorder=6)
            ax.plot([c[0], zc[0]], [c[1], zc[1]],
                    color="cyan", linewidth=0.5, linestyle="--",
                    alpha=0.6, zorder=5)

    for m in mastics:
        mv = pts[m["verts"]]
        ax.scatter(mv[:, 0], mv[:, 1], s=1, c="orange",
                   alpha=0.5, linewidths=0, zorder=3)

    critical = getattr(args, "critical_hi", 0.60)
    legend_elems = [
        mpatches.Patch(color="limegreen",  label=f"OK  (>{-args.warn_lo:.2f} mm)"),
        mpatches.Patch(color="darkorange", label=f"Warn  [{-args.warn_hi:.2f}..{-args.warn_lo:.2f}] mm"),
        mpatches.Patch(color="red",        label=f"Difetto  [{-critical:.2f}..{-args.warn_hi:.2f}] mm"),
        mpatches.Patch(color="black",      label=f"Critico  (<{-critical:.2f} mm)"),
        mpatches.Patch(color="gray",       label="piedi fuori pannello"),
    ]
    ax.legend(handles=legend_elems, fontsize=7, loc="upper right")
    ax.set_aspect("equal")
    ax.set_title("k_worst_mean  (cyan + = punto di zero  ·  arancio = mastice)", fontsize=9)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Zero-point map ────────────────────────────────────────────────────────────
def make_zero_plot(pts, mastics, results, label, out_path, args):
    """Per ogni rivetto: linea tratteggiata rivetto→zero + croce al punto di zero."""
    step = max(1, len(pts) // 300_000)
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.suptitle(
        f"Punti di azzeramento  —  {label}\n"
        f"Linea tratteggiata: rivetto → punto di zero  |  + = zero  |  cerchio = rivetto",
        fontsize=11, fontweight="bold",
    )

    ax.scatter(pts[::step, 0], pts[::step, 1], c=pts[::step, 2],
               s=0.1, cmap="gray", rasterized=True, alpha=0.35)

    zero_dists = []
    for r in results:
        c       = r["center"]
        zc      = r.get("zero_center")
        col     = _rivet_col(r, args)
        foot_ok = r.get("foot_ok", True)

        # Rivetto colorato per risultato
        ax.add_patch(plt.Circle((c[0], c[1]), r["hole_r"],
                                color=col, fill=True,
                                alpha=0.8 if foot_ok else 0.3, linewidth=0))

        if zc is not None:
            d = float(np.linalg.norm(np.array(zc[:2]) - np.array(c[:2])))
            zero_dists.append(d)
            ax.plot([c[0], zc[0]], [c[1], zc[1]],
                    color="cyan", alpha=0.5, linewidth=0.7,
                    linestyle="--", zorder=4)
            ax.plot(zc[0], zc[1], marker="+", ms=6, color="cyan",
                    lw=0, markeredgewidth=1.4, zorder=6)
        else:
            # Nessun zero trovato: cerchio grigio tratteggiato
            ax.add_patch(plt.Circle((c[0], c[1]), r["hole_r"] * 2,
                                    color="gray", fill=False, linewidth=0.8,
                                    linestyle=":", alpha=0.5))

    for m in mastics:
        mv = pts[m["verts"]]
        ax.scatter(mv[:, 0], mv[:, 1], s=1, c="orange",
                   alpha=0.5, linewidths=0, zorder=3)

    n_no_zero = sum(1 for r in results if r.get("zero_center") is None)
    if zero_dists:
        dist_str = (f"dist. zero–rivetto:  "
                    f"med={np.median(zero_dists):.1f} mm  "
                    f"max={max(zero_dists):.1f} mm  "
                    f"senza zero: {n_no_zero}")
    else:
        dist_str = f"nessun punto di zero trovato  (senza zero: {n_no_zero})"

    legend_elems = [
        mpatches.Patch(color="limegreen",  label="OK"),
        mpatches.Patch(color="darkorange", label="Warn"),
        mpatches.Patch(color="red",        label="Difetto"),
        mpatches.Patch(color="black",      label="Critico"),
    ]
    ax.legend(handles=legend_elems, fontsize=7, loc="upper right")
    ax.set_aspect("equal")
    ax.set_title(dist_str, fontsize=8)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Interactive plot ──────────────────────────────────────────────────────────
def make_interactive_plot(pts, holes, mastics, results, threshold, label, out_path, args):
    import plotly.graph_objects as go

    step  = max(1, len(pts) // 30_000)
    pts_d = pts[::step]

    fig = go.Figure()

    # Background point cloud
    fig.add_trace(go.Scatter3d(
        x=pts_d[:, 0], y=pts_d[:, 1], z=pts_d[:, 2],
        mode="markers",
        marker=dict(size=0.8, color=pts_d[:, 2],
                    colorscale="Greys", opacity=0.25),
        name="surface", showlegend=False,
    ))

    # Build hover text
    hover = []
    for r in results:
        s_str = "  ".join(
            f"S{i}:{v:.2f}" if v is not None else f"S{i}:—"
            for i, v in enumerate(r["sector_means"])
        )
        zone = ("RED" if _rivet_col(r, args) == "red"
                else "ORANGE" if _rivet_col(r, args) == "darkorange"
                else "ok")
        near_tag = " [near-mastic]" if r.get("near_mastic") else ""
        hover.append(
            f"Hole #{r['hole_idx'] + 1}  r={r['hole_r']:.1f} mm{near_tag}<br>"
            f"<b>k_worst_mean : {r['k_worst_mean']:.3f} mm  [{zone}]</b><br>"
            f"crown_mean    : {r['crown_mean']:.3f} mm<br>"
            f"crown_p10     : {r['crown_p10']:.3f} mm<br>"
            f"crown_min     : {r['crown_min']:.3f} mm<br>"
            f"zero_offset   : {r['zero_offset']:.3f} mm<br>"
            f"n_crown_pts   : {r['n_crown']}<br>"
            f"sectors < 0   : {r['n_sectors_below']}/{r['n_sectors_pop']}<br>"
            f"Sector means  : {s_str}"
        )

    # Rivet measurement markers — discrete 3-colour scheme
    marker_colors = [_rivet_col(r, args) for r in results]
    fig.add_trace(go.Scatter3d(
        x=[r["center"][0] for r in results],
        y=[r["center"][1] for r in results],
        z=[r["center"][2] for r in results],
        mode="markers",
        marker=dict(
            size=7,
            color=marker_colors,
            line=dict(width=1, color="black"),
        ),
        text=hover, hoverinfo="text",
        name="rivets",
    ))

    # Zero points
    zero_pts = [r for r in results if r.get("zero_center") is not None]
    if zero_pts:
        fig.add_trace(go.Scatter3d(
            x=[r["zero_center"][0] for r in zero_pts],
            y=[r["zero_center"][1] for r in zero_pts],
            z=[r["zero_center"][2] for r in zero_pts],
            mode="markers",
            marker=dict(size=4, color="cyan", symbol="cross"),
            name="zero points",
        ))

    # Mastic zones
    if mastics:
        mp   = np.concatenate([pts[m["verts"]] for m in mastics])
        sm   = max(1, len(mp) // 5000)
        fig.add_trace(go.Scatter3d(
            x=mp[::sm, 0], y=mp[::sm, 1], z=mp[::sm, 2],
            mode="markers",
            marker=dict(size=2, color="orange", opacity=0.8),
            name=f"mastic ({len(mastics)})",
        ))

    # Value labels floating above rivets
    fig.add_trace(go.Scatter3d(
        x=[r["center"][0] for r in results],
        y=[r["center"][1] for r in results],
        z=[r["center"][2] + 1.5 for r in results],
        mode="text",
        text=[f"{r['k_worst_mean']:.2f}" for r in results],
        textfont=dict(size=8, color="black"),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        title=(f"{label} — Virtual Comparator v1  "
               f"(crown {args.r_inner}..{args.r_outer} mm, "
               f"feet R={args.feet_radius:.1f} mm, "
               f"K={args.k_sectors}/{args.n_sectors} sectors — "
               f"orange [{-args.warn_hi:.2f},{-args.warn_lo:.2f}]  red <{-args.warn_hi:.2f} mm)"),
        scene=dict(xaxis_title="X mm", yaxis_title="Y mm", zaxis_title="Z mm",
                   aspectmode="data"),
        width=1400, height=900,
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


# ── CSV export ────────────────────────────────────────────────────────────────
def save_csv(results, label, out_path):
    rows = []
    for r in results:
        zc = r.get("zero_center")
        rows.append({
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
            "worst_sector_mm":   round(r["worst_sector_mean"], 4),
            "n_sectors_below":   r["n_sectors_below"],
            "n_sectors_pop":     r["n_sectors_pop"],
            "zero_offset_mm":    round(r["zero_offset"], 4),
            "has_zero":          int(zc is not None),
            "n_crown_pts":       r["n_crown"],
            "foot_ok":           int(r["foot_ok"]),
            "actual_feet_r_mm":  round(r["feet_r"], 2),
            "foot_max_dist_mm":  round(r["foot_max_dist"], 3),
            "near_mastic":       int(r.get("near_mastic", False)),
            "defect":            int(r["k_worst_mean"] <= -0.2 and r["foot_ok"]),
            "cx": round(r["center"][0], 2),
            "cy": round(r["center"][1], 2),
            "cz": round(r["center"][2], 2),
            "zero_cx": round(zc[0], 2) if zc is not None else "",
            "zero_cy": round(zc[1], 2) if zc is not None else "",
            "zero_cz": round(zc[2], 2) if zc is not None else "",
        })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args  = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    label = os.path.splitext(os.path.basename(args.ply))[0]

    print(f"\n{'='*68}")
    print(f"  Virtual Comparator v1  —  {label}")
    print(f"  Feet R : {args.feet_radius:.1f} mm  (equilateral triangle circumradius)")
    r_outer_str = f"{args.r_outer:.1f}" if args.r_outer is not None else "auto"
    print(f"  Crown  : [{args.r_inner:.1f}, {r_outer_str}] mm from hole edge")
    print(f"  Sectors: K={args.k_sectors}/{args.n_sectors} worst   threshold={args.threshold} mm")
    print(f"  Zero   : nominal_thresh={args.zero_nominal_thresh} mm   "
          f"from edge [{args.zero_from_edge_min},{args.zero_from_edge_max}] mm")
    print(f"{'='*68}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1] Loading …")
    mesh  = trimesh.load(args.ply, process=False)
    pts   = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces,    dtype=np.int32)
    print(f"    {len(pts):,} vertices,  {len(faces):,} faces")

    # ── KD-tree ───────────────────────────────────────────────────────────────
    print("[2] Building KD-tree …")
    kdtree = cKDTree(pts)

    # ── Hole detection ────────────────────────────────────────────────────────
    print("[3] Detecting rivet holes and mastic …")
    holes, mastics = find_holes(faces, pts,
                                r_min=args.hole_r_min, r_max=args.hole_r_max)
    print(f"    Rivets: {len(holes)}   Mastic: {len(mastics)}")
    if not holes:
        print("    No holes found — adjust --hole-r-min / --hole-r-max")
        return

    # ── Auto r_outer ──────────────────────────────────────────────────────────
    if args.r_outer is None:
        args.r_outer = auto_r_outer(holes)
        r_avg = np.mean([h["radius_mm"] for h in holes])
        d_avg = args.r_outer * 2 + 2 * r_avg
        print(f"    r_outer (auto): {args.r_outer:.2f} mm  "
              f"[mean rivet spacing ≈ {d_avg:.1f} mm centre-centre]")

    # ── Deviation map (for auto-zero green detection) ─────────────────────────
    print("[4] Computing deviation map for green-zone detection …")
    deviation, _, _, _ = compute_deviation(
        pts, args.grid_res, args.smooth_radius, args.poly_degree,
    )
    n_green = int((np.abs(deviation) < args.green_thresh).sum())
    print(f"    deviation range: [{deviation.min():.3f}, {deviation.max():.3f}] mm")
    print(f"    green vertices (|dev|<{args.green_thresh}): "
          f"{n_green:,}  ({100*n_green/len(pts):.1f}%)")

    # Collect rivet and mastic positions for zero search exclusion.
    # Panel perimeter is also detected as "mastic" (large radius) — exclude it
    # from the exclusion list by capping at a realistic mastic size.
    MASTIC_R_MAX = 80.0   # real mastic pieces; panel boundary >> this
    all_centers    = [h["center"]    for h in holes]
    all_radii      = [h["radius_mm"] for h in holes]
    mastic_centers = [m["center"]    for m in mastics if m["radius_mm"] < MASTIC_R_MAX]
    mastic_radii   = [m["radius_mm"] for m in mastics if m["radius_mm"] < MASTIC_R_MAX]
    print(f"    Mastic zones for exclusion: {len(mastic_centers)}/{len(mastics)} "
          f"(radius < {MASTIC_R_MAX} mm)")

    # ── Measure each rivet ────────────────────────────────────────────────────
    print(f"\n[5] Measuring {len(holes)} rivets …")
    results, skipped, no_zero = [], 0, 0

    for i, hole in enumerate(holes):
        c = hole["center"].copy()
        r = hole["radius_mm"]

        # --- Find and apply auto-zero ---
        other_c = [all_centers[j] for j in range(len(holes)) if j != i]
        other_r = [all_radii[j]   for j in range(len(holes)) if j != i]

        zero_pt = find_zero_point(
            c, r, pts, kdtree, deviation,
            nominal_thresh=args.zero_nominal_thresh,
            from_edge_min=args.zero_from_edge_min,
            from_edge_max=args.zero_from_edge_max,
            feet_radius=args.feet_radius,
            other_centers=other_c,
            other_radii=other_r,
            mastic_centers=mastic_centers,
            mastic_radii=mastic_radii,
        )

        if zero_pt is None:
            zero_offset = 0.0
            no_zero    += 1
        else:
            if args.measure_mode == "deviation":
                zo = deviation_zero_reading(zero_pt, pts, kdtree,
                                            deviation, args.zero_probe_radius)
            else:
                zo = zero_reading_at(zero_pt, pts, kdtree,
                                     args.feet_radius, args.zero_probe_radius)
            zero_offset = zo if zo is not None else 0.0

        # --- Crown measurement ---
        res = measure_crown_v1(
            c, r, pts, kdtree,
            feet_radius=args.feet_radius,
            r_inner_mm=args.r_inner,
            r_outer_mm=args.r_outer,
            n_sectors=args.n_sectors,
            k_worst=args.k_sectors,
            n_radial_bands=args.n_radial_bands,
            zero_offset=zero_offset,
            foot_dist_max=args.foot_dist_max,
            feet_radius_min=args.feet_radius_min,
            other_hole_centers=other_c,
            other_hole_radii=other_r,
            mastic_centers=mastic_centers,
            mastic_radii=mastic_radii,
            deviation_arr=deviation,
            measure_mode=args.measure_mode,
            local_poly_fit_radius=args.local_poly_fit_radius,
            local_poly_degree=args.local_poly_degree,
            local_poly_method=args.local_poly_method,
        )
        if res is None:
            skipped += 1
            continue

        # Filter: skip measurements where surface curvature in the crown
        # is too large for the plane comparator to be reliable.
        if (args.max_crown_dev_range is not None
                and np.isfinite(res["crown_dev_range"])
                and res["crown_dev_range"] > args.max_crown_dev_range):
            skipped += 1
            continue

        res["hole_idx"]    = i
        res["zero_center"] = zero_pt
        results.append(res)

    foot_off = sum(1 for r in results if not r["foot_ok"])
    print(f"    Measured: {len(results)}   Skipped: {skipped}   "
          f"No-zero fallback: {no_zero}   Foot off-panel: {foot_off}")

    if not results:
        print("  No results — cannot produce output.")
        return

    # ── Report ────────────────────────────────────────────────────────────────
    defects = [r for r in results if r["k_worst_mean"] <= args.threshold]

    print(f"\n{'─'*68}")
    print(f"  RESULTS   metric = k_worst_mean  (K={args.k_sectors} worst sectors)")
    print(f"{'─'*68}")
    print(f"  {'#':>4}  {'r(mm)':>6}  {'k_worst':>8}  {'mean':>8}  "
          f"{'p10':>7}  {'min':>7}  {'s<0':>5}  {'fR':>5}  {'fD':>5}  flag")
    for r in results:
        foot_tag = "" if r["foot_ok"] else " [FOOT!]"
        flag = ("⚠ DEFECT" if r["k_worst_mean"] <= args.threshold else "") + foot_tag
        nb   = r["n_sectors_below"]
        np_  = r["n_sectors_pop"]
        print(f"  {r['hole_idx']+1:>4}  {r['hole_r']:>6.2f}  "
              f"{r['k_worst_mean']:>8.4f}  "
              f"{r['crown_mean']:>8.4f}  "
              f"{r['crown_p10']:>7.4f}  "
              f"{r['crown_min']:>7.4f}  "
              f"{nb:>2}/{np_:<2}  "
              f"{r['feet_r']:>5.1f}  "
              f"{r['foot_max_dist']:>5.2f}  {flag}")

    print(f"\n  Rivets measured : {len(results)}")
    print(f"  Defects flagged : {len(defects)}  ({100*len(defects)/len(results):.1f}%)")
    if defects:
        worst = min(defects, key=lambda r: r["k_worst_mean"])
        wc    = worst["center"]
        print(f"  Worst pull-in   : {worst['k_worst_mean']:.4f} mm  "
              f"at ({wc[0]:.1f}, {wc[1]:.1f}, {wc[2]:.1f})")
    print(f"  No-zero fallback: {no_zero} rivets  "
          f"(zero_offset forced to 0)")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, f"{label}_comparator_v1.csv")
    save_csv(results, label, csv_path)
    print(f"\n  CSV  → {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plots:
        print("[6] Static plot …")
        out_png = os.path.join(args.out_dir, f"{label}_comparator_v1.png")
        make_static_plot(pts, holes, mastics, results,
                         args.threshold, label, out_png, args)
        print(f"    → {out_png}")

    if args.interactive:
        print("[7] Interactive plot …")
        out_html = os.path.join(args.out_dir, f"{label}_comparator_v1.html")
        make_interactive_plot(pts, holes, mastics, results,
                              args.threshold, label, out_html, args)
        print(f"    → {out_html}")

    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()