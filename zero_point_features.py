"""
Feature extraction for zero-point candidate scoring.

Each candidate zero point p is described by 9 local geometric features
that mimic what an NDT operator perceives when searching for a flat
nominal zone with a torch:

  planarity_std      std of Z-residuals from a local affine plane fit
  planarity_p10      10th percentile of residuals (negative → local depression)
  planarity_range    max - min of residuals (overall flatness)
  curvature          (a + c) from z = ax²+bxy+cy²+... (mean Hessian trace)
  sector_std         std of mean residual across 4 angular sectors
  dist_from_rivet    2D distance from the hole edge (mm)
  nearest_rivet_dist 2D distance to the nearest other rivet centre (mm)
  nearest_mastic_dist 2D distance to the nearest mastic zone (mm)
  panel_margin       distance to the panel boundary (mm; nan if unknown)

All spatial features are LOCAL — they depend only on a patch of radius
R_feat (~15 mm), not on global panel curvature, so the trained model
generalises across panels.
"""

import numpy as np
from scipy.spatial import cKDTree

FEATURE_NAMES = [
    "planarity_std",
    "planarity_p10",
    "planarity_range",
    "curvature",
    "sector_std",
    "dist_from_rivet",
    "nearest_rivet_dist",
    "nearest_mastic_dist",
    "panel_margin",
]


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_zero_candidate_features(
    p,
    rivet_center,
    hole_radius,
    pts,
    kdtree,
    other_centers,
    other_radii,
    mastic_centers,
    mastic_radii,
    panel_boundary_tree=None,
    R_feat=15.0,
    n_sectors=4,
    min_pts=10,
):
    """
    Extract local geometric features for candidate zero point p.

    Parameters
    ----------
    p                   : (3,) array — candidate zero point (mesh vertex)
    rivet_center        : (3,) array — centre of the rivet being measured
    hole_radius         : float — hole radius in mm
    pts                 : (N,3) array — full mesh vertices
    kdtree              : cKDTree built on pts
    other_centers       : list of (3,) arrays — other rivet centres
    other_radii         : list of floats — other rivet radii
    mastic_centers      : list of (3,) arrays — mastic zone centres
    mastic_radii        : list of floats — mastic zone radii
    panel_boundary_tree : cKDTree of 2D panel boundary points (optional)
    R_feat              : float — feature patch radius in mm (default 15)
    n_sectors           : int — number of angular sectors for sector_std
    min_pts             : int — minimum neighbours required; returns None if fewer

    Returns
    -------
    dict of features, or None if the patch has fewer than min_pts points.
    """
    nbr_idxs = np.array(kdtree.query_ball_point(p, R_feat), dtype=np.int32)
    if len(nbr_idxs) < min_pts:
        return None

    nbr_pts = pts[nbr_idxs]
    cx, cy  = float(p[0]), float(p[1])
    xc = nbr_pts[:, 0] - cx
    yc = nbr_pts[:, 1] - cy
    zc = nbr_pts[:, 2]

    # --- Local affine plane fit: z ≈ a·x + b·y + c ---
    A_plane = np.column_stack([xc, yc, np.ones(len(xc))])
    try:
        coeffs_plane, _, _, _ = np.linalg.lstsq(A_plane, zc, rcond=None)
        z_plane   = A_plane @ coeffs_plane
        residuals = zc - z_plane
    except np.linalg.LinAlgError:
        residuals = zc - zc.mean()

    feat_planarity_std   = float(np.std(residuals))
    feat_planarity_p10   = float(np.percentile(residuals, 10))
    feat_planarity_range = float(residuals.max() - residuals.min())

    # --- Degree-2 polynomial: z ≈ a·x²+b·x·y+c·y²+d·x+e·y+f ---
    # Mean curvature proxy = a + c  (trace of the quadratic Hessian / 2)
    A_poly = np.column_stack([
        xc**2, xc * yc, yc**2, xc, yc, np.ones(len(xc))
    ])
    try:
        coeffs_poly, _, _, _ = np.linalg.lstsq(A_poly, zc, rcond=None)
        feat_curvature = float(coeffs_poly[0] + coeffs_poly[2])
    except np.linalg.LinAlgError:
        feat_curvature = float("nan")

    # --- Sector analysis: std of mean residual per angular sector ---
    angles  = np.arctan2(yc, xc)
    bin_w   = 2 * np.pi / n_sectors
    s_means = []
    for s in range(n_sectors):
        lo = -np.pi + s * bin_w
        hi = lo + bin_w
        in_s = (angles >= lo) & (angles < hi) if s < n_sectors - 1 else angles >= lo
        if in_s.sum() >= 2:
            s_means.append(float(residuals[in_s].mean()))
    feat_sector_std = float(np.std(s_means)) if len(s_means) >= 2 else float("nan")

    # --- Geometric distances ---
    feat_dist_from_rivet = float(np.linalg.norm(p[:2] - rivet_center[:2])) - hole_radius

    if other_centers and len(other_centers) > 0:
        feat_nearest_rivet_dist = float(min(
            np.linalg.norm(p[:2] - np.asarray(oc[:2]))
            for oc in other_centers
        ))
    else:
        feat_nearest_rivet_dist = 9999.0

    if mastic_centers and len(mastic_centers) > 0:
        feat_nearest_mastic_dist = float(min(
            np.linalg.norm(p[:2] - np.asarray(mc[:2]))
            for mc in mastic_centers
        ))
    else:
        feat_nearest_mastic_dist = 9999.0

    if panel_boundary_tree is not None:
        d_bnd, _ = panel_boundary_tree.query(p[:2].reshape(1, -1))
        feat_panel_margin = float(d_bnd[0])
    else:
        feat_panel_margin = float("nan")

    return {
        "planarity_std":       feat_planarity_std,
        "planarity_p10":       feat_planarity_p10,
        "planarity_range":     feat_planarity_range,
        "curvature":           feat_curvature,
        "sector_std":          feat_sector_std,
        "dist_from_rivet":     feat_dist_from_rivet,
        "nearest_rivet_dist":  feat_nearest_rivet_dist,
        "nearest_mastic_dist": feat_nearest_mastic_dist,
        "panel_margin":        feat_panel_margin,
    }


def features_to_array(feat_dict):
    """Convert a feature dict to a 1D numpy array in FEATURE_NAMES order.
    NaN values are preserved; callers should impute before feeding to sklearn."""
    return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float64)


# ── Candidate generation ──────────────────────────────────────────────────────
def generate_zero_candidates(
    rivet_center,
    hole_radius,
    pts,
    kdtree,
    from_edge_min=13.0,
    from_edge_max=60.0,
    n_angular=16,
    n_radial=5,
    other_centers=None,
    other_radii=None,
    mastic_centers=None,
    mastic_radii=None,
    feet_radius=13.7,
    probe_r=4.0,
    snap_dist_max=5.0,
):
    """
    Generate candidate zero points on an angular × radial grid around a rivet.

    For each (angle, radius) cell, the nearest mesh vertex is found and
    returned as a candidate, subject to exclusion rules (other rivet holes,
    mastic zones).

    Returns
    -------
    np.ndarray of shape (M, 3) — unique candidate mesh vertex positions.
    """
    r_min   = hole_radius + from_edge_min
    r_max   = hole_radius + from_edge_max
    radii   = np.linspace(r_min, r_max, n_radial)
    angles  = np.linspace(0.0, 2.0 * np.pi, n_angular, endpoint=False)

    seen   = set()
    result = []

    for r in radii:
        for theta in angles:
            target = rivet_center.copy()
            target[0] += r * np.cos(theta)
            target[1] += r * np.sin(theta)

            d, idx = kdtree.query(target)
            if d > snap_dist_max:
                continue            # target too far from mesh (panel edge)
            if idx in seen:
                continue
            p = pts[idx]

            # Exclude if inside any rivet hole (own or neighbour)
            if np.linalg.norm(p[:2] - rivet_center[:2]) < hole_radius:
                continue
            if other_centers is not None:
                too_close = False
                for oc, or_ in zip(other_centers, other_radii):
                    if np.linalg.norm(p[:2] - np.asarray(oc[:2])) < or_ + probe_r:
                        too_close = True
                        break
                if too_close:
                    continue

            # Exclude if on mastic
            if mastic_centers is not None:
                on_mastic = False
                for mc, mr in zip(mastic_centers, mastic_radii):
                    if np.linalg.norm(p[:2] - np.asarray(mc[:2])) < mr + feet_radius + 2.0:
                        on_mastic = True
                        break
                if on_mastic:
                    continue

            seen.add(idx)
            result.append(p)

    return np.array(result) if result else np.empty((0, 3))