"""
Virtual Comparator for 3DFD surface defect detection.

Three zeroing strategies (--zero-method):
  flatness   Minimum local surface roughness — std of signed distances to a
             locally-fitted plane within --zero-flatness-radius (default 3 mm).
             Mimics the operator zeroing on the visually flattest area near
             the rivet.  No global deviation map required.
  local-dev  Composite score: local polynomial deviation + distance from rivet
             + neighbourhood roughness.  More discriminative than flatness alone.
             No global deviation map required.
  deviation  Minimum global deviation-map residual (legacy v1 behaviour).
             Requires computing the full polynomial+Gaussian reference surface.

Crown measurement and sector analysis are common to all methods:
  - Annular probe zone [r_inner, r_outer] mm outside the hole boundary.
  - N angular sectors; k_worst_mean = mean of K worst sectors → robust to
    half-corona pull-in.

Usage:
  python virtual_comparator.py --ply data/pc/Surface8_clean.ply
  python virtual_comparator.py --ply data/pc/Surface8_clean.ply \\
      --zero-method flatness --r-inner 1.5 --r-outer 7 \\
      --n-sectors 8 --k-sectors 4 --threshold -0.2
"""

import argparse
import csv
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from deviation_map import compute_deviation


# ── Geometry helpers ──────────────────────────────────────────────────────────
def local_frame(center, pts, kdtree, radius):
    """Estimate orthonormal frame (n, u, v) at center via PCA of local neighbourhood."""
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
    """PCA aspect ratio of a 2D point set."""
    centered = pts_xy - pts_xy.mean(axis=0)
    if len(centered) < 3:
        return 1.0
    cov = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    return float(eigvals[0] / eigvals[1]) if eigvals[1] > 1e-9 else 999.0


def find_holes(faces, pts, r_min=3.0, r_max=15.0,
               mastic_aspect_min=4.0, mastic_n_verts_min=150):
    """
    Classify boundary loops into rivets and mastics.
    Returns (rivets, mastics) — each a list of dicts:
      {center, radius_mm, minor_r_mm, n_verts, verts, aspect_ratio}
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

        hpts   = pts[comp]
        ctr    = hpts.mean(axis=0)
        r      = np.sqrt(((hpts[:, :2] - ctr[:2]) ** 2).sum(axis=1)).mean()
        ar     = _aspect_ratio(hpts[:, :2])
        centered2d = hpts[:, :2] - ctr[:2]
        eigvals    = np.sort(np.linalg.eigvalsh(np.cov(centered2d.T)))
        minor_r    = float(np.sqrt(max(eigvals[0], 0))) + 1.0
        entry  = {"center": ctr, "radius_mm": float(r), "minor_r_mm": minor_r,
                  "n_verts": len(comp), "verts": np.array(comp),
                  "aspect_ratio": float(ar)}

        if r_min <= r <= r_max and ar < mastic_aspect_min:
            rivets.append(entry)
        elif ar >= mastic_aspect_min and len(comp) >= mastic_n_verts_min:
            mastics.append(entry)

    rivets.sort(key=lambda x: -x["n_verts"])
    mastics.sort(key=lambda x: -x["n_verts"])
    return rivets, mastics


def build_mastic_boundary_tree(mastics, pts, buffer_r=3.0):
    """
    Build a 2D KDTree from mastic boundary vertices.
    Returns (tree, buffer_r) or (None, buffer_r) if no mastics.
    """
    if not mastics:
        return None, buffer_r
    all_verts = np.concatenate([pts[m["verts"]][:, :2] for m in mastics], axis=0)
    return cKDTree(all_verts), buffer_r


def build_panel_boundary_tree(faces, pts,
                               r_min=3.0, r_max=15.0,
                               mastic_aspect_min=4.0, mastic_n_verts_min=150):
    """
    Build a 2D KDTree from the outer panel boundary (scan perimeter).

    Detects all boundary loops, discards rivet-like and mastic-like ones,
    and returns a KDTree from the remaining (outer) boundary vertices.
    Use tree.query(xy)[0] < excl_r to exclude zero candidates near the edge.
    Returns (tree, None) or (None, None) if no boundary detected.
    """
    edges = np.sort(
        np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [0,2]]], axis=0),
        axis=1,
    )
    counts = Counter(map(tuple, edges))
    boundary = [e for e, c in counts.items() if c == 1]
    if not boundary:
        return None, None

    adj = defaultdict(set)
    for a, b in boundary:
        adj[a].add(b); adj[b].add(a)

    outer_verts = []
    visited = set()
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
        ar   = _aspect_ratio(hpts[:, :2])

        is_rivet  = r_min <= r <= r_max and ar < mastic_aspect_min
        is_mastic = ar >= mastic_aspect_min and len(comp) >= mastic_n_verts_min
        if not is_rivet and not is_mastic:
            outer_verts.extend(comp)

    if not outer_verts:
        return None, None
    return cKDTree(pts[outer_verts, :2]), None


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Virtual comparator v2 — local-flatness zeroing + crown sector analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ply", default="data/pc/Surface8_clean.ply",
                   help="Input PLY file")
    p.add_argument("--out-dir", default="comparator_output_v2",
                   help="Output directory")

    # Rivet detection
    p.add_argument("--hole-r-min", type=float, default=1.0,
                   help="Min rivet hole radius (mm)")
    p.add_argument("--hole-r-max", type=float, default=15.0,
                   help="Max rivet hole radius (mm)")

    # Physical comparator geometry
    p.add_argument("--feet-radius", type=float, default=13.7,
                   help="Circumradius of the 3-foot equilateral triangle (mm).")

    # Crown geometry
    p.add_argument("--r-inner", type=float, default=1.5,
                   help="Crown inner offset from hole edge (mm)")
    p.add_argument("--r-outer", type=float, default=None,
                   help="Crown outer offset from hole edge (mm). "
                        "Default: auto = half the mean edge-to-edge distance between rivets.")

    # Sector analysis
    p.add_argument("--n-sectors",      type=int, default=4,
                   help="Number of angular sectors")
    p.add_argument("--k-sectors",      type=int, default=2,
                   help="Number of worst sectors to average (default 2 = half-corona)")
    p.add_argument("--n-radial-bands", type=int, default=3,
                   help="Radial subdivisions per sector (inner→outer).")
    p.add_argument("--n-beams", type=int, default=16,
                   help="[beam] Number of probes placed at r_inner on the crown ring. "
                        "Each beam gets its own 3-foot plane. Used when --measure-mode beam. "
                        "Recommended: multiples of --n-sectors (e.g. 16 with 4 sectors).")
    p.add_argument("--zero-share-nn", type=int, default=2,
                   help="Share zero points among N nearest-neighbor rivets: among each rivet "
                        "and its N nearest neighbors the best zero (lowest score) is used for "
                        "all of them. Outer-row rivets with open-field zeros propagate inward. "
                        "0 = disabled.")
    p.add_argument("--beam-noise-sigma", type=float, default=0.0,
                   help="Gaussian XY noise added to beam target positions (mm). "
                        "0 = deterministic placement.")

    # Zero method
    p.add_argument("--zero-method", choices=["flatness", "deviation", "local-dev"],
                   default="local-dev",
                   help="local-dev: composite scoring — local polynomial deviation + distance "
                        "+ neighbourhood roughness (recommended). "
                        "flatness: minimum local surface roughness (mimics torch inspection). "
                        "deviation: legacy v1 — minimum global deviation-map residual.")
    p.add_argument("--zero-flatness-radius", type=float, default=3.0,
                   help="[flatness/local-dev] radius (mm) of the local neighbourhood used to "
                        "compute surface roughness at each zero candidate.")
    p.add_argument("--zero-probe-radius", type=float, default=4.0,
                   help="Probe disc radius at zero point (mm)")
    p.add_argument("--green-thresh", type=float, default=0.05,
                   help="Max |deviation| for reporting green vertex count (mm)")
    p.add_argument("--zero-nominal-thresh", type=float, default=0.10,
                   help="[deviation] Max |deviation| to qualify as nominal for zeroing (mm).")
    p.add_argument("--zero-from-edge-min", type=float, default=2.0,
                   help="Inner exclusion half-side from hole edge (mm) — square boundary")
    p.add_argument("--zero-from-edge-max", type=float, default=40.0,
                   help="Outer search half-side from hole edge (mm) — square boundary. "
                        "Also sets the local-dev poly-fit radius unless overridden.")

    # local-dev zeroing
    p.add_argument("--zero-pull-in-buffer", type=float, default=2.0,
                   help="[local-dev] Extra buffer beyond hole edge excluded from the local "
                        "polynomial fit (mm). Prevents pull-in from biasing the reference.")
    p.add_argument("--zero-local-fit-radius", type=float, default=None,
                   help="[local-dev] Radius (mm) of the local polynomial fit region "
                        "centred on the rivet. Defaults to --zero-from-edge-max so the "
                        "fit and search areas are always equal.")
    p.add_argument("--zero-dev-weight", type=float, default=0.5,
                   help="[local-dev] Weight of the local-deviation term in the composite penalty.")
    p.add_argument("--zero-dist-weight", type=float, default=0.3,
                   help="[local-dev] Weight of the distance-from-rivet term.")
    p.add_argument("--zero-rough-weight", type=float, default=0.2,
                   help="[local-dev] Weight of the neighbourhood-roughness term.")
    p.add_argument("--zero-dev-scale", type=float, default=0.05,
                   help="[local-dev] Deviation normalisation scale (mm). "
                        "1 unit = one green-threshold worth of deviation.")
    p.add_argument("--zero-rough-scale", type=float, default=0.003,
                   help="[local-dev] Roughness normalisation scale (mm). "
                        "Default 0.003 = 3 µm, typical clean-surface mesh roughness.")
    p.add_argument("--zero-excl-other-buffer", type=float, default=10.0,
                   help="Min clearance (mm) from any OTHER rivet's hole edge for a valid zero "
                        "candidate.  Total exclusion = other_hole_r + this value.  "
                        "Increase to prevent the zero from landing between two adjacent rivets.")
    p.add_argument("--boundary-excl-r", type=float, default=15.0,
                   help="Hard exclusion zone (mm) from the outer panel boundary (scan edge). "
                        "Zero candidates within this distance are discarded.")
    p.add_argument("--zero-open-weight", type=float, default=0.2,
                   help="Soft penalty weight for proximity to other rivets. "
                        "Higher = stronger preference for open fields away from rivets.")
    p.add_argument("--zero-open-scale", type=float, default=20.0,
                   help="Distance scale (mm) for open-field penalty: "
                        "d=open_scale → penalty halved; d>>open_scale → negligible.")
    p.add_argument("--zeroing-out-dir", default=None,
                   help="Directory for zeroing feasibility plots.  Defaults to --out-dir.")

    # Deviation map (needed only for measure-mode=deviation, zero-method=deviation,
    # or max-crown-dev-range filtering)
    p.add_argument("--grid-res",      type=float, default=0.4,  help="Grid resolution (mm)")
    p.add_argument("--smooth-radius", type=float, default=20.0, help="Gaussian smooth radius (mm)")
    p.add_argument("--poly-degree",   type=int,   default=4,    help="Polynomial degree for curvature removal")

    # Detection
    p.add_argument("--threshold", type=float, default=-0.2,
                   help="Defect threshold on k_worst_mean (mm). Values below are flagged.")

    # Colour zones
    p.add_argument("--warn-lo", type=float, default=0.14,
                   help="Orange zone start |pull-in| (mm)")
    p.add_argument("--warn-hi", type=float, default=0.21,
                   help="Red start |pull-in| (mm)")
    p.add_argument("--critical-hi", type=float, default=0.60,
                   help="Black (critical) start |pull-in| (mm)")

    # Foot validity
    p.add_argument("--foot-dist-max", type=float, default=3.0,
                   help="Max allowed distance (mm) between a foot target and the nearest mesh point.")
    p.add_argument("--feet-radius-min", type=float, default=6.0,
                   help="Minimum feet radius tried when full radius puts feet off-panel (mm)")

    # Measurement mode
    p.add_argument("--measure-mode", choices=["plane", "deviation", "local-poly", "per-point", "beam"],
                   default="plane",
                   help="plane: signed distance to 3-foot local plane (default). "
                        "deviation: deviation-map residual (global curvature removed). "
                        "local-poly: polynomial fit to the nominal ring outside the crown. "
                        "per-point: one comparator placement per crown point. "
                        "beam: N discrete probes at r_inner, each with its own 3-foot plane (--n-beams).")
    p.add_argument("--local-poly-fit-radius", type=float, default=30.0,
                   help="[local-poly] radius (mm) of the local fitting region around the rivet")
    p.add_argument("--local-poly-degree", type=int, default=2,
                   help="[local-poly] degree of the 2-D polynomial fit (default 2 = quadratic)")
    p.add_argument("--local-poly-method", choices=["exclude", "robust"], default="exclude",
                   help="[local-poly] exclude: fit only outside crown+1mm. "
                        "robust: fit all points, iteratively reject negative outliers.")
    p.add_argument("--max-crown-dev-range", type=float, default=None,
                   help="[plane] skip rivets whose crown deviation-map range exceeds this "
                        "value (mm). Requires the deviation map — enables it automatically.")

    # Output
    p.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=True,
                   help="Save interactive Plotly HTML")
    p.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True,
                   help="Save static matplotlib PNG")
    return p.parse_args()


# ── Geometry helpers (unchanged from v1) ─────────────────────────────────────
def compute_plane_at_point(center, pts, kdtree, feet_radius,
                           hole_centers=None, hole_radii=None,
                           mastic_centers=None, mastic_radii=None,
                           mastic_bound_tree=None, mastic_bound_r=3.0):
    n_vec, u_vec, v_vec = local_frame(center, pts, kdtree, feet_radius * 1.5)
    angles  = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    targets = center + feet_radius * (
        np.outer(np.cos(angles), u_vec) + np.outer(np.sin(angles), v_vec)
    )

    if hole_centers is not None:
        for hc, hr in zip(hole_centers, hole_radii):
            if any(np.linalg.norm(t[:2] - np.asarray(hc[:2])) < hr for t in targets):
                return None, None, float("inf")

    if mastic_centers is not None:
        for mc, mr in zip(mastic_centers, mastic_radii):
            if any(np.linalg.norm(t[:2] - np.asarray(mc[:2])) < mr for t in targets):
                return None, None, float("inf")

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

    plane_n, foot_pts, fmd = compute_plane_at_point(
        center, pts, kdtree, feet_radius_min,
        hole_centers=hole_centers, hole_radii=hole_radii,
        mastic_centers=mastic_centers, mastic_radii=mastic_radii,
        mastic_bound_tree=mastic_bound_tree, mastic_bound_r=mastic_bound_r,
    )
    ok = (plane_n is not None) and (fmd <= foot_dist_max)
    return plane_n, foot_pts, feet_radius_min, fmd, ok


def auto_r_outer(holes):
    if len(holes) < 2:
        return 7.0
    centers = np.array([h["center"][:2] for h in holes], dtype=np.float64)
    tree    = cKDTree(centers)
    nn_d, _ = tree.query(centers, k=2)
    d_avg   = float(nn_d[:, 1].mean())
    r_avg   = float(np.mean([h["radius_mm"] for h in holes]))
    r_out   = (d_avg - 2.0 * r_avg) / 2.0
    return max(r_out, 2.0)


# ── Local-flatness zeroing (v2) ───────────────────────────────────────────────
def _local_roughness_batch(pts, kdtree, center_idxs, radius):
    """
    For each vertex in center_idxs, fit a plane to all mesh vertices within
    `radius` mm and return the std of the signed distances to that plane.

    Low roughness = locally flat = good zero candidate.
    Returns inf for vertices with fewer than 6 neighbours (edge of mesh).
    """
    candidate_pts  = pts[center_idxs]
    neighbor_lists = kdtree.query_ball_point(candidate_pts, radius)
    roughness = np.full(len(center_idxs), float("inf"))
    for k, nbrs in enumerate(neighbor_lists):
        if len(nbrs) < 6:
            continue
        nbr_pts  = pts[nbrs]
        centroid = nbr_pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(nbr_pts - centroid, full_matrices=False)
        normal   = Vt[-1]                          # eigenvector of smallest eigenvalue
        residuals = (nbr_pts - centroid) @ normal
        roughness[k] = float(np.std(residuals))
    return roughness


def _subsample_angular(pts, rivet_center, idxs, max_candidates=256):
    """
    Reduce a large candidate set to at most max_candidates vertices while
    preserving coverage of all angular directions around the rivet.

    Strategy: divide into n_angular angular bins × n_radial radial steps;
    pick one vertex per (angle, radius) cell.  This ensures the flatness
    search is not biased toward whichever angular sector has the most mesh
    vertices.
    """
    if len(idxs) <= max_candidates:
        return idxs

    dx = pts[idxs, 0] - rivet_center[0]
    dy = pts[idxs, 1] - rivet_center[1]
    angles = np.arctan2(dy, dx)   # -pi .. pi
    d2d    = np.sqrt(dx ** 2 + dy ** 2)

    n_angular = 32
    n_radial  = max(1, max_candidates // n_angular)
    bin_w     = 2 * np.pi / n_angular

    selected = []
    for b in range(n_angular):
        lo = -np.pi + b * bin_w
        hi = lo + bin_w
        in_bin = (angles >= lo) & (angles < hi) if b < n_angular - 1 else angles >= lo
        bin_idxs = idxs[in_bin]
        if len(bin_idxs) == 0:
            continue
        d_bin = d2d[in_bin]
        order = np.argsort(d_bin)
        step  = max(1, len(order) // n_radial)
        selected.append(bin_idxs[order[::step][:n_radial]])

    return np.concatenate(selected) if selected else idxs[:max_candidates]


def _candidate_band(rivet_center, hole_radius, pts, kdtree,
                    from_edge_min, from_edge_max,
                    feet_radius,
                    other_centers, other_radii,
                    mastic_centers, mastic_radii,
                    mastic_bound_tree, mastic_bound_r,
                    zero_search, crown_buffer,
                    excl_other_buffer=10.0,
                    panel_boundary_tree=None, boundary_excl_r=15.0,
                    global_forbidden=None):
    """
    Return candidate zero vertex indices in the search band after all exclusions.

    If global_forbidden is provided (boolean array over all vertices, pre-computed
    as the union of ALL rivets + mastics + boundary exclusion zones), it is used
    directly — no per-rivet iteration needed.  Otherwise falls back to the legacy
    per-rivet exclusion loops.

    Search band (rivet-specific, always applied):
      - Outer square: half-side = hole_r + from_edge_max
      - Inner exclusion: candidates within the global_forbidden mask are removed,
        which implicitly excludes own rivet zone (hole_r + excl_other_buffer).
    """
    d_max = hole_radius + from_edge_max

    # Over-fetch with circumscribed circle, then clip to outer square
    idxs = np.array(kdtree.query_ball_point(rivet_center, d_max * 1.4143), dtype=np.int32)
    if len(idxs) == 0:
        return np.empty(0, dtype=np.int32)

    dx = pts[idxs, 0] - float(rivet_center[0])
    dy = pts[idxs, 1] - float(rivet_center[1])
    idxs = idxs[(np.abs(dx) <= d_max) & (np.abs(dy) <= d_max)]
    if len(idxs) == 0:
        return np.empty(0, dtype=np.int32)

    if global_forbidden is not None:
        # Global mask already encodes: all rivets + mastics + boundary
        return idxs[~global_forbidden[idxs]]

    # ── Legacy per-rivet exclusion (used when global_forbidden is not available) ──
    d_min = hole_radius + from_edge_min
    dx = pts[idxs, 0] - float(rivet_center[0])
    dy = pts[idxs, 1] - float(rivet_center[1])
    idxs = idxs[(np.abs(dx) >= d_min) | (np.abs(dy) >= d_min)]
    if len(idxs) == 0:
        return np.empty(0, dtype=np.int32)

    if other_centers is not None and len(other_centers) > 0:
        keep  = np.ones(len(idxs), dtype=bool)
        extra = crown_buffer if zero_search == "bounded" else 0.0
        for oc, or_ in zip(other_centers, other_radii):
            if np.linalg.norm(np.asarray(oc[:2]) - rivet_center[:2]) < 1e-3:
                continue
            half = or_ + extra + excl_other_buffer
            dx = np.abs(pts[idxs, 0] - float(oc[0]))
            dy = np.abs(pts[idxs, 1] - float(oc[1]))
            keep &= (dx >= half) | (dy >= half)
        idxs = idxs[keep]

    if mastic_centers is not None and len(mastic_centers) > 0:
        keep = np.ones(len(idxs), dtype=bool)
        for mc, mr in zip(mastic_centers, mastic_radii):
            d_mc = np.linalg.norm(pts[idxs, :2] - np.asarray(mc[:2]), axis=1)
            keep &= d_mc > (mr + feet_radius + 2.0)
        idxs = idxs[keep]

    if mastic_bound_tree is not None and len(idxs) > 0:
        d_boundary, _ = mastic_bound_tree.query(pts[idxs, :2])
        idxs = idxs[d_boundary >= (mastic_bound_r + feet_radius + 2.0)]

    if panel_boundary_tree is not None and len(idxs) > 0:
        d_edge, _ = panel_boundary_tree.query(pts[idxs, :2])
        idxs = idxs[d_edge >= boundary_excl_r]

    return idxs


def _open_field_penalty(pts_sub, other_centers, other_radii, open_scale):
    """
    Soft penalty for proximity to other rivets: 1/(1 + d_edge/open_scale).
    d_edge = distance to nearest other rivet edge (center - radius).
    Returns array of shape (len(pts_sub),) in [0, 1]; 0 = open field, 1 = on rivet edge.
    """
    if not other_centers:
        return np.zeros(len(pts_sub))
    d_edge = np.full(len(pts_sub), np.inf)
    for oc, or_ in zip(other_centers, other_radii):
        d = np.linalg.norm(pts_sub[:, :2] - np.asarray(oc[:2]), axis=1) - or_
        d_edge = np.minimum(d_edge, np.maximum(d, 0.0))
    return 1.0 / (1.0 + d_edge / open_scale)


def find_zero_point_flatness(rivet_center, hole_radius, pts, kdtree,
                              from_edge_min=20.0, from_edge_max=80.0,
                              feet_radius=13.7,
                              other_centers=None, other_radii=None,
                              mastic_centers=None, mastic_radii=None,
                              mastic_bound_tree=None, mastic_bound_r=3.0,
                              zero_search="free", crown_buffer=0.0,
                              flatness_radius=3.0, max_candidates=256,
                              excl_other_buffer=10.0,
                              panel_boundary_tree=None, boundary_excl_r=15.0,
                              w_open=0.3, open_scale=20.0,
                              global_forbidden=None):
    """
    Find the zero vertex with minimum local surface roughness.
    Score = roughness_norm + w_open * proximity_to_other_rivets
    Candidates near the panel boundary or between rivets are excluded (hard)
    or penalised (soft) to prefer open fields.
    Returns (zero_point, score) or (None, inf).
    """
    idxs = _candidate_band(
        rivet_center, hole_radius, pts, kdtree,
        from_edge_min, from_edge_max, feet_radius,
        other_centers, other_radii,
        mastic_centers, mastic_radii,
        mastic_bound_tree, mastic_bound_r,
        zero_search, crown_buffer,
        excl_other_buffer=excl_other_buffer,
        panel_boundary_tree=panel_boundary_tree, boundary_excl_r=boundary_excl_r,
        global_forbidden=global_forbidden,
    )
    if len(idxs) == 0:
        return None, float("inf")

    idxs     = _subsample_angular(pts, rivet_center, idxs, max_candidates)
    roughness = _local_roughness_batch(pts, kdtree, idxs, flatness_radius)
    rough_max = roughness[np.isfinite(roughness)].max() if np.any(np.isfinite(roughness)) else 1.0
    rough_norm = roughness / max(rough_max, 1e-9)

    open_pen = _open_field_penalty(pts[idxs], other_centers or [], other_radii or [], open_scale)
    score    = rough_norm + w_open * open_pen

    best = int(np.argmin(score))
    if not np.isinf(roughness[best]):
        return pts[idxs[best]].copy(), float(score[best])
    return None, float("inf")


def find_zero_point_deviation(rivet_center, hole_radius, pts, kdtree, deviation,
                               nominal_thresh=0.10, from_edge_min=20.0, from_edge_max=80.0,
                               feet_radius=13.7,
                               other_centers=None, other_radii=None,
                               mastic_centers=None, mastic_radii=None,
                               mastic_bound_tree=None, mastic_bound_r=3.0,
                               zero_search="free", crown_buffer=0.0,
                               excl_other_buffer=10.0,
                               panel_boundary_tree=None, boundary_excl_r=15.0,
                               w_open=0.3, open_scale=20.0,
                               global_forbidden=None):
    """
    Global-deviation zeroing: pick vertex with minimum |global deviation|,
    penalising proximity to other rivets to prefer open fields.
    Returns (zero_point, score) or (None, nan).
    """
    idxs = _candidate_band(
        rivet_center, hole_radius, pts, kdtree,
        from_edge_min, from_edge_max, feet_radius,
        other_centers, other_radii,
        mastic_centers, mastic_radii,
        mastic_bound_tree, mastic_bound_r,
        zero_search, crown_buffer,
        excl_other_buffer=excl_other_buffer,
        panel_boundary_tree=panel_boundary_tree, boundary_excl_r=boundary_excl_r,
    )
    if len(idxs) == 0:
        return None, float("nan")

    dev_abs  = np.abs(deviation[idxs])
    dev_norm = dev_abs / max(float(dev_abs.max()), 1e-9)
    open_pen = _open_field_penalty(pts[idxs], other_centers or [], other_radii or [], open_scale)
    score    = dev_norm + w_open * open_pen

    best = int(np.argmin(score))
    if dev_abs[best] < nominal_thresh:
        return pts[idxs[best]].copy(), float(score[best])
    return None, float("nan")



def zero_reading_at(zero_center, pts, kdtree, feet_radius, probe_radius,
                    foot_dist_max=3.0, feet_radius_min=6.0):
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
    idxs = np.array(kdtree.query_ball_point(zero_center, probe_radius), dtype=np.int32)
    if len(idxs) == 0:
        return None
    return float(np.median(deviation[idxs]))


# ── Local polynomial helpers (unchanged from v1) ──────────────────────────────
def _poly2d_basis(x, y, degree):
    cols = []
    for d in range(degree + 1):
        for i in range(d + 1):
            cols.append(x ** (d - i) * y ** i)
    return np.column_stack(cols)


def _robust_poly_fit(x, y, z, degree, n_iter=4, reject_sigma=1.5):
    A    = _poly2d_basis(x, y, degree)
    keep = np.ones(len(z), dtype=bool)
    min_pts = A.shape[1] * 2
    for _ in range(n_iter):
        Ak, zk = A[keep], z[keep]
        coeffs, _, _, _ = np.linalg.lstsq(Ak, zk, rcond=None)
        resid = z - A @ coeffs
        pos   = resid[resid > 0]
        sigma = float(pos.std()) if len(pos) > 3 else float(np.std(resid))
        new_keep = resid > -reject_sigma * sigma
        if new_keep.sum() < min_pts or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    coeffs, _, _, _ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
    return coeffs


# ── Local-deviation zeroing (v2 local-dev) ───────────────────────────────────

def compute_local_deviation_patch(rivet_center, pts, kdtree,
                                   fit_radius,
                                   all_hole_centers, all_hole_radii,
                                   pull_in_buffer=2.0,
                                   poly_degree=2,
                                   mastic_centers=None,
                                   mastic_radii=None):
    """
    Fit a 2-D polynomial to the surface within fit_radius of rivet_center,
    excluding pull-in zones (hole_r + pull_in_buffer around every rivet hole)
    and mastic patches.

    The fit uses only the nominally clean surface so that depressed regions
    (pull-ins, dents) do not bias the reference downward.

    Returns polynomial coefficients (centred on rivet_center) or None if the
    fit is underdetermined after exclusions.
    """
    # Direct square filter on the full point array — no KD-tree needed
    cx, cy = float(rivet_center[0]), float(rivet_center[1])
    all_cand = np.where(
        (np.abs(pts[:, 0] - cx) <= fit_radius) &
        (np.abs(pts[:, 1] - cy) <= fit_radius)
    )[0].astype(np.int32)
    if len(all_cand) == 0:
        return None

    keep = np.ones(len(all_cand), dtype=bool)
    for hc, hr in zip(all_hole_centers, all_hole_radii):
        d = np.linalg.norm(pts[all_cand, :2] - np.asarray(hc[:2]), axis=1)
        keep &= d > (hr + pull_in_buffer)
    if mastic_centers is not None:
        for mc, mr in zip(mastic_centers, mastic_radii):
            d = np.linalg.norm(pts[all_cand, :2] - np.asarray(mc[:2]), axis=1)
            keep &= d > mr

    fit_idx = all_cand[keep]
    min_pts = (poly_degree + 1) * (poly_degree + 2) // 2 * 3
    if len(fit_idx) < min_pts:
        return None

    xf = pts[fit_idx, 0] - cx
    yf = pts[fit_idx, 1] - cy
    coeffs, _, _, _ = np.linalg.lstsq(
        _poly2d_basis(xf, yf, poly_degree), pts[fit_idx, 2], rcond=None
    )
    return coeffs


def _eval_local_dev(pts_sub, rivet_center, coeffs, poly_degree):
    """Signed deviation (Z - Z_ref) for pts_sub using the local polynomial."""
    cx, cy = float(rivet_center[0]), float(rivet_center[1])
    z_ref = _poly2d_basis(pts_sub[:, 0] - cx, pts_sub[:, 1] - cy, poly_degree) @ coeffs
    return pts_sub[:, 2] - z_ref


def find_zero_point_local_dev(rivet_center, hole_radius, pts, kdtree,
                               from_edge_min=20.0, from_edge_max=80.0,
                               feet_radius=13.7,
                               other_centers=None, other_radii=None,
                               mastic_centers=None, mastic_radii=None,
                               mastic_bound_tree=None, mastic_bound_r=3.0,
                               zero_search="free", crown_buffer=0.0,
                               pull_in_buffer=2.0,
                               local_fit_radius=None,
                               poly_degree=2,
                               flatness_radius=3.0,
                               max_candidates=512,
                               w_dev=0.5, w_dist=0.3, w_rough=0.2,
                               dev_scale=0.05,
                               rough_scale=0.003,
                               excl_other_buffer=10.0,
                               panel_boundary_tree=None, boundary_excl_r=15.0,
                               w_open=0.2, open_scale=20.0,
                               global_forbidden=None):
    """
    Score zero candidates with a composite penalty:

        penalty = w_dev   * |local_dev(p)| / dev_scale
                + w_dist  * (d - d_min) / (d_max - d_min)
                + w_rough * roughness(p) / rough_scale
                + w_open  * 1/(1 + d_rivet_edge/open_scale)   ← prefer open fields

    Hard exclusions: other rivets, mastics, panel outer boundary (boundary_excl_r).
    Returns (zero_point, penalty) or (None, inf).
    """
    if local_fit_radius is None:
        local_fit_radius = 30.0

    all_centers_fit = list(other_centers or []) + [rivet_center]
    all_radii_fit   = list(other_radii   or []) + [hole_radius]

    coeffs = compute_local_deviation_patch(
        rivet_center, pts, kdtree,
        fit_radius=local_fit_radius,
        all_hole_centers=all_centers_fit,
        all_hole_radii=all_radii_fit,
        pull_in_buffer=pull_in_buffer,
        poly_degree=poly_degree,
        mastic_centers=mastic_centers,
        mastic_radii=mastic_radii,
    )

    idxs = _candidate_band(
        rivet_center, hole_radius, pts, kdtree,
        from_edge_min, from_edge_max, feet_radius,
        other_centers, other_radii,
        mastic_centers, mastic_radii,
        mastic_bound_tree, mastic_bound_r,
        zero_search, crown_buffer,
        excl_other_buffer=excl_other_buffer,
        panel_boundary_tree=panel_boundary_tree, boundary_excl_r=boundary_excl_r,
    )
    if len(idxs) == 0:
        return None, float("inf")

    idxs = _subsample_angular(pts, rivet_center, idxs, max_candidates)

    d2d    = np.linalg.norm(pts[idxs, :2] - rivet_center[:2], axis=1)
    d_span = max(from_edge_max - from_edge_min, 1.0)
    d_norm = np.clip((d2d - from_edge_min) / d_span, 0.0, 1.0)

    if coeffs is not None:
        dev = _eval_local_dev(pts[idxs], rivet_center, coeffs, poly_degree)
    else:
        dev = pts[idxs, 2] - np.median(pts[idxs, 2])

    dev_norm   = np.abs(dev) / dev_scale
    roughness  = _local_roughness_batch(pts, kdtree, idxs, flatness_radius)
    rough_norm = roughness / rough_scale
    open_pen   = _open_field_penalty(pts[idxs], other_centers or [], other_radii or [], open_scale)

    penalty = w_dev * dev_norm + w_dist * d_norm + w_rough * rough_norm + w_open * open_pen

    best = int(np.argmin(penalty))
    if np.isinf(roughness[best]):
        return None, float("inf")

    return pts[idxs[best]].copy(), float(penalty[best])


# ── Crown measurement (unchanged from v1) ────────────────────────────────────
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
                     local_poly_method="exclude",
                     n_beams=0,
                     beam_noise_sigma=0.0):
    plane_n, foot_pts, actual_feet_r, foot_max_dist, foot_ok = find_valid_plane(
        rivet_center, pts, kdtree, feet_radius, foot_dist_max, feet_radius_min,
        hole_centers=other_hole_centers, hole_radii=other_hole_radii,
        mastic_centers=mastic_centers, mastic_radii=mastic_radii,
        mastic_bound_tree=mastic_bound_tree, mastic_bound_r=mastic_bound_r,
    )
    if plane_n is None and measure_mode != "beam":
        return None
    if plane_n is None:
        actual_feet_r, foot_max_dist, foot_ok = feet_radius, float("inf"), True

    r_in  = hole_radius + r_inner_mm
    r_out = hole_radius + r_outer_mm

    if measure_mode == "beam":
        beam_angles = np.array([k * 2 * np.pi / n_beams for k in range(n_beams)])
        targets = rivet_center + r_in * np.column_stack(
            [np.cos(beam_angles), np.sin(beam_angles), np.zeros(n_beams)]
        )
        if beam_noise_sigma > 0:
            noise_xy = np.random.normal(0.0, beam_noise_sigma, (n_beams, 2))
            targets[:, :2] += noise_xy
        nn_d, nn_i = kdtree.query(targets)
        beam_mesh_pts = pts[nn_i]

        valid = nn_d < r_in * 0.5
        if other_hole_centers is not None:
            for hc, hr in zip(other_hole_centers, other_hole_radii):
                valid &= np.linalg.norm(beam_mesh_pts[:, :2] - np.asarray(hc[:2]), axis=1) > hr
        if mastic_centers is not None:
            for mc, mr in zip(mastic_centers, mastic_radii):
                valid &= np.linalg.norm(beam_mesh_pts[:, :2] - np.asarray(mc[:2]), axis=1) > mr

        all_hc = list(other_hole_centers or []) + [rivet_center]
        all_hr = list(other_hole_radii   or []) + [hole_radius]
        beam_dists = np.full(n_beams, np.nan)
        for k in range(n_beams):
            if not valid[k]:
                continue
            cp = beam_mesh_pts[k]
            pn, fp, _, _, fok_k = find_valid_plane(
                cp, pts, kdtree, feet_radius, foot_dist_max, feet_radius_min,
                hole_centers=all_hc, hole_radii=all_hr,
                mastic_centers=mastic_centers, mastic_radii=mastic_radii,
                mastic_bound_tree=mastic_bound_tree, mastic_bound_r=mastic_bound_r,
            )
            if pn is not None and fok_k:
                beam_dists[k] = float((cp - fp[0]) @ pn) - zero_offset

        bps = n_beams // n_sectors
        shared_boundary = (n_beams % n_sectors == 0)
        window = bps + 1 if shared_boundary else bps
        sector_worst = np.full(n_sectors, np.nan)
        sector_mean  = np.full(n_sectors, np.nan)
        sector_cnt   = np.zeros(n_sectors, dtype=int)
        for s in range(n_sectors):
            idxs = [(s * bps + j) % n_beams for j in range(window)]
            vals = beam_dists[idxs]
            fin  = vals[np.isfinite(vals)]
            if len(fin):
                sector_worst[s] = float(fin.min())
                sector_mean[s]  = float(fin.mean())
                sector_cnt[s]   = len(fin)

        sector_ok_b = np.isfinite(sector_worst)
        n_pop = int(sector_ok_b.sum())
        if n_pop < k_worst:
            return None

        sorted_w     = np.sort(sector_worst[sector_ok_b])
        k_worst_mean = float(sorted_w[:k_worst].mean())
        coh_mean     = float(sorted_w[k_worst:].mean()) if n_pop > k_worst else float("nan")
        fin_all      = beam_dists[np.isfinite(beam_dists)]

        return {
            "center":            rivet_center,
            "hole_r":            float(hole_radius),
            "feet_r":            float(actual_feet_r),
            "foot_max_dist":     float(foot_max_dist),
            "foot_ok":           True,
            "n_crown":           int(len(fin_all)),
            "crown_mean":        float(np.mean(fin_all))           if len(fin_all) else float("nan"),
            "crown_p10":         float(np.percentile(fin_all, 10)) if len(fin_all) else float("nan"),
            "crown_min":         float(np.min(fin_all))            if len(fin_all) else float("nan"),
            "k_worst_mean":      k_worst_mean,
            "worst_sector_mean": float(np.nanmin(sector_worst)),
            "sector_means":      [float(v) if np.isfinite(v) else None for v in sector_mean],
            "sector_counts":     sector_cnt.tolist(),
            "n_sectors_pop":     n_pop,
            "n_sectors_void":    int((~sector_ok_b).sum()),
            "n_sectors_below":   int((sector_worst[sector_ok_b] < 0).sum()),
            "consensus_band":    0,
            "coherence":         1.0,
            "coherent_mean":     coh_mean,
            "sector_grid":       [[float(v)] if np.isfinite(v) else [float("nan")]
                                  for v in sector_mean],
            "zero_offset":       float(zero_offset),
            "crown_dev_range":   float("nan"),
            "crown_dev_std":     float("nan"),
        }

    cand = np.array(kdtree.query_ball_point(rivet_center, r_out), dtype=np.int32)
    if len(cand) == 0:
        return None

    d2d        = np.linalg.norm(pts[cand, :2] - rivet_center[:2], axis=1)
    crown_mask = (d2d >= r_in) & (d2d <= r_out)
    crown_idx  = cand[crown_mask]
    crown_pts  = pts[crown_idx]
    d2d_crown  = d2d[crown_mask]

    valid_crown = np.ones(len(crown_pts), dtype=bool)
    if other_hole_centers is not None:
        for hc, hr in zip(other_hole_centers, other_hole_radii):
            d_h = np.linalg.norm(crown_pts[:, :2] - np.asarray(hc[:2]), axis=1)
            valid_crown &= d_h > hr
    if mastic_centers is not None:
        for mc, mr in zip(mastic_centers, mastic_radii):
            d_m = np.linalg.norm(crown_pts[:, :2] - np.asarray(mc[:2]), axis=1)
            valid_crown &= d_m > mr
    if mastic_bound_tree is not None:
        d_boundary, _ = mastic_bound_tree.query(crown_pts[:, :2])
        valid_crown &= d_boundary >= mastic_bound_r
    crown_idx  = crown_idx[valid_crown]
    crown_pts  = crown_pts[valid_crown]
    d2d_crown  = d2d_crown[valid_crown]

    if len(crown_pts) < max(n_sectors * n_radial_bands, 5):
        return None

    if measure_mode == "per-point":
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
        fit_cand = np.array(
            kdtree.query_ball_point(rivet_center, local_poly_fit_radius),
            dtype=np.int32,
        )
        fit_d2d = np.linalg.norm(pts[fit_cand, :2] - rivet_center[:2], axis=1)

        if local_poly_method == "exclude":
            r_out_fit = hole_radius + r_outer_mm
            fit_ok    = fit_d2d >= r_out_fit + 1.0
            fit_idx   = fit_cand[fit_ok]
            if other_hole_centers is not None:
                for hc, hr in zip(other_hole_centers, other_hole_radii):
                    d_h = np.linalg.norm(pts[fit_idx, :2] - np.asarray(hc[:2]), axis=1)
                    fit_idx = fit_idx[d_h > hr]
        else:
            fit_ok  = fit_d2d >= hole_radius
            fit_idx = fit_cand[fit_ok]
            if other_hole_centers is not None:
                for hc, hr in zip(other_hole_centers, other_hole_radii):
                    d_h = np.linalg.norm(pts[fit_idx, :2] - np.asarray(hc[:2]), axis=1)
                    fit_idx = fit_idx[d_h > hr]

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
            xc    = crown_pts[:, 0] - cx
            yc    = crown_pts[:, 1] - cy
            z_ref = _poly2d_basis(xc, yc, local_poly_degree) @ coeffs
            dists = crown_pts[:, 2] - z_ref
        else:
            dists = signed_distances(crown_pts, plane_n, foot_pts[0]) - zero_offset

    else:
        dists = signed_distances(crown_pts, plane_n, foot_pts[0]) - zero_offset

    dx = crown_pts[:, 0] - rivet_center[0]
    dy = crown_pts[:, 1] - rivet_center[1]
    angles_pts = np.arctan2(dy, dx)
    sector_w   = 2 * np.pi / n_sectors
    sector_ids = (np.floor((angles_pts + np.pi) / sector_w).astype(int) % n_sectors)

    band_w   = (r_out - r_in) / n_radial_bands
    band_ids = np.clip(
        ((d2d_crown - r_in) / band_w).astype(int), 0, n_radial_bands - 1
    )

    pts_per_sector   = np.array([(sector_ids == s).sum() for s in range(n_sectors)])
    expected_per_sec = len(crown_pts) / n_sectors
    sector_ok = pts_per_sector >= min_sector_coverage * expected_per_sec

    grid = np.full((n_sectors, n_radial_bands), np.nan)
    for s in range(n_sectors):
        if not sector_ok[s]:
            continue
        for b in range(n_radial_bands):
            m = (sector_ids == s) & (band_ids == b)
            if m.sum() >= 1:
                grid[s, b] = float(dists[m].mean())

    worst_val  = np.full(n_sectors, np.nan)
    worst_band = np.full(n_sectors, -1, dtype=int)
    for s in range(n_sectors):
        row = grid[s, :]
        if not np.all(np.isnan(row)):
            b = int(np.nanargmin(row))
            worst_band[s] = b
            worst_val[s]  = row[b]

    valid   = ~np.isnan(worst_val)
    n_pop   = int(valid.sum())
    k_act   = min(k_worst, n_pop)
    k_worst_mean = float(np.sort(worst_val[valid])[:k_act].mean()) if k_act > 0 else float("nan")

    valid_bands = worst_band[valid]
    if len(valid_bands) > 0:
        from collections import Counter
        consensus_band, cnt = Counter(valid_bands.tolist()).most_common(1)[0]
        coherence     = cnt / n_pop
        coherent_mean = float(np.nanmean(grid[:, consensus_band]))
    else:
        consensus_band = -1
        coherence      = 0.0
        coherent_mean  = float("nan")

    s_means  = np.nanmean(grid, axis=1)
    s_counts = np.array([int((sector_ids == s).sum()) for s in range(n_sectors)])

    if deviation_arr is not None and measure_mode == "plane":
        dev_crown       = deviation_arr[crown_idx]
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
    v = round(r["k_worst_mean"], 2)
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
    zero_method_str = getattr(args, "zero_method", "flatness")
    fig.suptitle(
        f"Virtual Comparator v2  —  {label}\n"
        f"Crown [{args.r_inner:.1f}..{args.r_outer:.1f}] mm  |  "
        f"feet R={args.feet_radius:.1f} mm  |  "
        f"K={args.k_sectors}/{args.n_sectors} sectors  |  "
        f"zero={zero_method_str}",
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
        r_in  = r["hole_r"] + args.r_inner
        beam_mode = getattr(args, "measure_mode", "") == "beam"
        r_crown = r_in if beam_mode else r["hole_r"] + args.r_outer

        ax.add_patch(plt.Circle((c[0], c[1]), r_crown,
                                color=col, fill=False, linewidth=0.8,
                                alpha=0.5, linestyle=ls))

        r_out     = r_crown
        sector_w  = 2 * np.pi / args.n_sectors
        if not beam_mode:
            band_w = (r_out - r_in) / args.n_radial_bands
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
    step = max(1, len(pts) // 300_000)
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    zero_method_str = getattr(args, "zero_method", "flatness")
    fig.suptitle(
        f"Punti di azzeramento  —  {label}  (zero-method={zero_method_str})\n"
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
            # Annotate roughness value at zero point (if available)
            zr = r.get("zero_roughness")
            if zr is not None and np.isfinite(zr):
                ax.annotate(f"{zr*1e3:.1f}µm", (zc[0], zc[1]),
                            fontsize=3, color="cyan", ha="left", va="bottom",
                            xytext=(2, 2), textcoords="offset points")
        else:
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


# ── Zero feasibility plot ─────────────────────────────────────────────────────
def make_zero_feasibility_plot(pts, holes, mastics, results, label, out_path, args):
    """
    Top-down map showing for each rivet:
      • Gray filled circle      = rivet hole (forbidden zone)
      • Red semi-transparent    = exclusion halo around every rivet
                                  (zero_excl_other_buffer mm from hole edge)
                                  — no other rivet can zero inside this ring
      • Orange dashed ring      = inner search boundary (zero_from_edge_min)
      • Blue dashed ring        = outer search boundary (zero_from_edge_max)
      • Cyan cross + dashed line = selected zero point → rivet
      • Orange fill             = mastic zones
    The valid zero region for a given rivet is the annular band [min, max]
    minus the red exclusion halos of all other rivets.
    """
    step = max(1, len(pts) // 300_000)
    fig, ax = plt.subplots(figsize=(14, 12))
    zero_method_str = getattr(args, "zero_method", "?")
    excl_buf        = getattr(args, "zero_excl_other_buffer", 10.0)
    d_min           = getattr(args, "zero_from_edge_min", 20.0)
    d_max           = getattr(args, "zero_from_edge_max", 80.0)
    fit_radius      = getattr(args, "zero_local_fit_radius", 30.0) or 30.0
    pull_in_buf     = getattr(args, "zero_pull_in_buffer", 2.0)
    is_local_dev    = (zero_method_str == "local-dev")
    fig.suptitle(
        f"Zero feasibility map  —  {label}  (method={zero_method_str})\n"
        f"Red halo = exclusion {excl_buf:.0f} mm  |  "
        f"Orange = search min {d_min:.0f} mm  |  Blue = search max {d_max:.0f} mm"
        + (f"  |  Green = poly-fit domain r={fit_radius:.0f} mm" if is_local_dev else ""),
        fontsize=10, fontweight="bold",
    )

    # Surface point cloud
    ax.scatter(pts[::step, 0], pts[::step, 1], c=pts[::step, 2],
               s=0.08, cmap="gray", rasterized=True, alpha=0.35, zorder=1)

    # Mastic zones
    for m in mastics:
        mv = pts[m["verts"]]
        ax.scatter(mv[:, 0], mv[:, 1], s=1.5, c="orange",
                   alpha=0.5, linewidths=0, zorder=2)

    # Per-rivet geometry
    for hole in holes:
        c  = hole["center"]
        hr = hole["radius_mm"]

        # Red exclusion halo — SQUARE (side = 2 × exclusion_radius)
        excl_r = hr + excl_buf
        ax.add_patch(plt.Rectangle(
            (c[0] - excl_r, c[1] - excl_r), 2 * excl_r, 2 * excl_r,
            color="tomato", fill=True, alpha=0.18, linewidth=0, zorder=3))
        ax.add_patch(plt.Rectangle(
            (c[0] - excl_r, c[1] - excl_r), 2 * excl_r, 2 * excl_r,
            color="red", fill=False, linewidth=0.6, linestyle="-", alpha=0.5, zorder=4))

        # Orange dashed square: inner search boundary
        s_min = hr + d_min
        ax.add_patch(plt.Rectangle(
            (c[0] - s_min, c[1] - s_min), 2 * s_min, 2 * s_min,
            color="darkorange", fill=False, linewidth=0.7,
            linestyle="--", alpha=0.45, zorder=4))

        # Blue dashed square: outer search boundary
        s_max = hr + d_max
        ax.add_patch(plt.Rectangle(
            (c[0] - s_max, c[1] - s_max), 2 * s_max, 2 * s_max,
            color="steelblue", fill=False, linewidth=0.5,
            linestyle=":", alpha=0.3, zorder=4))

        # Green square: local poly-fit domain (local-dev only)
        if is_local_dev:
            ax.add_patch(plt.Rectangle(
                (c[0] - fit_radius, c[1] - fit_radius), 2 * fit_radius, 2 * fit_radius,
                color="limegreen", fill=True, alpha=0.06, linewidth=0, zorder=2))
            ax.add_patch(plt.Rectangle(
                (c[0] - fit_radius, c[1] - fit_radius), 2 * fit_radius, 2 * fit_radius,
                color="limegreen", fill=False, linewidth=0.7,
                linestyle="-", alpha=0.5, zorder=4))
            # Inner pull-in exclusion — black circle (hole boundary + buffer)
            ax.add_patch(plt.Circle((c[0], c[1]), hr + pull_in_buf,
                                    color="black", fill=False, linewidth=0.8,
                                    linestyle="-", alpha=0.7, zorder=6))

        # Gray filled hole
        ax.add_patch(plt.Circle((c[0], c[1]), hr,
                                color="dimgray", fill=True, alpha=0.85,
                                linewidth=0, zorder=5))

    # Selected zero points
    zero_dists = []
    for r in results:
        c  = r["center"]
        zc = r.get("zero_center")
        col = _rivet_col(r, args)

        # Rivet circle coloured by result
        ax.add_patch(plt.Circle((c[0], c[1]), r["hole_r"],
                                color=col, fill=True, alpha=0.9,
                                linewidth=0, zorder=6))

        if zc is not None:
            d = float(np.linalg.norm(np.array(zc[:2]) - np.array(c[:2])))
            zero_dists.append(d)
            ax.plot([c[0], zc[0]], [c[1], zc[1]],
                    color="cyan", alpha=0.6, linewidth=0.7,
                    linestyle="--", zorder=7)
            ax.plot(zc[0], zc[1], marker="+", ms=6, color="cyan",
                    lw=0, markeredgewidth=1.4, zorder=8)
        else:
            # No zero found — mark rivet with gray X
            ax.plot(c[0], c[1], marker="x", ms=5, color="gray",
                    lw=0, markeredgewidth=1.2, zorder=8)

    n_no_zero = sum(1 for r in results if r.get("zero_center") is None)
    if zero_dists:
        title = (f"zero→rivet dist:  med={np.median(zero_dists):.1f} mm  "
                 f"max={max(zero_dists):.1f} mm  |  senza zero: {n_no_zero}")
    else:
        title = f"Nessun punto di zero trovato  (senza zero: {n_no_zero})"

    legend_elems = [
        mpatches.Patch(color="tomato",     alpha=0.5, label=f"Exclusion square ({excl_buf:.0f} mm from edge, lato={2*(excl_buf):.0f}+2r mm)"),
        mpatches.Patch(color="darkorange", alpha=0.6, label=f"Min search dist ({d_min:.0f} mm from edge)"),
        mpatches.Patch(color="steelblue",  alpha=0.5, label=f"Max search dist ({d_max:.0f} mm from edge)"),
        mpatches.Patch(color="limegreen",  label="Rivet OK"),
        mpatches.Patch(color="red",        label="Rivet DEFECT"),
        plt.Line2D([0], [0], color="cyan", lw=1.2, label="Zero point + line"),
    ]
    if is_local_dev:
        legend_elems.insert(3, mpatches.Patch(
            color="limegreen", alpha=0.4,
            label=f"Poly-fit domain r={fit_radius:.0f} mm"))
        legend_elems.insert(4, plt.Line2D(
            [0], [0], color="black", lw=0.8,
            label=f"Pull-in excl. r={pull_in_buf:.0f} mm from edge"))
    ax.legend(handles=legend_elems, fontsize=7, loc="upper right")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Interactive plot ──────────────────────────────────────────────────────────
def make_interactive_plot(pts, holes, mastics, results, threshold, label, out_path, args):
    import plotly.graph_objects as go

    step  = max(1, len(pts) // 30_000)
    pts_d = pts[::step]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=pts_d[:, 0], y=pts_d[:, 1], z=pts_d[:, 2],
        mode="markers",
        marker=dict(size=0.8, color=pts_d[:, 2],
                    colorscale="Greys", opacity=0.25),
        name="surface", showlegend=False,
    ))

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
        zr = r.get("zero_roughness")
        zr_str = f"{zr*1e3:.1f} µm" if (zr is not None and np.isfinite(zr)) else "—"
        hover.append(
            f"Hole #{r['hole_idx'] + 1}  r={r['hole_r']:.1f} mm{near_tag}<br>"
            f"<b>k_worst_mean : {r['k_worst_mean']:.3f} mm  [{zone}]</b><br>"
            f"crown_mean    : {r['crown_mean']:.3f} mm<br>"
            f"crown_p10     : {r['crown_p10']:.3f} mm<br>"
            f"crown_min     : {r['crown_min']:.3f} mm<br>"
            f"zero_offset   : {r['zero_offset']:.3f} mm<br>"
            f"zero_roughness: {zr_str}<br>"
            f"n_crown_pts   : {r['n_crown']}<br>"
            f"sectors < 0   : {r['n_sectors_below']}/{r['n_sectors_pop']}<br>"
            f"Sector means  : {s_str}"
        )

    marker_colors = [_rivet_col(r, args) for r in results]
    fig.add_trace(go.Scatter3d(
        x=[r["center"][0] for r in results],
        y=[r["center"][1] for r in results],
        z=[r["center"][2] for r in results],
        mode="markers",
        marker=dict(size=7, color=marker_colors,
                    line=dict(width=1, color="black")),
        text=hover, hoverinfo="text",
        name="rivets",
    ))

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

    if mastics:
        mp  = np.concatenate([pts[m["verts"]] for m in mastics])
        sm  = max(1, len(mp) // 5000)
        fig.add_trace(go.Scatter3d(
            x=mp[::sm, 0], y=mp[::sm, 1], z=mp[::sm, 2],
            mode="markers",
            marker=dict(size=2, color="orange", opacity=0.8),
            name=f"mastic ({len(mastics)})",
        ))

    fig.add_trace(go.Scatter3d(
        x=[r["center"][0] for r in results],
        y=[r["center"][1] for r in results],
        z=[r["center"][2] + 1.5 for r in results],
        mode="text",
        text=[f"{r['k_worst_mean']:.2f}" for r in results],
        textfont=dict(size=8, color="black"),
        showlegend=False, hoverinfo="skip",
    ))

    zero_method_str = getattr(args, "zero_method", "flatness")
    fig.update_layout(
        title=(f"{label} — Virtual Comparator v2  "
               f"(crown {args.r_inner}..{args.r_outer} mm, "
               f"feet R={args.feet_radius:.1f} mm, "
               f"K={args.k_sectors}/{args.n_sectors} sectors, "
               f"zero={zero_method_str} — "
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
        zr = r.get("zero_roughness")
        rows.append({
            "surface":              label,
            "hole_idx":             r["hole_idx"] + 1,
            "hole_r_mm":            round(r["hole_r"], 3),
            "k_worst_mean_mm":      round(r["k_worst_mean"], 4),
            "coherent_mean_mm":     round(r["coherent_mean"], 4) if np.isfinite(r["coherent_mean"]) else "",
            "coherence":            round(r["coherence"], 3),
            "consensus_band":       r["consensus_band"],
            "crown_mean_mm":        round(r["crown_mean"], 4),
            "crown_p10_mm":         round(r["crown_p10"], 4),
            "crown_min_mm":         round(r["crown_min"], 4),
            "worst_sector_mm":      round(r["worst_sector_mean"], 4),
            "n_sectors_below":      r["n_sectors_below"],
            "n_sectors_pop":        r["n_sectors_pop"],
            "zero_offset_mm":       round(r["zero_offset"], 4),
            "zero_roughness_mm":    round(zr, 5) if (zr is not None and np.isfinite(zr)) else "",
            "has_zero":             int(zc is not None),
            "n_crown_pts":          r["n_crown"],
            "foot_ok":              int(r["foot_ok"]),
            "actual_feet_r_mm":     round(r["feet_r"], 2),
            "foot_max_dist_mm":     round(r["foot_max_dist"], 3),
            "near_mastic":          int(r.get("near_mastic", False)),
            "defect":               int(r["k_worst_mean"] <= -0.2 and r["foot_ok"]),
            "cx":  round(r["center"][0], 2),
            "cy":  round(r["center"][1], 2),
            "cz":  round(r["center"][2], 2),
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
    if args.zero_local_fit_radius is None:
        args.zero_local_fit_radius = args.zero_from_edge_max
    os.makedirs(args.out_dir, exist_ok=True)
    label = os.path.splitext(os.path.basename(args.ply))[0]

    print(f"\n{'='*68}")
    print(f"  Virtual Comparator v2  —  {label}")
    print(f"  Feet R : {args.feet_radius:.1f} mm")
    r_outer_str = f"{args.r_outer:.1f}" if args.r_outer is not None else "auto"
    print(f"  Crown  : [{args.r_inner:.1f}, {r_outer_str}] mm from hole edge")
    print(f"  Sectors: K={args.k_sectors}/{args.n_sectors} worst   threshold={args.threshold} mm")
    print(f"  Zero   : method={args.zero_method}  "
          f"from edge [{args.zero_from_edge_min},{args.zero_from_edge_max}] mm")
    if args.zero_method == "flatness":
        print(f"           flatness_radius={args.zero_flatness_radius} mm")
    elif args.zero_method == "local-dev":
        fit_r_str = f"{args.zero_local_fit_radius:.0f}" if args.zero_local_fit_radius else "auto"
        print(f"           pull_in_buffer={args.zero_pull_in_buffer} mm  "
              f"local_fit_radius={fit_r_str} mm")
        print(f"           weights  dev={args.zero_dev_weight}  "
              f"dist={args.zero_dist_weight}  rough={args.zero_rough_weight}  "
              f"(scales dev={args.zero_dev_scale} mm  rough={args.zero_rough_scale} mm)")
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

    panel_boundary_tree, _ = build_panel_boundary_tree(
        faces, pts, r_min=args.hole_r_min, r_max=args.hole_r_max)
    if panel_boundary_tree is not None:
        print(f"    Panel boundary tree built  (excl-r={args.boundary_excl_r} mm)")
    else:
        print("    Panel boundary not detected")
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

    # ── Deviation map (only when required) ───────────────────────────────────
    need_deviation = (
        args.measure_mode == "deviation"
        or args.zero_method == "deviation"
        or args.max_crown_dev_range is not None
    )
    if need_deviation:
        print("[4] Computing deviation map …")
        deviation, _, _, _ = compute_deviation(
            pts, args.grid_res, args.smooth_radius, args.poly_degree,
        )
        n_green = int((np.abs(deviation) < args.green_thresh).sum())
        print(f"    deviation range: [{deviation.min():.3f}, {deviation.max():.3f}] mm")
        print(f"    green vertices (|dev|<{args.green_thresh}): "
              f"{n_green:,}  ({100*n_green/len(pts):.1f}%)")
    else:
        deviation = None
        print("[4] Deviation map skipped (zero-method=flatness, measure-mode=plane)")

    MASTIC_R_MAX    = 80.0
    all_centers     = [h["center"]    for h in holes]
    all_radii       = [h["radius_mm"] for h in holes]
    mastic_centers  = [m["center"]    for m in mastics if m["radius_mm"] < MASTIC_R_MAX]
    mastic_radii    = [m["radius_mm"] for m in mastics if m["radius_mm"] < MASTIC_R_MAX]
    print(f"    Mastic zones for exclusion: {len(mastic_centers)}/{len(mastics)} "
          f"(radius < {MASTIC_R_MAX} mm)")

    # Pre-compute per-rivet neighbour lists (used in both zeroing and measurement)
    other_centers_list = [[all_centers[j] for j in range(len(holes)) if j != i]
                          for i in range(len(holes))]
    other_radii_list   = [[all_radii[j]   for j in range(len(holes)) if j != i]
                          for i in range(len(holes))]

    # ── Pass 1: find zero points ──────────────────────────────────────────────
    print(f"\n[5a] Finding zero points for {len(holes)} rivets …")
    zero_pts       = []   # zero_pt  per rivet (may be None)
    zero_scores    = []   # zero_roughness/penalty per rivet

    for i, hole in enumerate(holes):
        c = hole["center"].copy()
        r = hole["radius_mm"]
        _zero_common = dict(
            from_edge_min=args.zero_from_edge_min,
            from_edge_max=args.zero_from_edge_max,
            feet_radius=args.feet_radius,
            other_centers=other_centers_list[i],
            other_radii=other_radii_list[i],
            mastic_centers=mastic_centers,
            mastic_radii=mastic_radii,
            excl_other_buffer=args.zero_excl_other_buffer,
            panel_boundary_tree=panel_boundary_tree,
            boundary_excl_r=args.boundary_excl_r,
            w_open=args.zero_open_weight,
            open_scale=args.zero_open_scale,
        )
        if args.zero_method == "local-dev":
            zp, zs = find_zero_point_local_dev(
                c, r, pts, kdtree,
                pull_in_buffer=args.zero_pull_in_buffer,
                local_fit_radius=args.zero_local_fit_radius,
                poly_degree=args.local_poly_degree,
                flatness_radius=args.zero_flatness_radius,
                w_dev=args.zero_dev_weight,
                w_dist=args.zero_dist_weight,
                w_rough=args.zero_rough_weight,
                dev_scale=args.zero_dev_scale,
                rough_scale=args.zero_rough_scale,
                **_zero_common,
            )
        elif args.zero_method == "flatness":
            zp, zs = find_zero_point_flatness(
                c, r, pts, kdtree,
                flatness_radius=args.zero_flatness_radius,
                **_zero_common,
            )
        else:
            zp, zs = find_zero_point_deviation(
                c, r, pts, kdtree, deviation,
                nominal_thresh=args.zero_nominal_thresh,
                **_zero_common,
            )
        zero_pts.append(zp)
        zero_scores.append(zs)

    # Post-validate: discard any zero that landed inside another rivet's forbidden zone
    excl_buf = args.zero_excl_other_buffer
    for i in range(len(holes)):
        if zero_pts[i] is None:
            continue
        zp = zero_pts[i]
        for j in range(len(holes)):
            if j == i:
                continue
            half = all_radii[j] + excl_buf
            if abs(zp[0] - all_centers[j][0]) < half and abs(zp[1] - all_centers[j][1]) < half:
                zero_pts[i]   = None
                zero_scores[i] = float("inf")
                break

    # ── Pass 2: share zero points among N nearest-neighbour rivets ────────────
    zero_shared_from = list(range(len(holes)))  # index of donor rivet

    if args.zero_share_nn > 0 and len(holes) > 1:
        print(f"[5b] Sharing zero points (nn={args.zero_share_nn}) …")
        centers_2d = np.array([h["center"][:2] for h in holes])
        rkd  = cKDTree(centers_2d)
        k    = min(args.zero_share_nn + 1, len(holes))
        _, nn_idxs = rkd.query(centers_2d, k=k)  # shape (n_rivets, k)

        donor_centers = np.array([h["center"][:2] for h in holes])
        donor_tree    = cKDTree(donor_centers)
        k_query       = min(args.zero_share_nn + 1, len(holes))
        all_c2        = np.array(all_centers, dtype=np.float64)[:, :2]
        all_r2        = np.array(all_radii,   dtype=np.float64)

        def _zero_valid_for(zp, ci, ri):
            """True if zp is in rivet i's search band and outside ALL rivets' forbidden zones."""
            adx = abs(zp[0] - ci[0])
            ady = abs(zp[1] - ci[1])
            d_min_i = ri + args.zero_from_edge_min
            d_max_i = ri + args.zero_from_edge_max
            if not (
                (adx <= d_max_i) and (ady <= d_max_i) and
                ((adx >= d_min_i) or (ady >= d_min_i)) and
                np.linalg.norm(zp[:2] - np.asarray(ci[:2])) > ri
            ):
                return False
            # Reject if inside any OTHER rivet's forbidden zone
            for cj, rj in zip(all_centers, all_radii):
                if abs(float(cj[0]) - float(ci[0])) < 1e-3 and abs(float(cj[1]) - float(ci[1])) < 1e-3:
                    continue  # skip self
                half = rj + excl_buf
                if abs(float(zp[0]) - float(cj[0])) < half and abs(float(zp[1]) - float(cj[1])) < half:
                    return False
            return True

        def _openness(zp):
            """Min clearance from zero point to any rivet edge (universal quality metric)."""
            return float((np.linalg.norm(all_c2 - zp[:2], axis=1) - all_r2).min())

        # Openness-based sharing: among self + N nearest neighbours pick the zero
        # with the highest openness (farthest from all rivets).
        # This ensures outer-row open-field zeros propagate inward to inner rows.
        for i in range(len(holes)):
            ci = holes[i]["center"]
            ri = holes[i]["radius_mm"]
            best_open = _openness(zero_pts[i]) if zero_pts[i] is not None else -float("inf")
            best_j    = i
            _, nbrs = donor_tree.query(ci[:2], k=k_query)
            for j in np.atleast_1d(nbrs):
                if j == i or zero_pts[j] is None:
                    continue
                if not _zero_valid_for(zero_pts[j], ci, ri):
                    continue
                op = _openness(zero_pts[j])
                if op > best_open:
                    best_open = op
                    best_j    = j
            if best_j != i:
                zero_pts[i]         = zero_pts[best_j]
                zero_scores[i]      = zero_scores[best_j]
                zero_shared_from[i] = best_j

        # Fallback: rivets still without zero → nearest valid donor in whole panel
        fallback_count = 0
        for i in range(len(holes)):
            if zero_pts[i] is not None:
                continue
            ci = holes[i]["center"]
            ri = holes[i]["radius_mm"]
            _, all_j = donor_tree.query(ci[:2], k=len(holes))
            for j in np.atleast_1d(all_j):
                if j == i or zero_pts[j] is None:
                    continue
                if _zero_valid_for(zero_pts[j], ci, ri):
                    zero_pts[i]         = zero_pts[j]
                    zero_scores[i]      = zero_scores[j]
                    zero_shared_from[i] = j
                    fallback_count     += 1
                    break

        n_shared = sum(1 for i, d in enumerate(zero_shared_from) if d != i)
        print(f"    Shared zeros: {n_shared}/{len(holes)} rivets use a neighbour's zero "
              f"(fallback: {fallback_count})")

        # Final post-validation: discard any zero (including shared) in a forbidden zone
        discarded = 0
        for i in range(len(holes)):
            if zero_pts[i] is None:
                continue
            zp = zero_pts[i]
            ci = holes[i]["center"]
            for j in range(len(holes)):
                if j == i:
                    continue
                half = all_radii[j] + excl_buf
                if abs(zp[0] - all_centers[j][0]) < half and abs(zp[1] - all_centers[j][1]) < half:
                    zero_pts[i]        = None
                    zero_scores[i]     = float("inf")
                    zero_shared_from[i] = i
                    discarded += 1
                    break
        if discarded:
            print(f"    Final post-validation: {discarded} zeros discarded (in forbidden zone after sharing)")

    # ── Pass 3: compute zero offsets and measure crowns ───────────────────────
    print(f"[5c] Measuring {len(holes)} rivets …")
    results, skipped, no_zero = [], 0, 0

    for i, hole in enumerate(holes):
        c      = hole["center"].copy()
        r      = hole["radius_mm"]
        zero_pt        = zero_pts[i]
        zero_roughness = zero_scores[i]

        if zero_pt is None:
            zero_offset = 0.0
            no_zero    += 1
        else:
            if args.measure_mode == "deviation" and deviation is not None:
                zo = deviation_zero_reading(zero_pt, pts, kdtree,
                                            deviation, args.zero_probe_radius)
            else:
                zo = zero_reading_at(zero_pt, pts, kdtree,
                                     args.feet_radius, args.zero_probe_radius)
            zero_offset = zo if zo is not None else 0.0

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
            other_hole_centers=other_centers_list[i],
            other_hole_radii=other_radii_list[i],
            mastic_centers=mastic_centers,
            mastic_radii=mastic_radii,
            deviation_arr=deviation,
            measure_mode=args.measure_mode,
            local_poly_fit_radius=args.local_poly_fit_radius,
            local_poly_degree=args.local_poly_degree,
            local_poly_method=args.local_poly_method,
            n_beams=args.n_beams,
            beam_noise_sigma=args.beam_noise_sigma,
        )
        if res is None:
            skipped += 1
            continue

        if (args.max_crown_dev_range is not None
                and deviation is not None
                and np.isfinite(res["crown_dev_range"])
                and res["crown_dev_range"] > args.max_crown_dev_range):
            skipped += 1
            continue

        res["hole_idx"]        = i
        res["zero_center"]     = zero_pt
        res["zero_roughness"]  = zero_roughness
        res["zero_shared_from"] = zero_shared_from[i]
        results.append(res)

    foot_off = sum(1 for r in results if not r["foot_ok"])
    n_shared_res = sum(1 for r in results if r["zero_shared_from"] != r["hole_idx"])
    print(f"    Measured: {len(results)}   Skipped: {skipped}   "
          f"No-zero: {no_zero}   Shared zero: {n_shared_res}   Foot off: {foot_off}")

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

    # Zero quality summary
    zr_vals = [r["zero_roughness"] for r in results
               if r.get("zero_roughness") is not None and np.isfinite(r["zero_roughness"])]
    if zr_vals:
        if args.zero_method == "local-dev":
            print(f"\n  Zero quality (composite penalty — lower = better zero):")
            print(f"    median={np.median(zr_vals):.3f}  "
                  f"p90={np.percentile(zr_vals, 90):.3f}  "
                  f"max={max(zr_vals):.3f}")
        else:
            print(f"\n  Zero roughness (flatness at zero point):")
            print(f"    median={np.median(zr_vals)*1e3:.2f} µm  "
                  f"p90={np.percentile(zr_vals, 90)*1e3:.2f} µm  "
                  f"max={max(zr_vals)*1e3:.2f} µm")

    print(f"\n  Rivets measured : {len(results)}")
    print(f"  Defects flagged : {len(defects)}  ({100*len(defects)/len(results):.1f}%)")
    if defects:
        worst = min(defects, key=lambda r: r["k_worst_mean"])
        wc    = worst["center"]
        print(f"  Worst pull-in   : {worst['k_worst_mean']:.4f} mm  "
              f"at ({wc[0]:.1f}, {wc[1]:.1f}, {wc[2]:.1f})")
    print(f"  No-zero fallback: {no_zero} rivets")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, f"{label}_comparator_v2.csv")
    save_csv(results, label, csv_path)
    print(f"\n  CSV  → {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    zeroing_dir = args.zeroing_out_dir if args.zeroing_out_dir else args.out_dir
    os.makedirs(zeroing_dir, exist_ok=True)

    if args.plots:
        print("[6] Static plot …")
        out_png = os.path.join(args.out_dir, f"{label}_comparator_v2.png")
        make_static_plot(pts, holes, mastics, results,
                         args.threshold, label, out_png, args)
        print(f"    → {out_png}")

        # Zeroing plots go to dedicated dir (separate from rivet-result plots)
        out_zero = os.path.join(zeroing_dir, f"{label}_zero_positions.png")
        make_zero_plot(pts, mastics, results, label, out_zero, args)
        print(f"    → {out_zero}")

        out_feasibility = os.path.join(zeroing_dir, f"{label}_zero_feasibility.png")
        make_zero_feasibility_plot(pts, holes, mastics, results, label,
                                   out_feasibility, args)
        print(f"    → {out_feasibility}")

    if args.interactive:
        print("[7] Interactive plot …")
        out_html = os.path.join(args.out_dir, f"{label}_comparator_v2.html")
        make_interactive_plot(pts, holes, mastics, results,
                              args.threshold, label, out_html, args)
        print(f"    → {out_html}")

    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()