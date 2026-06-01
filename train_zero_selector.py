"""
Train a zero-point selector model from annotated rivet measurements.

For each labeled rivet the script generates candidate zero points on an
angular-radial grid, extracts local geometric features for each, and
determines the "oracle" candidate — the one whose resulting k_worst_mean
is closest to the operator GT reading.  A Random Forest classifier is
then trained to recognise oracle-like candidates from their features alone,
enabling zero-point selection on unlabeled panels at inference time.

Expected GT CSV format
----------------------
surface,hole_idx,gt_mm,defect
Surface140_clean,1,-0.21,1
Surface140_clean,3,-0.08,0
...

  surface   : base filename (without .ply) matching what the comparator outputs
  hole_idx  : 1-based rivet index from the comparator output CSV
  gt_mm     : operator comparator reading in mm (negative = pull-in)
  defect    : 1 if pull-in defect, 0 if OK  (used for diagnostic summary)

PLY files are looked up in --mesh-dir first (mesh3D) then --pc-dir (pc).

Usage
-----
  python train_zero_selector.py --gt-csv data/gt_labeled.csv
  python train_zero_selector.py --gt-csv data/gt_labeled.csv \\
      --model-out models/zero_selector.pkl \\
      --n-angular 20 --n-radial 6 --R-feat 18
"""

import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser(
        description="Train zero-point selector (Random Forest) from operator GT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input
    p.add_argument("--gt-csv",   required=True,
                   help="CSV with columns: surface,hole_idx,gt_mm,defect")
    p.add_argument("--mesh-dir", default="data/mesh3D",
                   help="Directory with mesh3D PLY files (checked first)")
    p.add_argument("--pc-dir",   default="data/pc",
                   help="Directory with pc PLY files (fallback)")

    # Candidate generation
    p.add_argument("--from-edge-min", type=float, default=13.0,
                   help="Min distance from hole edge to zero centre (mm)")
    p.add_argument("--from-edge-max", type=float, default=60.0,
                   help="Max distance from hole edge to zero centre (mm)")
    p.add_argument("--n-angular",  type=int,   default=20,
                   help="Angular grid divisions (candidates per radial ring)")
    p.add_argument("--n-radial",   type=int,   default=6,
                   help="Radial rings between from-edge-min and from-edge-max")
    p.add_argument("--feet-radius", type=float, default=13.7)

    # Feature extraction
    p.add_argument("--R-feat", type=float, default=15.0,
                   help="Feature patch radius in mm")

    # Hole detection
    p.add_argument("--hole-r-min", type=float, default=1.0)
    p.add_argument("--hole-r-max", type=float, default=15.0)

    # Crown geometry (for k_worst_mean computation)
    p.add_argument("--r-inner",       type=float, default=1.5)
    p.add_argument("--r-outer",       type=float, default=None,
                   help="Crown outer offset (mm). Default: auto from rivet spacing.")
    p.add_argument("--n-sectors",     type=int,   default=4)
    p.add_argument("--k-sectors",     type=int,   default=2)
    p.add_argument("--n-radial-bands", type=int,  default=3)
    p.add_argument("--zero-probe-radius", type=float, default=4.0)
    p.add_argument("--foot-dist-max",    type=float, default=3.0)
    p.add_argument("--feet-radius-min",  type=float, default=6.0)

    # Model
    p.add_argument("--model-out", default="models/zero_selector.pkl",
                   help="Output path for the trained model (joblib pickle)")
    p.add_argument("--n-estimators", type=int, default=300,
                   help="Number of trees in the Random Forest")
    p.add_argument("--max-depth",    type=int, default=None,
                   help="Max tree depth (None = unlimited)")

    # Evaluation
    p.add_argument("--leave-one-out", action="store_true",
                   help="Report leave-one-surface-out cross-validation accuracy")
    return p.parse_args()


# ── PLY lookup ────────────────────────────────────────────────────────────────
def _find_ply(surface_label, mesh_dir, pc_dir):
    for d in [mesh_dir, pc_dir]:
        for fname in os.listdir(d):
            if not fname.endswith(".ply"):
                continue
            stem = os.path.splitext(fname)[0]
            if stem == surface_label:
                return os.path.join(d, fname)
    return None


# ── Surface data loading ──────────────────────────────────────────────────────
_surface_cache = {}    # label → (pts, kdtree, holes, mastics, r_outer)


def _load_surface(surface_label, mesh_dir, pc_dir, args):
    if surface_label in _surface_cache:
        return _surface_cache[surface_label]

    from virtual_comparator import find_holes, build_mastic_boundary_tree
    from virtual_comparator_v2 import auto_r_outer

    ply = _find_ply(surface_label, mesh_dir, pc_dir)
    if ply is None:
        raise FileNotFoundError(
            f"PLY not found for surface '{surface_label}' "
            f"in {mesh_dir!r} or {pc_dir!r}"
        )
    print(f"  Loading {os.path.basename(ply)} …")
    mesh   = trimesh.load(ply, process=False)
    pts    = np.asarray(mesh.vertices, dtype=np.float64)
    faces  = np.asarray(mesh.faces,    dtype=np.int32)
    kdtree = cKDTree(pts)

    holes, mastics = find_holes(faces, pts,
                                r_min=args.hole_r_min, r_max=args.hole_r_max)
    r_outer = args.r_outer if args.r_outer is not None else auto_r_outer(holes)

    mastic_bound_tree, mastic_bound_r = build_mastic_boundary_tree(
        mastics, pts, buffer_r=3.0
    )

    result = (pts, kdtree, holes, mastics, r_outer, mastic_bound_tree, mastic_bound_r)
    _surface_cache[surface_label] = result
    return result


# ── Training sample builder for one rivet ────────────────────────────────────
def build_rivet_samples(
    surface_label, hole_idx_1based, gt_mm,
    mesh_dir, pc_dir, args
):
    """
    For one labeled rivet, generate all candidates, extract their features,
    compute predicted k_worst_mean for each, and return a list of
    (feature_array, is_oracle) tuples.

    The oracle is the candidate whose predicted k_worst_mean is closest
    to the operator GT reading.

    Returns [] if the rivet can't be processed (too few candidates, etc.).
    """
    from virtual_comparator_v2 import (
        measure_crown_v1, zero_reading_at,
    )
    from zero_point_features import (
        generate_zero_candidates, extract_zero_candidate_features,
        features_to_array, FEATURE_NAMES,
    )
    from virtual_comparator import build_mastic_boundary_tree

    try:
        pts, kdtree, holes, mastics, r_outer, mastic_bound_tree, mastic_bound_r = \
            _load_surface(surface_label, mesh_dir, pc_dir, args)
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        return []

    idx_0 = hole_idx_1based - 1
    if idx_0 < 0 or idx_0 >= len(holes):
        print(f"  WARNING: hole_idx {hole_idx_1based} out of range "
              f"(surface has {len(holes)} holes)")
        return []

    hole        = holes[idx_0]
    rivet_center = hole["center"].copy()
    hole_radius  = hole["radius_mm"]

    MASTIC_R_MAX   = 80.0
    all_centers    = [h["center"]    for h in holes]
    all_radii      = [h["radius_mm"] for h in holes]
    mastic_centers = [m["center"]    for m in mastics if m["radius_mm"] < MASTIC_R_MAX]
    mastic_radii   = [m.get("minor_r_mm", m["radius_mm"])
                      for m in mastics if m["radius_mm"] < MASTIC_R_MAX]
    other_centers  = [all_centers[j] for j in range(len(holes)) if j != idx_0]
    other_radii    = [all_radii[j]   for j in range(len(holes)) if j != idx_0]

    # Optional panel boundary tree: use all mastic boundary points
    # including the large-radius panel perimeter
    panel_bnd_pts  = [pts[m["verts"]] for m in mastics if m["radius_mm"] >= MASTIC_R_MAX]
    panel_bnd_tree = None
    if panel_bnd_pts:
        panel_bnd_all = np.concatenate(panel_bnd_pts)
        step = max(1, len(panel_bnd_all) // 2000)
        panel_bnd_tree = cKDTree(panel_bnd_all[::step, :2])

    # ── Baseline crown measurement (zero_offset = 0) ──────────────────────────
    # k_worst_mean with any zero candidate = k_worst_mean_raw - zero_reading
    # This holds exactly because zero_offset shifts all crown distances uniformly.
    res_raw = measure_crown_v1(
        rivet_center, hole_radius, pts, kdtree,
        feet_radius=args.feet_radius,
        r_inner_mm=args.r_inner,
        r_outer_mm=r_outer,
        n_sectors=args.n_sectors,
        k_worst=args.k_sectors,
        n_radial_bands=args.n_radial_bands,
        zero_offset=0.0,
        foot_dist_max=args.foot_dist_max,
        feet_radius_min=args.feet_radius_min,
        other_hole_centers=other_centers,
        other_hole_radii=other_radii,
        mastic_centers=mastic_centers,
        mastic_radii=mastic_radii,
        mastic_bound_tree=mastic_bound_tree,
        mastic_bound_r=mastic_bound_r,
    )
    if res_raw is None:
        print(f"  WARNING: crown measurement failed for {surface_label} hole {hole_idx_1based}")
        return []

    k_raw = res_raw["k_worst_mean"]

    # ── Generate candidates ───────────────────────────────────────────────────
    candidates = generate_zero_candidates(
        rivet_center, hole_radius, pts, kdtree,
        from_edge_min=args.from_edge_min,
        from_edge_max=args.from_edge_max,
        n_angular=args.n_angular,
        n_radial=args.n_radial,
        other_centers=other_centers,
        other_radii=other_radii,
        mastic_centers=mastic_centers,
        mastic_radii=mastic_radii,
        feet_radius=args.feet_radius,
    )
    if len(candidates) == 0:
        print(f"  WARNING: no candidates for {surface_label} hole {hole_idx_1based}")
        return []

    # ── Features + predicted k_worst_mean for each candidate ─────────────────
    feat_arrays = []
    errors      = []

    for p in candidates:
        feat = extract_zero_candidate_features(
            p, rivet_center, hole_radius, pts, kdtree,
            other_centers, other_radii,
            mastic_centers, mastic_radii,
            panel_boundary_tree=panel_bnd_tree,
            R_feat=args.R_feat,
        )
        if feat is None:
            continue

        z_reading = zero_reading_at(
            p, pts, kdtree,
            args.feet_radius, args.zero_probe_radius,
            foot_dist_max=args.foot_dist_max,
            feet_radius_min=args.feet_radius_min,
        )
        if z_reading is None:
            continue

        predicted_k = k_raw - z_reading
        error = abs(predicted_k - gt_mm)

        feat_arrays.append(features_to_array(feat))
        errors.append(error)

    if not feat_arrays:
        return []

    feat_arrays = np.array(feat_arrays)
    errors      = np.array(errors)

    # Oracle = candidate closest to GT reading
    oracle_idx = int(np.argmin(errors))
    labels     = np.zeros(len(feat_arrays), dtype=np.int32)
    labels[oracle_idx] = 1

    print(f"    {surface_label} hole {hole_idx_1based:>3}:  "
          f"{len(feat_arrays):>3} candidates  "
          f"gt={gt_mm:+.3f}  "
          f"oracle_err={errors[oracle_idx]:.4f} mm  "
          f"oracle_feat[planarity_std]={feat_arrays[oracle_idx, 0]:.4f}")

    return list(zip(feat_arrays.tolist(), labels.tolist(), [errors[i] for i in range(len(errors))]))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)

    # Load GT CSV
    gt_rows = []
    with open(args.gt_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("gt_mm"):
                continue
            gt_rows.append({
                "surface":  row["surface"],
                "hole_idx": int(row["hole_idx"]),
                "gt_mm":    float(row["gt_mm"]),
                "defect":   int(row.get("defect", 0)),
            })

    print(f"\nGT rows loaded: {len(gt_rows)}")
    surfaces = sorted({r["surface"] for r in gt_rows})
    print(f"Surfaces: {surfaces}\n")

    # Build training dataset
    all_feats  = []
    all_labels = []
    all_errors = []
    per_surface = {s: {"feats": [], "labels": [], "errors": []} for s in surfaces}

    for row in gt_rows:
        print(f"Processing {row['surface']} hole {row['hole_idx']} …")
        samples = build_rivet_samples(
            row["surface"], row["hole_idx"], row["gt_mm"],
            args.mesh_dir, args.pc_dir, args,
        )
        for feat, label, error in samples:
            all_feats.append(feat)
            all_labels.append(label)
            all_errors.append(error)
            per_surface[row["surface"]]["feats"].append(feat)
            per_surface[row["surface"]]["labels"].append(label)
            per_surface[row["surface"]]["errors"].append(error)

    if not all_feats:
        print("\nERROR: no training samples generated. Check GT CSV and PLY paths.")
        return

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    import joblib

    X = np.array(all_feats)
    y = np.array(all_labels)

    print(f"\nTraining dataset: {len(X)} samples  "
          f"({y.sum()} oracle / {(~y.astype(bool)).sum()} non-oracle)")
    print(f"Oracle fraction: {y.mean():.3f}  "
          f"(1:{int(1/y.mean())-1} imbalance)")

    # ── Train ─────────────────────────────────────────────────────────────────
    clf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf",     RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    clf.fit(X, y)

    # Training set metrics
    y_pred = clf.predict(X)
    print(f"\nTraining classification report:")
    print(classification_report(y, y_pred, target_names=["non-oracle", "oracle"],
                                 zero_division=0))

    # Feature importances
    from zero_point_features import FEATURE_NAMES
    importances = clf.named_steps["rf"].feature_importances_
    print("Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, importances),
                             key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {name:<25} {imp:.4f}  {bar}")

    # ── Leave-one-surface-out cross-validation ────────────────────────────────
    if args.leave_one_out and len(surfaces) > 1:
        print("\nLeave-one-surface-out cross-validation …")
        from sklearn.metrics import roc_auc_score

        loo_results = []
        for test_surf in surfaces:
            train_feats, train_labels = [], []
            test_feats,  test_labels  = [], []
            for s, sd in per_surface.items():
                if s == test_surf:
                    test_feats.extend(sd["feats"])
                    test_labels.extend(sd["labels"])
                else:
                    train_feats.extend(sd["feats"])
                    train_labels.extend(sd["labels"])

            if not train_feats or not test_feats:
                continue

            clf_loo = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("rf",     RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    class_weight="balanced",
                    random_state=42, n_jobs=-1,
                )),
            ])
            clf_loo.fit(np.array(train_feats), np.array(train_labels))
            proba = clf_loo.predict_proba(np.array(test_feats))[:, 1]
            y_t   = np.array(test_labels)
            try:
                auc = roc_auc_score(y_t, proba)
            except Exception:
                auc = float("nan")

            # Per-rivet top-1 accuracy: does the oracle have the highest score?
            # We need to group by rivet. Approximate: oracle rank in the full set.
            oracle_rank = int(np.argsort(-proba)[0] == np.argmax(y_t))
            loo_results.append((test_surf, auc))
            print(f"  LOO {test_surf:<30}  AUC={auc:.3f}")

        mean_auc = np.nanmean([r[1] for r in loo_results])
        print(f"  Mean LOO AUC: {mean_auc:.3f}")

    # ── Save model ────────────────────────────────────────────────────────────
    joblib.dump(clf, args.model_out)
    print(f"\nModel saved → {args.model_out}")
    print(f"  Features: {FEATURE_NAMES}")
    print(f"  Trees: {args.n_estimators}   R_feat used at training: {args.R_feat} mm")
    print(f"  Imputation: median (handles nan panel_margin gracefully)")


if __name__ == "__main__":
    main()