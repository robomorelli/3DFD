"""
Train a zero-point selector model from rivet defect labels.

GT CSV format (from make_label_images.py, after annotation):
  surface,hole_idx,cx,cy,cz,label
  Surface140_clean,1,25.3,-45.2,77.1,0
  Surface140_clean,5,25.3,-62.1,77.0,2

  label:  0 = OK (no pull-in)
          1 = warn (marginal pull-in)
          2 = defect (clear pull-in)

The training objective:
  - For label >= 1: oracle = candidate with MOST NEGATIVE k_worst_mean
    (the zero on the true nominal surface reveals the full pull-in depth)
  - For label == 0: oracle = candidate with k_worst_mean CLOSEST TO ZERO
    (nominal surface shows no depression)

k_worst_mean per candidate is computed efficiently:
  k_worst(z) = k_worst_raw - zero_reading_at(z)
This is exact: zero_offset shifts all crown distances uniformly, so the
sector-worst selection and its mean shift by the same constant.

After training, optionally calibrate classification thresholds from the
labeled data using a simple grid search (--calibrate-thresholds).

Usage:
  python train_zero_selector.py --gt-csv labels/all_labels.csv
  python train_zero_selector.py --gt-csv labels/all_labels.csv \\
      --model-out models/zero_selector.pkl --calibrate-thresholds \\
      --leave-one-out
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    p = argparse.ArgumentParser(
        description="Train zero-point selector (Random Forest) from rivet labels (0/1/2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input
    p.add_argument("--gt-csv",   required=True,
                   help="CSV: surface,hole_idx,[cx,cy,cz,]label  (from make_label_images.py)")
    p.add_argument("--mesh-dir", default="data/mesh3D")
    p.add_argument("--pc-dir",   default="data/pc")

    # Which labels to use for training
    p.add_argument("--skip-label-0",  action="store_true",
                   help="Exclude OK rivets (label=0) from training — "
                        "reduces class imbalance but loses specificity signal")

    # Candidate generation
    p.add_argument("--from-edge-min", type=float, default=13.0)
    p.add_argument("--from-edge-max", type=float, default=60.0)
    p.add_argument("--n-angular",     type=int,   default=20)
    p.add_argument("--n-radial",      type=int,   default=6)
    p.add_argument("--feet-radius",   type=float, default=13.7)

    # Feature extraction
    p.add_argument("--R-feat",  type=float, default=15.0,
                   help="Feature patch radius in mm")

    # Hole detection
    p.add_argument("--hole-r-min", type=float, default=1.0)
    p.add_argument("--hole-r-max", type=float, default=15.0)

    # Crown geometry
    p.add_argument("--r-inner",           type=float, default=1.5)
    p.add_argument("--r-outer",           type=float, default=None)
    p.add_argument("--n-sectors",         type=int,   default=4)
    p.add_argument("--k-sectors",         type=int,   default=2)
    p.add_argument("--n-radial-bands",    type=int,   default=3)
    p.add_argument("--zero-probe-radius", type=float, default=4.0)
    p.add_argument("--foot-dist-max",     type=float, default=3.0)
    p.add_argument("--feet-radius-min",   type=float, default=6.0)

    # Model
    p.add_argument("--model-out",    default="models/zero_selector.pkl")
    p.add_argument("--n-estimators", type=int,  default=300)
    p.add_argument("--max-depth",    type=int,  default=None)

    # Evaluation
    p.add_argument("--calibrate-thresholds", action="store_true",
                   help="After training, find warn/defect thresholds that "
                        "maximise F1 on labeled data")
    p.add_argument("--leave-one-out", action="store_true",
                   help="Leave-one-surface-out cross-validation")
    return p.parse_args()


# ── Surface cache ─────────────────────────────────────────────────────────────
_surface_cache = {}


def _find_ply(surface_label, mesh_dir, pc_dir):
    for d in [mesh_dir, pc_dir]:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.endswith(".ply") and os.path.splitext(fname)[0] == surface_label:
                return os.path.join(d, fname)
    return None


def _load_surface(surface_label, mesh_dir, pc_dir, args):
    if surface_label in _surface_cache:
        return _surface_cache[surface_label]

    from virtual_comparator import find_holes, build_mastic_boundary_tree
    from virtual_comparator_v2 import auto_r_outer

    ply = _find_ply(surface_label, mesh_dir, pc_dir)
    if ply is None:
        raise FileNotFoundError(
            f"PLY not found for '{surface_label}' in {mesh_dir!r} or {pc_dir!r}"
        )
    print(f"  Loading {os.path.basename(ply)} …", flush=True)
    mesh   = trimesh.load(ply, process=False)
    pts    = np.asarray(mesh.vertices, dtype=np.float64)
    faces  = np.asarray(mesh.faces,    dtype=np.int32)
    kdtree = cKDTree(pts)
    holes, mastics = find_holes(faces, pts,
                                r_min=args.hole_r_min, r_max=args.hole_r_max)
    r_outer = args.r_outer if args.r_outer is not None else auto_r_outer(holes)
    mastic_bound_tree, mastic_bound_r = build_mastic_boundary_tree(
        mastics, pts, buffer_r=3.0)

    result = (pts, kdtree, holes, mastics, r_outer, mastic_bound_tree, mastic_bound_r)
    _surface_cache[surface_label] = result
    return result


# ── Per-rivet sample generation ───────────────────────────────────────────────
def build_rivet_samples(surface_label, hole_idx_1based, gt_label,
                        mesh_dir, pc_dir, args):
    """
    Generate (feature_array, is_oracle) training pairs for one labeled rivet.

    Oracle selection:
      label >= 1  →  candidate with MOST NEGATIVE k_worst_mean
      label == 0  →  candidate with k_worst_mean CLOSEST TO ZERO

    Returns list of (feat_array, label_int, k_worst_mean_float) tuples.
    """
    from virtual_comparator_v2 import measure_crown_v1, zero_reading_at
    from virtual_comparator import build_mastic_boundary_tree
    from zero_point_features import (
        generate_zero_candidates, extract_zero_candidate_features,
        features_to_array,
    )

    try:
        pts, kdtree, holes, mastics, r_outer, mastic_bound_tree, mastic_bound_r = \
            _load_surface(surface_label, mesh_dir, pc_dir, args)
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        return []

    idx_0 = hole_idx_1based - 1
    if idx_0 < 0 or idx_0 >= len(holes):
        print(f"  WARNING: hole_idx={hole_idx_1based} out of range "
              f"({len(holes)} holes in {surface_label})")
        return []

    hole         = holes[idx_0]
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

    # Panel boundary KD-tree (large-radius mastics = panel perimeter)
    panel_bnd_pts = [pts[m["verts"]] for m in mastics if m["radius_mm"] >= MASTIC_R_MAX]
    panel_bnd_tree = None
    if panel_bnd_pts:
        panel_bnd_all = np.concatenate(panel_bnd_pts)
        step = max(1, len(panel_bnd_all) // 2000)
        panel_bnd_tree = cKDTree(panel_bnd_all[::step, :2])

    # ── Crown measurement with zero_offset=0 (baseline) ──────────────────────
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
        return []

    # ── Features + predicted k_worst for each candidate ───────────────────────
    feat_rows = []
    k_worsts  = []

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

        feat_rows.append(features_to_array(feat))
        k_worsts.append(k_raw - z_reading)

    if not feat_rows:
        return []

    feat_rows = np.array(feat_rows)
    k_worsts  = np.array(k_worsts)

    # ── Oracle selection ──────────────────────────────────────────────────────
    # Only defect/warn rivets define an oracle: the candidate that reveals the
    # deepest pull-in (most negative k_worst_mean) is on the true nominal
    # surface and correctly reflects the physical state.
    #
    # OK rivets (label=0) do NOT get an oracle: their "true" reading can be
    # positive, negative, or near zero (nominal surface variation), so there is
    # no single correct zero to target without a gt_mm value.  Their candidates
    # are included as non-oracle negative examples — they still teach the model
    # which features correspond to poor zero choices in a defect-free context.
    oracle_labels = np.zeros(len(feat_rows), dtype=np.int32)
    if gt_label >= 1:
        oracle_idx = int(np.argmin(k_worsts))
        oracle_labels[oracle_idx] = 1
        print(f"    {surface_label} hole {hole_idx_1based:>3}  "
              f"label={gt_label}  "
              f"{len(feat_rows):>3} candidates  "
              f"oracle k_worst={k_worsts[oracle_idx]:+.4f}  "
              f"feat[plan_std]={feat_rows[oracle_idx, 0]:.4f}")
    else:
        print(f"    {surface_label} hole {hole_idx_1based:>3}  "
              f"label=0  "
              f"{len(feat_rows):>3} candidates  "
              f"(no oracle — negative examples only)")

    return [(feat_rows[i].tolist(), int(oracle_labels[i]), float(k_worsts[i]))
            for i in range(len(feat_rows))]


# ── Threshold calibration ─────────────────────────────────────────────────────
def calibrate_thresholds(surface_labels_df, mesh_dir, pc_dir, model, args):
    """
    Run the trained model on all labeled rivets, collect k_worst_mean per rivet,
    then find warn/defect thresholds that maximise per-class F1.

    Returns (warn_threshold, defect_threshold).
    """
    from virtual_comparator_v2 import (
        measure_crown_v1, zero_reading_at, find_zero_point_learned,
    )
    from virtual_comparator import find_holes, build_mastic_boundary_tree
    from virtual_comparator_v2 import auto_r_outer

    results = []
    for row in surface_labels_df:
        surface_label   = row["surface"]
        hole_idx_1based = row["hole_idx"]
        gt_label        = row["label"]

        try:
            pts, kdtree, holes, mastics, r_outer, mastic_bound_tree, mastic_bound_r = \
                _load_surface(surface_label, mesh_dir, pc_dir, args)
        except FileNotFoundError:
            continue

        idx_0 = hole_idx_1based - 1
        if idx_0 < 0 or idx_0 >= len(holes):
            continue

        hole         = holes[idx_0]
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

        zero_pt, _ = find_zero_point_learned(
            rivet_center, hole_radius, pts, kdtree,
            model=model,
            from_edge_min=args.from_edge_min,
            from_edge_max=args.from_edge_max,
            feet_radius=args.feet_radius,
            other_centers=other_centers,
            other_radii=other_radii,
            mastic_centers=mastic_centers,
            mastic_radii=mastic_radii,
        )
        if zero_pt is None:
            continue

        z_reading = zero_reading_at(
            zero_pt, pts, kdtree,
            args.feet_radius, args.zero_probe_radius,
        )
        if z_reading is None:
            continue

        res_raw = measure_crown_v1(
            rivet_center, hole_radius, pts, kdtree,
            feet_radius=args.feet_radius,
            r_inner_mm=args.r_inner,
            r_outer_mm=r_outer,
            n_sectors=args.n_sectors,
            k_worst=args.k_sectors,
            n_radial_bands=args.n_radial_bands,
            zero_offset=z_reading,
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
            continue

        results.append((res_raw["k_worst_mean"], gt_label))

    if not results:
        return -0.14, -0.21   # fallback defaults

    k_vals  = np.array([r[0] for r in results])
    gt_labs = np.array([r[1] for r in results])

    # Grid search: find warn_thresh and defect_thresh that maximise macro-F1
    # Constraint: defect_thresh <= warn_thresh <= 0
    best_f1, best_wt, best_dt = -1, -0.14, -0.21
    for wt in np.arange(-0.05, -0.50, -0.01):
        for dt in np.arange(wt - 0.01, -0.80, -0.01):
            pred = np.where(k_vals <= dt, 2,
                            np.where(k_vals <= wt, 1, 0))
            # F1 per class
            f1s = []
            for cls in [0, 1, 2]:
                tp = ((pred == cls) & (gt_labs == cls)).sum()
                fp = ((pred == cls) & (gt_labs != cls)).sum()
                fn = ((pred != cls) & (gt_labs == cls)).sum()
                p  = tp / (tp + fp) if (tp + fp) > 0 else 0
                r  = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1s.append(2*p*r/(p+r) if (p+r) > 0 else 0)
            macro_f1 = np.mean(f1s)
            if macro_f1 > best_f1:
                best_f1, best_wt, best_dt = macro_f1, float(wt), float(dt)

    return best_wt, best_dt


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)

    # Load GT
    gt_rows = []
    with open(args.gt_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row.get("label", 0))
            if args.skip_label_0 and label == 0:
                continue
            gt_rows.append({
                "surface":  row["surface"],
                "hole_idx": int(row["hole_idx"]),
                "label":    label,
            })

    print(f"\nGT rows: {len(gt_rows)}")
    label_counts = {0: 0, 1: 0, 2: 0}
    for r in gt_rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    print(f"  OK (0): {label_counts[0]}   "
          f"warn (1): {label_counts[1]}   "
          f"defect (2): {label_counts[2]}")

    surfaces = sorted({r["surface"] for r in gt_rows})
    print(f"Surfaces: {surfaces}\n")

    # Build training dataset
    all_feats, all_labels = [], []
    per_surface = {s: {"feats": [], "labels": []} for s in surfaces}

    for row in gt_rows:
        samples = build_rivet_samples(
            row["surface"], row["hole_idx"], row["label"],
            args.mesh_dir, args.pc_dir, args,
        )
        for feat, oracle_label, k_worst in samples:
            all_feats.append(feat)
            all_labels.append(oracle_label)
            per_surface[row["surface"]]["feats"].append(feat)
            per_surface[row["surface"]]["labels"].append(oracle_label)

    if not all_feats:
        print("\nERROR: no training samples generated.")
        return

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    import joblib
    from zero_point_features import FEATURE_NAMES

    X = np.array(all_feats)
    y = np.array(all_labels)

    print(f"\nTraining dataset: {len(X)} samples  "
          f"({y.sum()} oracle / {(~y.astype(bool)).sum()} non-oracle)  "
          f"imbalance 1:{int((~y.astype(bool)).sum() / max(y.sum(), 1))}")

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

    y_pred = clf.predict(X)
    print(f"\nTraining classification report:")
    print(classification_report(y, y_pred, target_names=["non-oracle", "oracle"],
                                 zero_division=0))

    importances = clf.named_steps["rf"].feature_importances_
    print("Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {name:<25} {imp:.4f}  {bar}")

    # ── Leave-one-surface-out ─────────────────────────────────────────────────
    if args.leave_one_out and len(surfaces) > 1:
        print("\nLeave-one-surface-out CV …")
        from sklearn.metrics import roc_auc_score

        for test_surf in surfaces:
            Xtr = np.array(per_surface[test_surf]["feats"])
            ytr = np.array(per_surface[test_surf]["labels"])
            Xte_list = [v["feats"]  for s, v in per_surface.items() if s != test_surf]
            yte_list = [v["labels"] for s, v in per_surface.items() if s != test_surf]
            if not Xte_list:
                continue
            Xte = np.array([f for sub in Xte_list for f in sub])
            yte = np.array([l for sub in yte_list for l in sub])

            clf_loo = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("rf",     RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    class_weight="balanced",
                    random_state=42, n_jobs=-1,
                )),
            ])
            clf_loo.fit(Xtr, ytr)
            proba = clf_loo.predict_proba(Xte)[:, 1]
            try:
                auc = roc_auc_score(yte, proba)
            except Exception:
                auc = float("nan")
            print(f"  {test_surf:<30}  AUC={auc:.3f}")

    # ── Threshold calibration ─────────────────────────────────────────────────
    if args.calibrate_thresholds:
        print("\nCalibrating classification thresholds …")
        warn_t, defect_t = calibrate_thresholds(gt_rows, args.mesh_dir, args.pc_dir, clf, args)
        print(f"  Suggested warn threshold   : {warn_t:.3f} mm")
        print(f"  Suggested defect threshold : {defect_t:.3f} mm")
        print(f"  Use:  --warn-lo {-warn_t:.2f} --warn-hi {-defect_t:.2f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    joblib.dump(clf, args.model_out)
    print(f"\nModel saved → {args.model_out}")
    print(f"  To use: python virtual_comparator_v2.py "
          f"--zero-method learned --zero-model {args.model_out}")


if __name__ == "__main__":
    main()
