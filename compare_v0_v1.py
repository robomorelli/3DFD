"""
Compare v0 (bnd_mean) vs v1 (k_worst_mean) results.
Both CSVs are in comparator_output_v1/:
  all_measurements.csv     → v0
  all_measurements_v1.csv  → v1
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

THRESHOLD = -0.20
OUT_DIR   = "comparison_output"
import os; os.makedirs(OUT_DIR, exist_ok=True)

v0 = pd.read_csv("comparator_output_v1/all_measurements.csv")
v1 = pd.read_csv("comparator_output_v1/all_measurements_v1.csv")

# ── Per-surface summary ────────────────────────────────────────────────────────
surfaces = sorted(set(v0["surface"]) | set(v1["surface"]))

s0 = v0.groupby("surface").agg(
    n_holes   = ("hole_idx",    "count"),
    worst     = ("bnd_mean_mm", "min"),
    n_defects = ("defect",      "sum"),
).reindex(surfaces)

s1 = v1.groupby("surface").agg(
    n_holes   = ("hole_idx",       "count"),
    worst     = ("k_worst_mean_mm","min"),
    n_defects = ("defect",         "sum"),
).reindex(surfaces)

# ── Plot 1: defects per surface ────────────────────────────────────────────────
x   = np.arange(len(surfaces))
w   = 0.38
lbl = [s.replace("_clean", "") for s in surfaces]

fig, ax = plt.subplots(figsize=(max(14, len(surfaces) * 0.45), 6))
ax.bar(x - w/2, s0["n_defects"].fillna(0), w,
       color="steelblue", alpha=0.85, label="v0  (bnd_mean)")
ax.bar(x + w/2, s1["n_defects"].fillna(0), w,
       color="tomato",    alpha=0.85, label="v1  (k_worst_mean)")
ax.set_xticks(x)
ax.set_xticklabels(lbl, rotation=60, ha="right", fontsize=7)
ax.set_ylabel("defects flagged")
ax.set_title(f"Defects per surface  (threshold {THRESHOLD} mm)  —  v0 vs v1")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
p1 = f"{OUT_DIR}/compare_defects_v0_v1.png"
fig.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"→ {p1}")

# ── Plot 2: worst pull-in per surface ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(14, len(surfaces) * 0.45), 6))
ax.bar(x - w/2, s0["worst"].fillna(0), w,
       color="steelblue", alpha=0.85, label="v0  (bnd_mean)")
ax.bar(x + w/2, s1["worst"].fillna(0), w,
       color="tomato",    alpha=0.85, label="v1  (k_worst_mean)")
ax.axhline(THRESHOLD, color="red", ls="--", lw=1.2, label=f"threshold {THRESHOLD}")
ax.axhline(0,         color="gray",ls=":",  lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(lbl, rotation=60, ha="right", fontsize=7)
ax.set_ylabel("worst reading (mm)")
ax.set_title("Worst pull-in per surface  —  v0 vs v1")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
p2 = f"{OUT_DIR}/compare_worst_v0_v1.png"
fig.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"→ {p2}")

# ── Plot 3: per-rivet scatter (matched by position) ────────────────────────────
matched = []
for surf in surfaces:
    df0 = v0[v0["surface"] == surf].copy()
    df1 = v1[v1["surface"] == surf].copy()
    if df0.empty or df1.empty:
        continue
    pts0 = df0[["cx","cy"]].values
    pts1 = df1[["cx","cy"]].values
    tree = cKDTree(pts1)
    dists, idxs = tree.query(pts0, k=1)
    for i, (d, j) in enumerate(zip(dists, idxs)):
        if d < 3.0:   # max 3mm positional mismatch
            matched.append({
                "surface": surf,
                "v0": df0.iloc[i]["bnd_mean_mm"],
                "v1": df1.iloc[j]["k_worst_mean_mm"],
            })

mdf = pd.DataFrame(matched).dropna()
corr = mdf[["v0","v1"]].corr().iloc[0,1]
mae  = (mdf["v0"] - mdf["v1"]).abs().mean()

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(mdf["v0"], mdf["v1"], s=6, alpha=0.35, color="steelblue", zorder=3)
lim = min(mdf["v0"].min(), mdf["v1"].min()) - 0.05
ax.plot([lim, 0.1], [lim, 0.1], "k--", lw=0.8, alpha=0.4, label="identity")
ax.axhline(THRESHOLD, color="red", ls="--", lw=1.0, alpha=0.7)
ax.axvline(THRESHOLD, color="red", ls="--", lw=1.0, alpha=0.7)
ax.axhline(0, color="gray", ls=":", lw=0.7)
ax.axvline(0, color="gray", ls=":", lw=0.7)
ax.set_xlabel("v0  bnd_mean (mm)")
ax.set_ylabel("v1  k_worst_mean (mm)")
ax.set_title(f"Per-rivet scatter  ({len(mdf)} matched rivets)\n"
             f"Pearson r={corr:.3f}   MAE={mae:.3f} mm")
ax.legend(fontsize=8)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
plt.tight_layout()
p3 = f"{OUT_DIR}/compare_scatter_v0_v1.png"
fig.savefig(p3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"→ {p3}")

# ── Console summary ────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"  v0 total defects : {int(s0['n_defects'].sum())}")
print(f"  v1 total defects : {int(s1['n_defects'].sum())}")
print(f"  Matched rivets   : {len(mdf)}")
print(f"  Pearson r        : {corr:.3f}")
print(f"  MAE              : {mae:.3f} mm")
print(f"{'─'*50}\n")
