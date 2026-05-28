"""
Comparison: mini-GT (Surface140) vs Virtual Comparator v1.

Strategy:
  The GT photo right column corresponds to the left column of v1 rivets
  (cx ≈ -160 mm), containing 21 rivets sorted top→bottom by cy descending.
  GT photo_rank 1..N is matched to v1 rank 1..N within that column.

  The GT photo left column (large values -0.54, -0.42…) is matched to the
  middle column of v1 (cx ≈ 8-30 mm), same positional approach.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

THRESHOLD = -0.20   # mm — defect flag used in v1

# ── Load data ─────────────────────────────────────────────────────────────────
v1  = pd.read_csv("comparator_output_v1/Surface140_clean_comparator_v1.csv")
gt  = pd.read_csv("data/mini_gt_140.csv", comment="#")

# ── Identify column groups in v1 ──────────────────────────────────────────────
# Left column: cx ≈ -160 to -165
left_mask  = (v1["cx"] < -140) & (v1["cx"] > -200)
# Middle column: cx ≈ 8 to 30
mid_mask   = (v1["cx"] > 0) & (v1["cx"] < 50)
# Right column: cx > 200
right_mask = v1["cx"] > 150

left_col  = v1[left_mask].sort_values("cy", ascending=False).reset_index(drop=True)
mid_col   = v1[mid_mask].sort_values("cy",  ascending=False).reset_index(drop=True)
right_col = v1[right_mask].sort_values("cy", ascending=False).reset_index(drop=True)

print(f"v1 column sizes:  left={len(left_col)}  mid={len(mid_col)}  right={len(right_col)}")

# ── Match GT → v1 by photo_rank within column ─────────────────────────────────
def match_column(gt_col, v1_df, col_name):
    rows = gt[gt["photo_col"] == col_name].copy()
    rows = rows.sort_values("photo_rank").reset_index(drop=True)
    matched = []
    for _, r in rows.iterrows():
        rank = int(r["photo_rank"]) - 1      # 0-indexed
        if rank < len(v1_df):
            v1r = v1_df.iloc[rank]
            matched.append({
                "photo_rank":    r["photo_rank"],
                "gt_mm":         r["gt_mm"],
                "readable":      r["readable"],
                "notes":         r["notes"],
                "hole_idx":      v1r["hole_idx"],
                "cx":            v1r["cx"],
                "cy":            v1r["cy"],
                "v1_k_worst":    v1r["k_worst_mean_mm"],
                "v1_defect":     int(v1r["k_worst_mean_mm"] <= THRESHOLD),
                "gt_defect":     int(r["gt_mm"] <= THRESHOLD) if pd.notna(r["gt_mm"]) else None,
            })
    return pd.DataFrame(matched)

right_match = match_column(gt, left_col, "right")   # GT-right ↔ v1-left
left_match  = match_column(gt, mid_col,  "left")    # GT-left  ↔ v1-mid

all_match = pd.concat([right_match, left_match], ignore_index=True)
readable  = all_match[all_match["readable"] == 1].dropna(subset=["gt_mm"])

# ── Print comparison table ─────────────────────────────────────────────────────
print(f"\n{'─'*72}")
print(f"  Surface 140 — GT vs v1 comparison  (threshold = {THRESHOLD} mm)")
print(f"{'─'*72}")
print(f"  {'col':5}  {'rank':4}  {'hole':4}  {'cx':7}  {'cy':7}  "
      f"{'GT(mm)':8}  {'v1(mm)':8}  {'GT_def':7}  {'v1_def':7}  match")
for _, r in all_match.iterrows():
    if not r["readable"] or pd.isna(r["gt_mm"]):
        continue
    gt_d  = "⚠" if r["gt_defect"]  else "ok"
    v1_d  = "⚠" if r["v1_defect"]  else "ok"
    match = "✓" if r["gt_defect"] == r["v1_defect"] else "✗"
    col   = "right→left" if r["cy"] < 0 or True else "left→mid"
    print(f"  {r['cx']:>7.0f}  {int(r['photo_rank']):>4}  "
          f"{int(r['hole_idx']):>4}  {r['cx']:>7.1f}  {r['cy']:>7.1f}  "
          f"{r['gt_mm']:>8.3f}  {r['v1_k_worst']:>8.3f}  "
          f"{gt_d:>7}  {v1_d:>7}  {match}")

# Summary stats
gt_def  = readable["gt_defect"].sum()
v1_def  = readable["v1_defect"].sum()
agree   = (readable["gt_defect"] == readable["v1_defect"]).sum()
corr    = readable[["gt_mm", "v1_k_worst"]].corr().iloc[0, 1]
mae     = (readable["gt_mm"] - readable["v1_k_worst"]).abs().mean()
print(f"\n  Readable annotations matched: {len(readable)}")
print(f"  GT defects (|reading|>0.2):   {int(gt_def)}")
print(f"  v1 defects detected:          {int(v1_def)}")
print(f"  Classification agreement:     {int(agree)}/{len(readable)}")
print(f"  Pearson correlation GT↔v1:    {corr:.3f}")
print(f"  MAE  GT reading vs v1:        {mae:.3f} mm")

# ── Scatter plot GT reading vs v1 k_worst_mean ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Surface 140 — Mini-GT vs Virtual Comparator v1",
             fontsize=13, fontweight="bold")

# Left: scatter
ax = axes[0]
colors = []
for _, r in readable.iterrows():
    if r["gt_defect"] == 1 and r["v1_defect"] == 1:
        colors.append("red")        # both flag
    elif r["gt_defect"] == 0 and r["v1_defect"] == 0:
        colors.append("green")      # both ok
    elif r["gt_defect"] == 1:
        colors.append("orange")     # GT defect, v1 misses
    else:
        colors.append("blue")       # v1 flags, GT doesn't

sc = ax.scatter(readable["gt_mm"], readable["v1_k_worst"],
                c=colors, s=80, zorder=3, edgecolors="black", linewidths=0.5)

lim = min(readable["gt_mm"].min(), readable["v1_k_worst"].min()) - 0.05
ax.plot([lim, 0.05], [lim, 0.05], "k--", lw=0.8, alpha=0.4, label="identity")
ax.axhline(THRESHOLD, color="red",  ls="--", lw=1.0, alpha=0.7,
           label=f"threshold {THRESHOLD}")
ax.axvline(THRESHOLD, color="red",  ls="--", lw=1.0, alpha=0.7)
ax.axhline(0, color="gray", ls=":", lw=0.7)
ax.axvline(0, color="gray", ls=":", lw=0.7)

# Annotate each point with hole index
for _, r in readable.iterrows():
    ax.annotate(f"h{int(r['hole_idx'])}", (r["gt_mm"], r["v1_k_worst"]),
                fontsize=7, ha="left", va="bottom",
                xytext=(3, 3), textcoords="offset points")

patches = [
    mpatches.Patch(color="red",    label="both flag (TP)"),
    mpatches.Patch(color="orange", label="GT defect, v1 miss (FN)"),
    mpatches.Patch(color="blue",   label="v1 flags, GT ok (FP)"),
    mpatches.Patch(color="green",  label="both ok (TN)"),
]
ax.legend(handles=patches, fontsize=8, loc="upper left")
ax.set_xlabel("GT reading (mm)"); ax.set_ylabel("v1 k_worst_mean (mm)")
ax.set_title(f"Scatter  (Pearson r={corr:.2f}  MAE={mae:.3f} mm)")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

# Right: bar chart side by side
ax2 = axes[1]
x  = np.arange(len(readable))
w  = 0.38
bars_gt = ax2.bar(x - w/2, readable["gt_mm"].values, w,
                  color="steelblue", alpha=0.8, label="GT reading")
bars_v1 = ax2.bar(x + w/2, readable["v1_k_worst"].values, w,
                  color="tomato", alpha=0.8, label="v1 k_worst_mean")
ax2.axhline(THRESHOLD, color="red", ls="--", lw=1.2,
            label=f"threshold {THRESHOLD}")
ax2.axhline(0, color="gray", ls=":", lw=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(
    [f"h{int(r.hole_idx)}\n(r{int(r.photo_rank)})"
     for _, r in readable.iterrows()],
    fontsize=7, rotation=45, ha="right",
)
ax2.set_ylabel("reading (mm)")
ax2.set_title("GT vs v1 per rivet  (h=hole idx, r=GT photo rank)")
ax2.legend(fontsize=9)
ax2.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
out = "comparison_output/compare_gt140_vs_v1.png"
os.makedirs("comparison_output", exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Plot → {out}")
