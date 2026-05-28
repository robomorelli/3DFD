"""
Side-by-side comparison: GT photo vs Virtual Comparator v1 output.

For each surface that has both a GT image and a v1 PNG, generates a
2-panel figure: left = operator photo, right = v1 deviation map.

Usage:
  python compare_gt_v1.py
  python compare_gt_v1.py --out-dir comparison_output
"""

import argparse
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir",    default="data/immagini_GT")
    p.add_argument("--v1-dir",    default="comparator_output_v1")
    p.add_argument("--out-dir",   default="comparison_output")
    p.add_argument("--dpi",       type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Build map: surface_number → v1 PNG path
    v1_map = {}
    for path in glob.glob(os.path.join(args.v1_dir, "*_comparator_v1.png")):
        name = os.path.basename(path)
        num  = name.replace("Surface", "").replace("_clean_comparator_v1.png", "")
        v1_map[num] = path

    # Build map: surface_number → GT photo path
    gt_map = {}
    for path in glob.glob(os.path.join(args.gt_dir, "*.png")):
        num = os.path.splitext(os.path.basename(path))[0]
        gt_map[num] = path

    common = sorted(set(gt_map) & set(v1_map), key=lambda x: int(x) if x.isdigit() else 9999)
    print(f"Surfaces with both GT and v1: {len(common)}")
    print(f"  {common}\n")

    for num in common:
        gt_img  = mpimg.imread(gt_map[num])
        v1_img  = mpimg.imread(v1_map[num])

        fig, axes = plt.subplots(1, 2, figsize=(22, 8))
        fig.suptitle(f"Surface {num}  —  GT photo  vs  Virtual Comparator v1",
                     fontsize=13, fontweight="bold")

        axes[0].imshow(gt_img, cmap="gray" if gt_img.ndim == 2 else None)
        axes[0].set_title("Ground truth (operator photo)", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(v1_img)
        axes[1].set_title("Virtual Comparator v1  (red = pull-in · green = nominal)",
                           fontsize=11)
        axes[1].axis("off")

        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f"compare_{num}.png")
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_path}")

    print(f"\nDone. {len(common)} comparison images in {args.out_dir}/")


if __name__ == "__main__":
    main()
