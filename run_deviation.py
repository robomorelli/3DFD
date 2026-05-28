"""
Run deviation_map.py on all PLY files in parallel.

Usage:
  python run_deviation.py
  python run_deviation.py --workers 4 --smooth-radius 15 --no-interactive
"""

import argparse
import glob
import os
import time
import numpy as np
import trimesh
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.insert(0, os.path.dirname(__file__))
from deviation_map import compute_deviation, make_static_plot, make_interactive_plot


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ply-dir",       default="data/pc")
    p.add_argument("--out-dir",       default="deviation_output")
    p.add_argument("--grid-res",      type=float, default=0.4)
    p.add_argument("--smooth-radius", type=float, default=15.0)
    p.add_argument("--poly-degree",   type=int,   default=4)
    p.add_argument("--clip-mm",       type=float, default=0.5)
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--interactive",   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--plots",         action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-html-pts",  type=int,   default=300_000)
    return p.parse_args()


def process_one(ply_path, out_dir, grid_res, smooth_radius, poly_degree,
                clip_mm, save_plot, save_html, max_html_pts):
    label = os.path.splitext(os.path.basename(ply_path))[0]
    t0    = time.time()

    mesh = trimesh.load(ply_path, process=False)
    pts  = np.asarray(mesh.vertices, dtype=np.float64)

    deviation, ref_grid, xi, yi = compute_deviation(pts, grid_res, smooth_radius, poly_degree)

    if save_plot:
        out_png = os.path.join(out_dir, f"{label}_deviation.png")
        make_static_plot(pts, deviation, smooth_radius, clip_mm, label, out_png)

    if save_html:
        out_html = os.path.join(out_dir, f"{label}_deviation.html")
        make_interactive_plot(pts, deviation, smooth_radius, clip_mm,
                              label, max_html_pts, out_html)

    return {
        "label":   label,
        "n_pts":   len(pts),
        "dev_min": float(deviation.min()),
        "dev_max": float(deviation.max()),
        "pct_neg01": float((deviation < -0.1).mean() * 100),
        "pct_neg02": float((deviation < -0.2).mean() * 100),
        "elapsed": round(time.time() - t0, 1),
    }


def main():
    args     = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ply_files = sorted(glob.glob(os.path.join(args.ply_dir, "*.ply")))
    print(f"\nFound {len(ply_files)} PLY files  —  workers={args.workers}")
    print(f"poly_degree={args.poly_degree}  smooth_radius={args.smooth_radius} mm  "
          f"clip={args.clip_mm} mm\n")

    results = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                process_one, f, args.out_dir,
                args.grid_res, args.smooth_radius, args.poly_degree,
                args.clip_mm, args.plots, args.interactive, args.max_html_pts,
            ): f for f in ply_files
        }
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            print(f"  [{done:>2}/{len(ply_files)}] {r['label']:<25}  "
                  f"range [{r['dev_min']:+.3f}, {r['dev_max']:+.3f}] mm  "
                  f"<−0.2mm: {r['pct_neg02']:.1f}%  {r['elapsed']}s")
            results.append(r)

    print(f"\nTotal time: {time.time()-t_start:.1f}s")
    print(f"Output in: {args.out_dir}/\n")


if __name__ == "__main__":
    main()
