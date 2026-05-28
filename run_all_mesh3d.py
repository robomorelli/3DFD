"""
Run the virtual comparator (auto-zero) on the mesh3D point clouds.

Same algorithm as run_all_autozero.py — only the input/output directories differ.
Results go to comparator_output_mesh3d/.

Usage:
  python run_all_mesh3d.py
  python run_all_mesh3d.py --workers 4
  python run_all_mesh3d.py --warn-lo 0.14 --warn-hi 0.21
"""

import sys
import os

# Inject default paths before argparse runs
_DEFAULTS = ["--ply-dir", "data/mesh3D", "--out-dir", "comparator_output_mesh3d"]

if __name__ == "__main__":
    # Prepend our defaults; explicit user flags will override them because
    # argparse keeps the last occurrence when defaults are in argv.
    sys.argv = [sys.argv[0]] + _DEFAULTS + sys.argv[1:]

    sys.path.insert(0, os.path.dirname(__file__))
    from run_all_autozero import main
    main()