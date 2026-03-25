"""
run_all.py
==========
Master entry point.  Run from the project root (the directory that contains
the `results/` folder):

    python visualizations/run_all.py

All figures are written to  visualizations/outputs/{basic,comparative,ablation,ranking}/
"""

from __future__ import annotations

import os
import sys
import time

# Ensure we can import sibling modules regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from utils import load_all_results
from plot_basic       import run_all as run_basic
from plot_comparative import run_all as run_comparative
from plot_ablation    import run_all as run_ablation
from plot_ranking     import run_all as run_ranking

# ─── Resolve results directory ────────────────────────────────────────────────
# Assumes this script lives in   <project_root>/visualizations/run_all.py
# and data lives in              <project_root>/results/<dr_method>/

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")


def main() -> None:
    print("=" * 65)
    print("  Parkinsonism Classification — Evaluation Visualisations")
    print("=" * 65)

    t0 = time.time()

    print(f"\n→ Loading data from:  {os.path.abspath(RESULTS_DIR)}")
    all_data = load_all_results(RESULTS_DIR)

    if not all_data:
        print("\n[ERROR] No data loaded.  Check that results/ subdirectories exist.")
        sys.exit(1)

    print(f"  Loaded DR methods: {list(all_data.keys())}")
    for dr_key, data in all_data.items():
        s = data["summary"]
        print(
            f"    {dr_key:20s}  "
            f"train={s['n_train']} test={s['n_test']}  "
            f"features {s['n_features_before_dr']}→{s['n_features_after_dr']}"
        )

    print("\n─── 1 / 4  Basic Visualisations ────────────────────────────────")
    run_basic(all_data)

    print("\n─── 2 / 4  Comparative Analysis ─────────────────────────────────")
    run_comparative(all_data)

    print("\n─── 3 / 4  DR Ablation Study ────────────────────────────────────")
    run_ablation(all_data)

    print("\n─── 4 / 4  Model Ranking ────────────────────────────────────────")
    run_ranking(all_data)

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Done in {elapsed:.1f} s")
    print(f"  Figures → {os.path.abspath(os.path.join(SCRIPT_DIR, 'outputs'))}")
    print("=" * 65)


if __name__ == "__main__":
    main()