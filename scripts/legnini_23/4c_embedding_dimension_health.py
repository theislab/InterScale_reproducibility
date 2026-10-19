"""Are the embedding dimensions actually populated, per seed?

Context: the cross-seed comparison (4b) aligns dimensions by Hungarian matching on
|corr(z_i^a, z_j^b)| and discards matches below |r| = 0.3. Local loses 22.5% of its matches that
way, global loses 66.9%. Two very different things produce that asymmetry:

  * the global embedding has many COLLAPSED dimensions -- a near-constant dimension correlates
    with nothing, so it is discarded for a boring reason and the asymmetry is an artefact;
  * the global dimensions are genuinely populated but land on a DIFFERENT BASIS in every run,
    which is a real statement about the identifiability of the global component.

This script separates them. For each seed and component it reports the per-dimension spread, how
many dimensions are effectively dead, and the effective rank (PCs needed for 90% of the
embedding's own variance).

    python scripts/legnini_23/4c_embedding_dimension_health.py

Result when first run (seeds 40-44, legnini23): zero dead dimensions in either component,
comparable median sd (0.32-0.44 both), effective rank 9-11 of 16 for both -- so the discard
asymmetry is NOT collapse, and the |r| >= 0.3 threshold is being applied to equally populated
dimensions in both cases.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ARRAYS_DIR = Path(
    "/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/results/"
    "legnini23_revision/seed_stability/analysis"
)
SEEDS = [40, 41, 42, 43, 44]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arrays-dir", default=str(ARRAYS_DIR))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--dead-sd", type=float, default=1e-3, help="A dimension below this sd counts as dead.")
    ap.add_argument("--var-frac", type=float, default=0.90, help="Variance fraction defining effective rank.")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    rows = []
    for seed in args.seeds:
        arr = np.load(Path(args.arrays_dir) / f"seed{seed}_arrays.npz", allow_pickle=True)
        for comp in ("local", "global"):
            Z = arr[f"Z_{comp}"].astype(np.float64)
            Z = Z[np.isfinite(Z).all(axis=1)]
            sd = Z.std(axis=0)

            Zc = Z - Z.mean(axis=0, keepdims=True)
            sv = np.linalg.svd(Zc, full_matrices=False, compute_uv=False)
            frac = sv**2 / (sv**2).sum()
            eff_rank = int(np.searchsorted(np.cumsum(frac), args.var_frac) + 1)

            rows.append(
                {
                    "seed": seed, "component": comp, "n_dims": Z.shape[1],
                    "n_dead": int((sd < args.dead_sd).sum()),
                    "sd_median": float(np.median(sd)), "sd_min": float(sd.min()), "sd_max": float(sd.max()),
                    "eff_rank": eff_rank,
                }
            )

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nper component, across seeds:")
    print(df.groupby("component")[["n_dead", "sd_median", "eff_rank"]].mean().to_string(
        float_format=lambda v: f"{v:.2f}"))

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
