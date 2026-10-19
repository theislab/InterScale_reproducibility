"""Build the sliding-window overlap ladder on legnini23, in SHH-ring space.

Reviewer comment 5 asks for a sensitivity analysis over window size *and overlap*. Overlap is the
half that needs a trained model to be convincing, and legnini23 is the dataset to do it on: its
windows are defined from `dist_to_SHH` rings, the interactions are read on the same zone
annotation, and the zones are sharply bounded. So a zone pair either shares a window or it does
not, with no ambiguity.

The ladder holds the window WIDTH fixed at two zones and varies only the stride, which is the
comparison the reviewer asked for:

    step            windows per sample                          zone pairs measurable
    0  width 1      {SHH} {near} {medium} {far}                  0 of 6
    1  width 2      {SHH,near} {medium,far}                      2 of 6   SHH-near, medium-far
       stride 2     (no overlap: a partition)
    2  width 2      as step 1, PLUS {SHH} {near,medium} {far}    3 of 6   + near-medium
       stride 1     (50% overlap: every cell in 2 windows)
    3  whole sample {SHH,near,medium,far}                        6 of 6

Steps 0 and 3 reproduce the existing `legnini23_coarse` and `legnini23` node_reg arms exactly
(64 graphs / largest 2812, and 17 graphs / largest 4318), so the ladder is anchored at both ends by
configurations already in the manuscript.

--- how overlapping windows are represented ----------------------------------------------------

Not as boolean membership masks with the complement filled in, which is what
3_sliding_window_ablation.ipynb had to do and which produced single-zone complement windows that
diluted every entry of the flow matrix. Instead each step is one or more NaN-free **partitions**,
which is how squidpy represents an overlapping tiling too: step 2 is the step-1 partition plus a
second partition offset by one ring. Their union puts every `near` and `medium` cell in two
windows, while each column on its own is still a clean partition and therefore usable as a
`sample_key` directly.

`prepare_geome_dataset` loops over every column in `sample_key` and extends the dataset, so
training on step 2 sees both partitions. Note that `prepare_a2d_dataset` returns inside that loop,
so `get_model_output` only ever builds graphs from `sample_key[0]`; the extraction script evaluates
each column separately and merges the windows, which is the correct aggregation for an overlapping
tiling.

Writes a new h5ad rather than editing legnini23_pp.h5ad in place, so the arms already trained
against that file keep reading exactly what they were trained on.

    python scripts/legnini_23/3c_overlap_ladder_prep.py
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore", category=FutureWarning)

DATA = Path("/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data")
IN_H5AD = DATA / "legnini23_pp.h5ad"
OUT_H5AD = DATA / "legnini23_overlap.h5ad"

ZONES = ["SHH", "near", "medium", "far"]  # ordered by distance from the SHH source
PX_UM = 0.138

# Each step is a list of partitions; each partition is a list of zone groups covering all 4 zones.
LADDER = {
    "step0": [[["SHH"], ["near"], ["medium"], ["far"]]],
    "step1": [[["SHH", "near"], ["medium", "far"]]],
    "step2": [[["SHH", "near"], ["medium", "far"]],
              [["SHH"], ["near", "medium"], ["far"]]],
    "step3": [[["SHH", "near", "medium", "far"]]],
}
STEP_LABEL = {
    "step0": "width 1, no overlap",
    "step1": "width 2, stride 2 (no overlap)",
    "step2": "width 2, stride 1 (50% overlap)",
    "step3": "whole sample",
}


def column_name(step, partition_index):
    """`overlap_step2_p1` -- the partition index matters, since step 2 has two."""
    return f"overlap_{step}_p{partition_index}"


def measurable_pairs(partitions):
    """Unordered zone pairs that share a window somewhere in this step."""
    out = set()
    for groups in partitions:
        for g in groups:
            out |= {frozenset(p) for p in itertools.combinations(g, 2)}
    return sorted("-".join(sorted(p, key=ZONES.index)) for p in out)


def build(adata):
    """Add one categorical column per (step, partition). Every column is NaN-free."""
    zone = adata.obs["SHH_zone"].astype(str)
    sample = adata.obs["sample"].astype(str)
    assert zone.isin(ZONES).all(), f"unexpected SHH_zone values: {set(zone) - set(ZONES)}"

    info = {}
    for step, partitions in LADDER.items():
        cols, sizes, mult = [], [], np.zeros(adata.n_obs, dtype=int)
        for pi, groups in enumerate(partitions):
            covered = sorted(z for g in groups for z in g)
            assert covered == sorted(ZONES), (
                f"{step} partition {pi} does not cover every zone exactly once: {covered}"
            )
            lab = pd.Series(pd.NA, index=adata.obs_names, dtype=object)
            for g in groups:
                m = zone.isin(g).to_numpy()
                # Window id is scoped to the sample: a "window" that pooled all 17 samples would
                # be one enormous graph, not a window. '__' separates the halves because the
                # sample names themselves contain '_'.
                lab[m] = sample[m] + "__" + "_".join(g)
                mult += m.astype(int)
            # scvi's CategoricalObsField maps an unseen value to code -1 and aborts
            # register_fields, so a sample_key column must be NaN-free.
            assert lab.notna().all(), f"{step} p{pi} leaves NaN; not usable as a sample_key"
            name = column_name(step, pi)
            adata.obs[name] = pd.Categorical(lab)
            cols.append(name)
            v = adata.obs[name].value_counts()
            sizes.extend(v[v > 0].tolist())
        info[step] = {
            "columns": cols,
            "n_graphs": len(sizes),
            "largest": int(max(sizes)),
            "median": int(np.median(sizes)),
            "pairs": measurable_pairs(partitions),
            "cells_in_multiple_windows": int((mult > 1).sum()),
        }
    return adata, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_H5AD))
    args = ap.parse_args()

    adata = sc.read_h5ad(IN_H5AD)
    print(f"{adata.n_obs} cells, {adata.n_vars} genes, {adata.obs['sample'].nunique()} samples")
    print("zone sizes:", adata.obs["SHH_zone"].value_counts().reindex(ZONES).to_dict())
    print("ring bounds (px -> um at 0.138):  SHH 0-200 (0-28)  near 200-2000 (28-276)  "
          "medium 2000-3000 (276-414)  far >3000 (>414)")

    adata, info = build(adata)

    print(f"\n{'step':7s} {'definition':32s} {'cols':>4s} {'graphs':>7s} {'largest':>8s} "
          f"{'median':>7s} {'pairs':>6s}  in >1 window")
    for step, d in info.items():
        print(f"{step:7s} {STEP_LABEL[step]:32s} {len(d['columns']):4d} {d['n_graphs']:7d} "
              f"{d['largest']:8d} {d['median']:7d} {len(d['pairs']):4d}/6 "
              f"{d['cells_in_multiple_windows']:13d}")
    print("\nzone pairs each step can measure:")
    for step, d in info.items():
        print(f"  {step}: {d['pairs'] or '(none)'}")

    print("\nregistry values for these arms:")
    for step, d in info.items():
        print(f"  {step}:  sample_key: {d['columns']}   max_seq_len: {d['largest']}")

    adata.write_h5ad(args.out)
    print(f"\nwrote {args.out} ({Path(args.out).stat().st_size / 1e6:.0f} MB)")
    pd.DataFrame(info).T.to_csv(Path(args.out).with_suffix(".ladder.csv"))


if __name__ == "__main__":
    main()
