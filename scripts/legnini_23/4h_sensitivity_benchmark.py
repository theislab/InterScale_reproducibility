"""Legnini - which single-seed sensitivity score predicts cross-seed gene recurrence?

Reviewer question this answers (part 3): given that the top-K gene sets differ between training
seeds, can a score computed from ONE model tell which of its genes the other seeds would also
have picked? Running this needs no retraining and no inference: it reads the per-seed arrays
`4a_seed_stability_extract.py` wrote and, if present, the per-gene input-perturbation shifts
`4g_perturbation_sensitivity.py` wrote.

What it prints, in order:

1. **Agreement check** -- the AnnData rebuilt from the cached arrays reproduces the gene sets in
   `seed*_meta.json` exactly, so everything below is scored against interscale's own selection
   code rather than a reimplementation of it.
2. **The invariance result** -- no resampling of the cells can change the within-dimension top-K,
   because the ranking statistic reduces to the frozen decoder weight. This is why the coherence
   pruning and the correlation permutation test in the notebook never moved the gene sets in the
   direction they were meant to.
3. **The dropout finding** -- the published embeddings were computed with dropout active, since
   neither `BaseModel.load()` nor `get_model_output` calls `.eval()`.
4. **The benchmark** -- leave-one-seed-out AUROC / average precision / Spearman for every
   candidate score, under both selection rules and for both decoders.

Usage
-----
    python scripts/legnini_23/4h_sensitivity_benchmark.py
    python scripts/legnini_23/4h_sensitivity_benchmark.py --arm gene --metric ap

Run it inside the interscale apptainer container. Takes a few minutes (the permutation nulls
dominate).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

REPO = Path("/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility")
LUSTRE = Path("/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer")
sys.path.insert(0, str(REPO))

from src.loading_sensitivity import (  # noqa: E402
    RULES,
    SEEDS,
    build_loading_adata,
    dropout_report,
    evaluate,
    load_perturbation,
    load_seed_arrays,
    plot_recurrence_vs_score,
    plot_score_comparison,
    prove_selection_is_data_independent,
    selected_sets,
    summarise,
)


def load_expression(layer_key: str = "log1p_norm") -> np.ndarray:
    """The matrix the cached arrays were computed on: the training layer, zero cells removed."""
    import scanpy as sc
    from scipy.sparse import issparse

    from interscale.config.registry import resolve_config
    from interscale.tl import remove_zero_expression_cells

    cfg = resolve_config(
        "legnini23", "node_reg",
        registry_path="/home/icb/francesca.drummer/1-Projects/A3_InterScale/interscale/config_files/registry.yaml",
    )
    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    # remove_zero_expression_cells reads .X, which on this h5ad holds raw counts; restoring it
    # explicitly keeps the filter identical to the one training applied regardless of what a
    # previous cell left in .X.
    adata.X = adata.layers["raw"]
    adata = remove_zero_expression_cells(adata)
    Xl = adata.layers[layer_key]
    return np.asarray(Xl.todense() if issparse(Xl) else Xl, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="cell", choices=("cell", "gene"))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--metric", default="auroc", choices=("auroc", "ap", "spearman"))
    args = ap.parse_args()

    analysis = LUSTRE / "results" / "legnini23_revision" / f"seed_stability_{args.arm}" / "analysis"
    out_dir = LUSTRE / "results" / "legnini23" / f"sensitivity_{args.arm}"
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays, meta = load_seed_arrays(analysis, args.seeds)
    X = load_expression()
    adata = build_loading_adata(arrays, X)
    perturbation = load_perturbation(analysis, args.seeds)
    pd.set_option("display.width", 220)

    # -- 1. the rebuilt AnnData must reproduce the stored programs exactly ---------------------
    print("\n" + "=" * 90 + "\n1. rebuilt-AnnData agreement with seed*_meta.json\n" + "=" * 90)
    from interscale.evaluation import get_genes_dim

    for s in args.seeds:
        for which in ("local", "global"):
            df = get_genes_dim(adata, which=which, prefix=s, dims=None, n_top=meta[s]["n_top"],
                               X_layer="log1p_norm", min_frac=0.1, rank_by="loading_x_sd",
                               enforce_specificity=True, specificity_min=1.5, plot=False)
            same = list(df.index) == meta[s][f"program_{which}"]
            print(f"  seed {s} {which:6s}: {len(df.index):3d} genes, identical to stored program: {same}")
            assert same, "the rebuilt AnnData does not reproduce get_genes_dim's stored output"

    # -- 2. the invariance result ---------------------------------------------------------------
    print("\n" + "=" * 90 +
          "\n2. can ANY resampling of the cells move the within-dimension top-K?\n" + "=" * 90)
    inv = pd.DataFrame([
        prove_selection_is_data_independent(arrays, X, s, w)
        for s in args.seeds for w in ("local", "global")
    ])
    print(inv.to_string(index=False))
    print(
        "\n  The ranking statistic |S[f,e]|*sd(X_f) equals |W[f,e]|*sd(Z_e) to ~1e-4, the "
        "rankings\n  are identical in every dimension, and 50 half-sample resamples changed the "
        "top-K zero times.\n  The gene sets are a function of the frozen decoder weights alone: "
        "permuting a gene, masking\n  it, or bootstrapping cells cannot move them. Only the "
        "choice of DIMENSIONS is data-dependent."
    )

    # -- 3. dropout ------------------------------------------------------------------------------
    if perturbation:
        print("\n" + "=" * 90 +
              "\n3. the published embeddings were computed with dropout active\n" + "=" * 90)
        dr = dropout_report(arrays, perturbation, args.seeds)
        print(dr.round(4).to_string(index=False))
        print(
            "\n  BaseModel.load() does not call .eval() and neither does get_model_output, so "
            "dropout\n  (0.1 on both components) was on. mean_dim_r is the correlation between "
            "the dropout-free\n  embedding and the published one; sd_ratio < 1 means dropout "
            "inflated sd(Z_e). sd(Z_e) is\n  a per-dimension constant so it cannot reorder genes, "
            "but it does feed calculate_dim_importance,\n  which is what the 'left20' rule uses "
            "to choose dimensions."
        )
        dr.to_csv(out_dir / "dropout_report.csv", index=False)

    # -- 4. set sizes and the prediction target --------------------------------------------------
    print("\n" + "=" * 90 + "\n4. the target: how often is a gene selected across seeds?\n" + "=" * 90)
    genes = np.asarray([str(g) for g in adata.var_names])
    for rule in RULES:
        for which in ("local", "global"):
            sets = selected_sets(adata, rule, which, seeds=args.seeds)
            cnt = np.array([sum(g in sets[s] for s in args.seeds) for g in genes])
            hist = np.bincount(cnt, minlength=len(args.seeds) + 1).tolist()
            print(f"  {rule:7s} {which:6s}  set sizes {[len(sets[s]) for s in args.seeds]}"
                  f"   genes selected by 0..{len(args.seeds)} seeds: {hist}")

    # -- 5. the benchmark -------------------------------------------------------------------------
    print("\n" + "=" * 90 + "\n5. leave-one-seed-out benchmark\n" + "=" * 90)
    res = pd.concat(
        [evaluate(adata, arrays, X, rule, which, perturbation=perturbation, seeds=args.seeds)
         for rule in RULES for which in ("local", "global")],
        ignore_index=True,
    )
    res.to_csv(out_dir / "sensitivity_benchmark_raw.csv", index=False)

    for scope in ("selected_only", "all_genes"):
        print(f"\n--- {args.metric}, scope = {scope} "
              f"(mean over the 5 held-out seeds; 'overall' averages the four settings) ---")
        tbl = summarise(res.query("scope == @scope"), args.metric).round(3)
        print(tbl.to_string())
        tbl.to_csv(out_dir / f"sensitivity_summary_{args.metric}_{scope}.csv")

    # -- 6. figures ---------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, scope in zip(axes, ("selected_only", "all_genes")):
        plot_score_comparison(res, metric=args.metric, scope=scope, ax=ax)
    fig.savefig(out_dir / f"sensitivity_scores_{args.metric}.png", dpi=200, bbox_inches="tight")

    best = (summarise(res.query("scope == 'selected_only'"), args.metric)["overall"]
            .drop(index=["expr_sd", "expr_frac"], errors="ignore").idxmax())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, (rule, which) in zip(axes.ravel(), [(r, w) for r in RULES for w in ("local", "global")]):
        plot_recurrence_vs_score(adata, arrays, X, best, rule, which,
                                 perturbation=perturbation, seeds=args.seeds, ax=ax, annotate=5)
    # tight_layout runs inside plot_recurrence_vs_score, so the suptitle has to be added after
    # it and given its own strip of the figure or it lands on top of the first row's titles.
    fig.suptitle(f"best score that is not a plain expression control: {best}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / f"recurrence_vs_{best}.png", dpi=200, bbox_inches="tight")
    print(f"\nwrote csvs and figures to {out_dir}")


if __name__ == "__main__":
    main()
