"""Model-free reference for the masked-cell reconstruction task.

InterScale's node-regression task asks: given a cell's spatial neighbourhood (its own expression
masked out), reconstruct its expression. The obvious floor for that task is to predict each cell
as the *mean of its spatial neighbours* -- no learning, no parameters, same graph the model uses.

Why this matters: the CosMx hyperparameter sweep reached only val Pearson 0.065 (best of 60
trials) where legnini23 reached 0.335. Two very different conclusions fit that number, and the
sweep alone cannot separate them:

  * the task has little local predictability on this panel/resolution -- then ~0.06 is close to
    the ceiling and the model is not the problem;
  * or the model/setup underperforms something trivial -- then the baseline beats it and the
    sweep result is a bug, not a finding.

Run per dataset:

    python scripts/neighbour_mean_baseline.py --dataset legnini23
    python scripts/neighbour_mean_baseline.py --dataset melton25

Reports per-gene Pearson between the neighbour mean and the truth, on the val split only (the
split the sweep selected on), so the numbers are directly comparable to val_pearson_corr.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import squidpy as sq

warnings.filterwarnings("ignore")

# Mirrors config_files/datasets/*.yaml -- kept explicit rather than resolved through the registry
# so this script stays runnable without the interscale config stack.
DATASETS = {
    "legnini23": dict(
        h5ad="/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/legnini23_pp.h5ad",
        layer="log1p_norm",
        radius=200,
        library_key="sample",
    ),
    "melton25": dict(
        h5ad="/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/melton25.h5ad",
        layer=None,  # X already holds log-normalised expression on this file
        radius=110,
        library_key="slide_fov",
    ),
}


def per_gene_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Pearson r per gene (column), NaN where either side is constant."""
    t = y_true - y_true.mean(axis=0, keepdims=True)
    p = y_pred - y_pred.mean(axis=0, keepdims=True)
    num = (t * p).sum(axis=0)
    den = np.sqrt((t**2).sum(axis=0) * (p**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--split-key", default="split")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out-json", default=None, help="Write the baseline metrics here, for the figure script.")
    args = parser.parse_args()
    spec = DATASETS[args.dataset]

    adata = sc.read_h5ad(spec["h5ad"])
    print(f"{args.dataset}: {adata.n_obs} cells x {adata.n_vars} genes")

    # Same neighbour graph the model is given.
    sq.gr.spatial_neighbors(
        adata, coord_type="generic", radius=spec["radius"], library_key=spec["library_key"]
    )
    A = adata.obsp["spatial_connectivities"].astype(np.float32)

    X = adata.layers[spec["layer"]] if spec["layer"] else adata.X
    X = X.astype(np.float32)
    if not sp.issparse(X):
        X = sp.csr_matrix(X)

    # Row-normalised adjacency WITHOUT self-loops: the cell's own expression is what the task
    # masks out, so including it would leak the answer.
    deg = np.asarray(A.sum(axis=1)).ravel()
    keep = deg > 0
    inv = np.zeros_like(deg)
    inv[keep] = 1.0 / deg[keep]
    pred = sp.diags(inv) @ (A @ X)

    mask = np.asarray(adata.obs[args.split_key] == args.split).ravel() & keep
    print(f"{args.split} cells with >=1 neighbour: {int(mask.sum())} of {int((adata.obs[args.split_key] == args.split).sum())}")
    print(f"mean neighbours (all cells): {deg.mean():.1f}")

    y_true = np.asarray(X[mask].todense(), dtype=np.float64)
    y_pred = np.asarray(pred[mask].todense(), dtype=np.float64)

    # Second, weaker reference: predict the SAME vector (the training-split mean profile) for
    # every cell. It carries no cell-specific information at all, which makes it the yardstick
    # for any metric that is not mean-centred -- notably cosine similarity, which on non-negative
    # expression data is dominated by the profile all cells share.
    train_mask = np.asarray(adata.obs[args.split_key] == "train").ravel()
    global_mean = np.asarray(X[train_mask].mean(axis=0)).ravel()

    y_true = np.asarray(X[mask].todense(), dtype=np.float64)
    y_nb = np.asarray(pred[mask].todense(), dtype=np.float64)
    y_const = np.broadcast_to(global_mean.astype(np.float64), y_true.shape)

    def cosine_per_cell(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Uncentred cosine similarity per cell -- what the training plan logs."""
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where((na > 0) & (nb > 0), (a * b).sum(axis=1) / (na * nb), np.nan)

    def per_gene_ccc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Lin's concordance per gene: correlation with the calibration error folded in.

        Unlike Pearson this stays DEFINED for a constant predictor -- sd_pred enters the
        numerator, so a predictor that never varies scores exactly 0 rather than NaN. That is
        what makes it usable as the yardstick cosine similarity cannot provide.
        """
        ma, mb = a.mean(axis=0), b.mean(axis=0)
        va, vb = a.var(axis=0), b.var(axis=0)
        cov = ((a - ma) * (b - mb)).mean(axis=0)
        den = va + vb + (ma - mb) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, 2 * cov / den, np.nan)

    ss_tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    results = {}

    for label, y_pred_ in (("neighbour mean", y_nb), ("constant (train mean profile)", y_const)):
        r = per_gene_pearson(y_true, y_pred_)
        ss_res = ((y_true - y_pred_) ** 2).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan)
        cos = cosine_per_cell(y_true, y_pred_)
        ccc = per_gene_ccc(y_true, y_pred_)
        print(f"\n{label} on {args.split}:")
        print(
            f"  per-gene Pearson : mean {np.nanmean(r):.4f}  median {np.nanmedian(r):.4f}  "
            f"q90 {np.nanpercentile(r, 90):.4f}"
        )
        print(f"  per-gene CCC     : mean {np.nanmean(ccc):.4f}")
        print(f"  per-gene R2      : median {np.nanmedian(r2):.4f}")
        print(f"  cosine (per cell): mean {np.nanmean(cos):.4f}  median {np.nanmedian(cos):.4f}")
        print(f"  genes with r>0.2 : {int(np.nansum(r > 0.2))} of {len(r)}")
        results[label] = dict(
            pearson_mean=float(np.nanmean(r)) if np.isfinite(r).any() else None,
            ccc_mean=float(np.nanmean(ccc)),
            r2_median=float(np.nanmedian(r2)),
            cosine_mean=float(np.nanmean(cos)),
            genes_r_gt_02=int(np.nansum(r > 0.2)),
            n_genes=int(len(r)),
        )

    if args.out_json:
        import json

        with open(args.out_json, "w") as fh:
            json.dump({"dataset": args.dataset, "split": args.split, "baselines": results}, fh, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
