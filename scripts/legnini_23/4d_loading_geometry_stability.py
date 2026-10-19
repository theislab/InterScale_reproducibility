"""Cross-seed stability of what the decoders learn, measured WITHOUT dimension matching.

Why this exists
---------------
`4b_seed_stability_compare.py` answers "do matched latent dimensions carry the same top genes?"
and gets ~0.54 top-20 Jaccard -- which looks like half the panel recurring until you read it
against that script's own shuffled-dimension control, which is ALSO ~0.55. The whole overlap is
an artifact of how a standardized loading is built:

    S_gk = W_gk * std(z_k) / std(x_g)          (evaluation/_gene_loadings.py)

The `1 / std(x_g)` factor is a property of the DATA, so it is byte-identical across seeds. Genes
with small expression sd are therefore pushed to the top of |S| in every dimension of every run,
and top-k overlap measures that shared denominator rather than anything the model learned.

Everything here is built to be immune to that, and to the two other things that make
dimension-wise comparison fragile (latent dimensions are defined only up to permutation and
sign across independent runs):

  1. `gene_geometry`   Gene-gene co-loading similarity, G = rownorm(S) rownorm(S)^T. Row-
                       normalising DIVIDES OUT the 1/std(x_g) factor exactly, and G is invariant
                       to any orthogonal change of latent basis (S -> S Q leaves S S^T fixed),
                       hence to permutation and sign. Compared across seeds by Spearman over the
                       off-diagonal upper triangle, against a gene-label permutation null.
  2. `cka`             Linear CKA between the two loading matrices -- the scalar summary of (1).
  3. `pred_pattern`    Per-gene Pearson between the two seeds' PREDICTED expression across cells.
                       No basis, no matching, no sign: the functional question, "do two seeds
                       predict the same spatial pattern for this gene?"
  4. `contrast`        The up/down-regulation check. Per gene, the predicted contrast between one
                       cell group and the rest (a log-fold-change, since the target is log1p):
                       cross-seed Spearman and sign agreement, AND -- the part that matters --
                       agreement with the SAME contrast measured on the observed data. Seeds
                       agreeing with each other is necessary; agreeing with the biology is the
                       claim worth making.
  5. `co_membership`   The "pairings" check. Genes are clustered by their loading vectors per
                       seed; cross-seed agreement of which gene PAIRS land together is reported
                       as ARI and as pair-co-membership Jaccard, both basis-free.
  6. `topk_artifact`   The diagnostic for the paragraph above: top-20 Jaccard computed on S, on
                       raw W, and on rownorm(S), each against its own shuffled-dimension
                       control. Quantifies how much of the original 0.54 was the denominator.

Nothing here needs a GPU or a model: predictions are rebuilt as `Z @ W.T` from the arrays
`4a_seed_stability_extract.py` already wrote. That is exact -- the decoder is a single Linear
layer -- and the per-gene Pearson it reproduces matches the stored `pearson_{local,global}` to
8e-9. The decoder bias is not stored and is not needed: it cancels in a correlation and in a
difference of group means.

Usage
-----
    python scripts/legnini_23/4d_loading_geometry_stability.py                  # legnini23
    python scripts/legnini_23/4d_loading_geometry_stability.py --spec melton25  # CosMx
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

LUSTRE = Path("/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer")
REPO = Path("/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility")

# One entry per dataset the seed-stability pipeline has run on. `group_key` is the obs column the
# up/down-regulation contrast is taken over, and `truth` names where the expression the model was
# trained to reconstruct lives ("X" when cfg.dataset.layer_key is None, as for melton25).
SPECS = {
    "legnini23": {
        "h5ad": LUSTRE / "data" / "legnini23_pp.h5ad",
        "arrays_dir": LUSTRE / "results" / "legnini23_revision" / "seed_stability" / "analysis",
        "out_dir": REPO / "figures" / "legnini23" / "seed_stability",
        "truth": "layers/log1p_norm",
        "group_key": "SHH_zone",
        "seeds": [40, 41, 42, 43, 44],
    },
    "melton25": {
        "h5ad": LUSTRE / "data" / "melton25.h5ad",
        "arrays_dir": LUSTRE / "results" / "melton25_revision" / "best_recon" / "analysis",
        "out_dir": REPO / "figures" / "melton25" / "seed_stability",
        "truth": "X",
        "group_key": "cell_type_coarse",
        "seeds": [40, 41, 42, 43, 44],
    },
    # Masking-granularity pair. Matched on everything but the masking -- same five seeds, same
    # 300 epochs, same architecture, both with MASK_VALUE = -1 -- so a difference in cross-seed
    # cluster agreement is attributable to cell-vs-gene masking. These mirror the specs of the
    # same name in 4a_seed_stability_extract.py, which writes the arrays these read; keep the two
    # in step. Distinct from "legnini23" above, which is the PUBLISHED arm (100 epochs,
    # MASK_VALUE = 0) and is left untouched.
    "legnini23_mask_cell": {
        "h5ad": LUSTRE / "data" / "legnini23_pp.h5ad",
        "arrays_dir": LUSTRE / "results" / "legnini23_revision" / "seed_stability_cell" / "analysis",
        "out_dir": REPO / "figures" / "legnini23" / "seed_stability_cell",
        "truth": "layers/log1p_norm",
        "group_key": "SHH_zone",
        "seeds": [40, 41, 42, 43, 44],
    },
    "legnini23_mask_gene": {
        "h5ad": LUSTRE / "data" / "legnini23_pp.h5ad",
        "arrays_dir": LUSTRE / "results" / "legnini23_revision" / "seed_stability_gene" / "analysis",
        "out_dir": REPO / "figures" / "legnini23" / "seed_stability_gene",
        "truth": "layers/log1p_norm",
        "group_key": "SHH_zone",
        "seeds": [40, 41, 42, 43, 44],
    },
}

COMPONENTS = ("local", "global")


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------
def load_h5ad_parts(path: Path, truth: str, group_key: str):
    """Expression, group labels and split, filtered exactly as training filtered them.

    Read through h5py rather than anndata so this runs on a login node. The filter reproduces
    `interscale.tl.remove_zero_expression_cells`, which drops cells whose .X row sums to zero --
    equivalently, on a CSR matrix, whose row has no stored entries.
    """
    import h5py

    with h5py.File(path, "r") as f:
        n_obs, n_var = (int(v) for v in f["X"].attrs["shape"])
        keep = np.diff(f["X"]["indptr"][:]) > 0

        node = f[truth] if truth != "X" else f["X"]
        if "indptr" in node:
            X = csr_matrix(
                (node["data"][:], node["indices"][:], node["indptr"][:]), shape=(n_obs, n_var)
            )
            y_true = np.asarray(X[keep].todense(), dtype=np.float32)
        else:
            y_true = np.asarray(node[:], dtype=np.float32)[keep]

        def as_str(node):
            """Strings out of any of the three layouts anndata writes them in.

            A plain string-array is a Dataset; a nullable-string-array (what legnini23_pp uses
            for var/_index) is a Group holding `values` + `mask`; a categorical is a Group
            holding `categories` + `codes`.
            """
            if isinstance(node, h5py.Dataset):
                raw = node[:]
            elif "categories" in node:
                cats = np.asarray(node["categories"][:])
                raw = cats[node["codes"][:]]
            else:
                raw = node["values"][:]
            return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in raw])

        groups = as_str(f["obs"][group_key])[keep]
        var_names = as_str(f["var"]["_index"])

    return y_true, groups, var_names, int(keep.sum())


def load_seed(seed: int, arrays_dir: Path):
    npz = np.load(arrays_dir / f"seed{seed}_arrays.npz", allow_pickle=True)
    with open(arrays_dir / f"seed{seed}_meta.json") as f:
        meta = json.load(f)
    return npz, meta


# --------------------------------------------------------------------------------------------
# basis-free primitives
# --------------------------------------------------------------------------------------------
def row_normalize(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale every gene's loading vector to unit length.

    This is the step that removes the standardization artifact: S_g = W_g diag(z_std) / std(x_g),
    so the per-gene scalar 1/std(x_g) -- identical across seeds because it comes from the data --
    divides out exactly, leaving only the DIRECTION of the gene's loading vector.
    """
    n = np.linalg.norm(S, axis=1, keepdims=True)
    return S / np.maximum(n, eps)


def gene_gram(S: np.ndarray) -> np.ndarray:
    """Gene-gene cosine similarity of loading vectors; invariant to orthogonal basis change."""
    R = row_normalize(np.asarray(S, dtype=np.float64))
    return R @ R.T


def offdiag(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    """Linear CKA between two gene x dim matrices, genes as samples.

    Invariant to orthogonal transforms and isotropic scaling of either side, so it compares the
    geometry of the gene cloud without ever pairing a dimension of A to one of B.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(A.T @ B, ord="fro") ** 2
    na = np.linalg.norm(A.T @ A, ord="fro")
    nb = np.linalg.norm(B.T @ B, ord="fro")
    return float(hsic / (na * nb)) if na > 0 and nb > 0 else np.nan


def columnwise_pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson r between matching columns of A and B, over rows finite in both."""
    ok = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
    a, b = A[ok].astype(np.float64), B[ok].astype(np.float64)
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ac * bc).sum(axis=0) / np.sqrt((ac**2).sum(axis=0) * (bc**2).sum(axis=0))


def predictions(npz, which: str) -> np.ndarray:
    """Rebuild the decoder output as Z @ W.T.

    Exact up to the decoder bias, which is not stored and is not needed anywhere here: it is a
    per-gene constant, so it cancels in a correlation and in a difference of group means.
    """
    return npz[f"Z_{which}"].astype(np.float64) @ npz[f"W_{which}"].astype(np.float64).T


def group_contrast(y: np.ndarray, groups: np.ndarray, group: str, total=None) -> np.ndarray:
    """Per-gene mean(group) - mean(rest). On log1p data this is a log fold change.

    `total` is y.sum(axis=0), which the caller can compute once: with it, the cost is O(k) in the
    GROUP size rather than O(n) in the whole matrix, because mean(rest) comes from the difference.
    That matters on melton25, where y is 386727 x 979 and the permutation null calls this
    thousands of times -- masking the complement each time reads ~3 GB per call.
    """
    m = groups == group
    k = int(m.sum())
    n = y.shape[0]
    if k == 0 or k == n:
        return np.full(y.shape[1], np.nan)
    s_group = y[m].sum(axis=0)
    s_total = y.sum(axis=0) if total is None else total
    return s_group / k - (s_total - s_group) / (n - k)


def top_k(v: np.ndarray, k: int) -> set:
    return set(np.argsort(np.abs(v))[::-1][:k].tolist())


def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else np.nan


def co_membership(labels: np.ndarray) -> np.ndarray:
    """Boolean vector over gene pairs: do the two genes share a cluster."""
    return offdiag((labels[:, None] == labels[None, :]).astype(np.int8)).astype(bool)


def cluster_genes(S: np.ndarray, k: int) -> np.ndarray:
    """Cluster genes on the correlation distance between their loading DIRECTIONS."""
    R = row_normalize(np.asarray(S, dtype=np.float64))
    D = np.clip(1.0 - R @ R.T, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=k, criterion="maxclust")


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    """ARI, computed from the contingency table (no sklearn dependency)."""
    ua, ub = np.unique(a, return_inverse=True)[1], np.unique(b, return_inverse=True)[1]
    n = len(ua)
    cont = np.zeros((ua.max() + 1, ub.max() + 1), dtype=np.int64)
    np.add.at(cont, (ua, ub), 1)
    comb2 = lambda x: (x * (x - 1) / 2).sum()  # noqa: E731
    sum_ij = comb2(cont)
    sum_i = comb2(cont.sum(axis=1))
    sum_j = comb2(cont.sum(axis=0))
    total = n * (n - 1) / 2
    expected = sum_i * sum_j / total
    maximum = 0.5 * (sum_i + sum_j)
    return float((sum_ij - expected) / (maximum - expected)) if maximum != expected else np.nan


# --------------------------------------------------------------------------------------------
# the six analyses
# --------------------------------------------------------------------------------------------
def geometry_rows(data, seeds, rng, n_perm):
    """(1) + (2): gene-gene co-loading geometry and CKA, against a gene-permutation null."""
    rows = []
    for which in COMPONENTS:
        grams = {s: gene_gram(data[s][0][f"S_{which}"]) for s in seeds}
        vecs = {s: offdiag(grams[s]) for s in seeds}
        for a, b in itertools.combinations(seeds, 2):
            obs = float(spearmanr(vecs[a], vecs[b]).statistic)
            # Null: relabel the genes of b. This keeps b's geometry intact and asks only whether
            # the SAME genes occupy the same relative positions in both runs.
            null = []
            Sb = np.asarray(data[b][0][f"S_{which}"], dtype=np.float64)
            for _ in range(n_perm):
                p = rng.permutation(Sb.shape[0])
                null.append(float(spearmanr(vecs[a], offdiag(gene_gram(Sb[p]))).statistic))
            rows.append(
                {
                    "component": which,
                    "seed_a": a,
                    "seed_b": b,
                    "spearman_gene_gram": obs,
                    "spearman_gene_gram_null": float(np.mean(null)),
                    "spearman_gene_gram_null_sd": float(np.std(null, ddof=1)),
                    "z_vs_null": (obs - np.mean(null)) / np.std(null, ddof=1) if np.std(null) > 0 else np.nan,
                }
            )
            # CKA on raw S is NOT artifact-free: S rows have wildly different norms (1/std(x_g)
            # spans orders of magnitude), so it largely measures agreement about which genes have
            # big loadings -- the same shared denominator. Reported alongside the row-normalised
            # and raw-W versions so the contamination is visible rather than assumed away.
            Sa = np.asarray(data[a][0][f"S_{which}"], dtype=np.float64)
            for name, Ma, Mb in (
                ("S", Sa, Sb),
                ("S_rownorm", row_normalize(Sa), row_normalize(Sb)),
                ("W", np.asarray(data[a][0][f"W_{which}"], dtype=np.float64),
                 np.asarray(data[b][0][f"W_{which}"], dtype=np.float64)),
            ):
                rows[-1][f"cka_{name}"] = linear_cka(Ma, Mb)
                rows[-1][f"cka_{name}_null"] = float(
                    np.mean([linear_cka(Ma, Mb[rng.permutation(Mb.shape[0])]) for _ in range(n_perm)])
                )
    return rows


def prediction_rows(data, seeds, y_true, var_names):
    """(3): per-gene agreement of the two seeds' predicted expression across cells."""
    preds = {(s, w): predictions(data[s][0], w) for s in seeds for w in COMPONENTS}
    rows = []
    cached = {}
    for which in COMPONENTS:
        for a, b in itertools.combinations(seeds, 2):
            r = columnwise_pearson(preds[(a, which)], preds[(b, which)])
            cached[(which, a, b)] = r
            # Null: correlate seed b's prediction for a DIFFERENT gene. Predictions of any two
            # genes share the same low-dimensional Z, so they are correlated by construction --
            # this is the only honest baseline.
            shifted = np.roll(preds[(b, which)], 1, axis=1)
            r_null = columnwise_pearson(preds[(a, which)], shifted)
            rows.append(
                {
                    "component": which,
                    "seed_a": a,
                    "seed_b": b,
                    "median_abs_r": float(np.nanmedian(np.abs(r))),
                    "median_r": float(np.nanmedian(r)),
                    "frac_abs_r_gt_0.5": float(np.nanmean(np.abs(r) > 0.5)),
                    "median_abs_r_gene_shifted_null": float(np.nanmedian(np.abs(r_null))),
                }
            )
    per_gene = {}
    for which in COMPONENTS:
        # Reuse the r vectors already computed above rather than recorrelating every pair: on
        # melton25 each call reads two 3 GB matrices.
        M = np.vstack([np.abs(cached[(which, a, b)] ) for a, b in itertools.combinations(seeds, 2)])
        per_gene[which] = pd.DataFrame(
            {"gene": var_names, "component": which,
             "mean_abs_r_across_seed_pairs": M.mean(axis=0),
             "min_abs_r_across_seed_pairs": M.min(axis=0)}
        )
    return rows, pd.concat(per_gene.values(), ignore_index=True)


def contrast_rows(data, seeds, y_true, groups, var_names, top_n, rng, n_label_perm):
    """(4): the up/down-regulation check, cross-seed AND against the observed data."""
    levels = [g for g in pd.unique(groups) if (groups == g).sum() >= 20]
    preds = {(s, w): predictions(data[s][0], w) for s in seeds for w in COMPONENTS}
    # One pass each, reused by every group and every permutation below.
    totals = {k: v.sum(axis=0) for k, v in preds.items()}
    y_true64 = y_true.astype(np.float64)
    true_total = y_true64.sum(axis=0)

    pair_rows, truth_rows, per_gene = [], [], []
    for group in levels:
        true_lfc = group_contrast(y_true64, groups, group, total=true_total)
        # The genes the contrast is actually about; sign agreement over all genes is dominated by
        # genes whose true contrast is ~0, where a sign is noise by definition.
        focus = np.argsort(np.abs(true_lfc))[::-1][:top_n]

        for which in COMPONENTS:
            lfc = {s: group_contrast(preds[(s, which)], groups, group, total=totals[(s, which)])
                   for s in seeds}

            for a, b in itertools.combinations(seeds, 2):
                sa, sb = lfc[a], lfc[b]
                pair_rows.append(
                    {
                        "group": group,
                        "component": which,
                        "seed_a": a,
                        "seed_b": b,
                        "spearman_lfc": float(spearmanr(sa, sb).statistic),
                        "sign_agreement_all": float(np.mean(np.sign(sa) == np.sign(sb))),
                        f"sign_agreement_top{top_n}_true": float(
                            np.mean(np.sign(sa[focus]) == np.sign(sb[focus]))
                        ),
                    }
                )

            for s in seeds:
                truth_rows.append(
                    {
                        "group": group,
                        "component": which,
                        "seed": s,
                        "spearman_vs_observed": float(spearmanr(lfc[s], true_lfc).statistic),
                        "sign_agreement_vs_observed_all": float(
                            np.mean(np.sign(lfc[s]) == np.sign(true_lfc))
                        ),
                        f"sign_agreement_vs_observed_top{top_n}": float(
                            np.mean(np.sign(lfc[s][focus]) == np.sign(true_lfc[focus]))
                        ),
                    }
                )

            L = np.vstack([lfc[s] for s in seeds])

            # The recommendation this analysis exists to test. If each seed recovers a weak but
            # REAL projection of the group contrast and their errors are independent, the
            # cross-seed mean must track the observed contrast better than any single seed --
            # and pairwise seed agreement (~rho^2) must be far below either. Averaging is only
            # legitimate here because a contrast is sign-unambiguous, unlike a loading vector.
            ens = L.mean(axis=0)
            truth_rows.append(
                {
                    "group": group,
                    "component": which,
                    "seed": "ensemble5",
                    "spearman_vs_observed": float(spearmanr(ens, true_lfc).statistic),
                    "sign_agreement_vs_observed_all": float(np.mean(np.sign(ens) == np.sign(true_lfc))),
                    f"sign_agreement_vs_observed_top{top_n}": float(
                        np.mean(np.sign(ens[focus]) == np.sign(true_lfc[focus]))
                    ),
                }
            )

            # Null for the headline number: recompute the SAME ensemble contrast after shuffling
            # which cell belongs to which group, and score it against the unshuffled observed
            # contrast. This destroys the spatial grouping while leaving the predictions, the
            # group sizes and the averaging over seeds exactly as they are, so anything the
            # ensemble scores above this is genuinely group-specific.
            null_rho, null_sign = [], []
            for _ in range(n_label_perm):
                g_perm = groups[rng.permutation(len(groups))]
                Lp = np.vstack([
                    group_contrast(preds[(s, which)], g_perm, group, total=totals[(s, which)])
                    for s in seeds
                ])
                ep = Lp.mean(axis=0)
                null_rho.append(float(spearmanr(ep, true_lfc).statistic))
                null_sign.append(float(np.mean(np.sign(ep[focus]) == np.sign(true_lfc[focus]))))
            truth_rows.append(
                {
                    "group": group,
                    "component": which,
                    "seed": "ensemble5_shuffled_labels",
                    "spearman_vs_observed": float(np.mean(null_rho)),
                    "sign_agreement_vs_observed_all": np.nan,
                    f"sign_agreement_vs_observed_top{top_n}": float(np.mean(null_sign)),
                    "null_rho_sd": float(np.std(null_rho, ddof=1)),
                }
            )

            per_gene.append(
                pd.DataFrame(
                    {
                        "gene": var_names,
                        "group": group,
                        "component": which,
                        "true_lfc": true_lfc,
                        "pred_lfc_mean": L.mean(axis=0),
                        "pred_lfc_sd": L.std(axis=0, ddof=1),
                        # 5/5 seeds agreeing on a direction is the statement a reviewer can read.
                        "n_seeds_positive": (L > 0).sum(axis=0),
                        "unanimous": np.abs((L > 0).sum(axis=0) - len(seeds) / 2) == len(seeds) / 2,
                        "direction_matches_observed": np.sign(L.mean(axis=0)) == np.sign(true_lfc),
                    }
                )
            )
    return pair_rows, truth_rows, pd.concat(per_gene, ignore_index=True)


def membership_rows(data, seeds, k, rng, n_perm):
    """(5): do the same gene PAIRS cluster together across seeds."""
    rows = []
    for which in COMPONENTS:
        labels = {s: cluster_genes(data[s][0][f"S_{which}"], k) for s in seeds}
        for a, b in itertools.combinations(seeds, 2):
            la, lb = labels[a], labels[b]
            ca, cb = co_membership(la), co_membership(lb)
            obs = jaccard(set(np.flatnonzero(ca)), set(np.flatnonzero(cb)))
            null_ari, null_jac = [], []
            for _ in range(n_perm):
                p = rng.permutation(len(lb))
                null_ari.append(adjusted_rand(la, lb[p]))
                null_jac.append(jaccard(set(np.flatnonzero(ca)), set(np.flatnonzero(co_membership(lb[p])))))
            rows.append(
                {
                    "component": which,
                    "seed_a": a,
                    "seed_b": b,
                    "k": k,
                    "ari": adjusted_rand(la, lb),
                    "ari_null": float(np.mean(null_ari)),
                    "pair_jaccard": obs,
                    "pair_jaccard_null": float(np.mean(null_jac)),
                }
            )
    return rows


def topk_artifact_rows(data, seeds, k, rng, n_shuffle):
    """(6): how much of the original top-k Jaccard is the 1/std(x_g) denominator.

    Three parameterisations of the same decoder, each compared BOTH at matched dimension index
    and at shuffled dimension pairing. `S` is what 4b used; `W` drops the standardization
    entirely; `S_rownorm` keeps std(z_k) but divides out std(x_g). Dimensions are compared at
    equal index only as a cheap stand-in for matching -- the point here is the gap between the
    matched and shuffled columns, which no matching scheme can manufacture.
    """
    rows = []
    for which in COMPONENTS:
        variants = {
            "S": {s: np.asarray(data[s][0][f"S_{which}"], dtype=np.float64) for s in seeds},
            "W": {s: np.asarray(data[s][0][f"W_{which}"], dtype=np.float64) for s in seeds},
            "S_rownorm": {s: row_normalize(np.asarray(data[s][0][f"S_{which}"], dtype=np.float64)) for s in seeds},
        }
        for name, mats in variants.items():
            n_dims = mats[seeds[0]].shape[1]
            matched, shuffled = [], []
            for a, b in itertools.combinations(seeds, 2):
                A, B = mats[a], mats[b]
                matched += [jaccard(top_k(A[:, d], k), top_k(B[:, d], k)) for d in range(n_dims)]
                for _ in range(n_shuffle):
                    for d in range(n_dims):
                        e = int(rng.choice([j for j in range(n_dims) if j != d]))
                        shuffled.append(jaccard(top_k(A[:, d], k), top_k(B[:, e], k)))
            rows.append(
                {
                    "component": which,
                    "parameterisation": name,
                    f"jaccard_top{k}_matched_index": float(np.nanmean(matched)),
                    f"jaccard_top{k}_shuffled_dims": float(np.nanmean(shuffled)),
                    "excess_over_shuffled": float(np.nanmean(matched) - np.nanmean(shuffled)),
                }
            )
    return rows


def drop_diagonal(M: np.ndarray) -> np.ndarray:
    """(n, n) -> (n, n-1), each row with its own diagonal entry removed."""
    n = M.shape[0]
    return M[~np.eye(n, dtype=bool)].reshape(n, n - 1)


def rowwise_spearman(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Spearman correlation between matching ROWS of A and B.

    Ranks each row once and then takes a row-wise Pearson, which is the same quantity
    scipy.stats.spearmanr returns but vectorised -- 979 genes x 10 seed pairs x 2 components is
    ~20k correlations, and calling spearmanr in a loop for those dominates the whole script.
    """
    from scipy.stats import rankdata

    Ra = rankdata(A, axis=1).astype(np.float64)
    Rb = rankdata(B, axis=1).astype(np.float64)
    Ra -= Ra.mean(axis=1, keepdims=True)
    Rb -= Rb.mean(axis=1, keepdims=True)
    num = (Ra * Rb).sum(axis=1)
    den = np.sqrt((Ra**2).sum(axis=1) * (Rb**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def loading_stability_per_gene(data, seeds, var_names):
    """(8) Per-gene loading stability -- the per-gene version of the gene-gram statistic.

    For each gene, its ROW of the gene-gene co-loading matrix is its similarity profile against
    every other gene. That profile is invariant to any orthogonal change of latent basis (so no
    dimension matching, no sign convention) and, because the matrix is built from row-normalised
    loadings, free of the 1/std(x_g) factor. Correlating gene g's profile between two seeds asks
    exactly: does THIS gene sit in the same place in loading space when the model is retrained?

    The null is the same correlation against a DIFFERENT gene's profile in the second seed. Some
    profile shape is shared by every gene (they all live in the same low-rank space), so a
    per-gene number has to be read against that, not against zero.
    """
    out = []
    for which in COMPONENTS:
        profiles = {s: drop_diagonal(gene_gram(data[s][0][f"S_{which}"])) for s in seeds}
        obs, null = [], []
        for a, b in itertools.combinations(seeds, 2):
            obs.append(rowwise_spearman(profiles[a], profiles[b]))
            # roll by one gene: gene g in seed a against gene g+1 in seed b
            null.append(rowwise_spearman(profiles[a], np.roll(profiles[b], 1, axis=0)))
        O, N = np.vstack(obs), np.vstack(null)
        out.append(
            pd.DataFrame(
                {
                    "gene": var_names,
                    "component": which,
                    "profile_rho_mean": O.mean(axis=0),
                    "profile_rho_sd": O.std(axis=0, ddof=1),
                    "profile_rho_min": O.min(axis=0),
                    "profile_rho_null_mean": N.mean(axis=0),
                    "excess_over_null": O.mean(axis=0) - N.mean(axis=0),
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def attribution_rows(data, seeds, var_names):
    """(7) Is the local-vs-global assignment of a gene reproducible? The Fig. 3E/3F claim.

    Measured directly on how well each decoder predicts that gene -- delta_g = r_local(g) -
    r_global(g) -- rather than on the rank transform 4b uses. Needs no dimension matching and no
    sign convention, so a null result here cannot be blamed on the alignment step.

    Also reports leave-one-out stability: the correlation between the attribution averaged over
    four seeds and the held-out fifth. That is the honest test of "can I read a gene's component
    off an ensemble", and it is the number to quote if the per-pair agreement is at chance.
    """
    delta = {s: data[s][0]["pearson_local"] - data[s][0]["pearson_global"] for s in seeds}
    D = np.vstack([delta[s] for s in seeds])

    pair_rows = []
    for a, b in itertools.combinations(seeds, 2):
        m = np.isfinite(delta[a]) & np.isfinite(delta[b])
        pair_rows.append(
            {
                "seed_a": a,
                "seed_b": b,
                "spearman_delta_pearson": float(spearmanr(delta[a][m], delta[b][m]).statistic),
                "sign_agreement": float(np.mean(np.sign(delta[a][m]) == np.sign(delta[b][m]))),
            }
        )

    loo = []
    for i, s in enumerate(seeds):
        keep = [j for j in range(len(seeds)) if j != i]
        m = np.isfinite(D[keep].mean(axis=0)) & np.isfinite(D[i])
        loo.append(
            {
                "held_out_seed": s,
                "spearman_mean_of_others_vs_held_out": float(
                    spearmanr(D[keep].mean(axis=0)[m], D[i][m]).statistic
                ),
            }
        )

    unanimous = (D > 0).all(axis=0) | (D < 0).all(axis=0)
    per_gene = pd.DataFrame(
        {
            "gene": var_names,
            "delta_pearson_mean": D.mean(axis=0),
            "delta_pearson_sd": D.std(axis=0, ddof=1),
            "n_seeds_favour_local": (D > 0).sum(axis=0),
            "all_seeds_agree": unanimous,
            "favoured_component": np.where(D.mean(axis=0) > 0, "local", "global"),
        }
    )
    return pair_rows, loo, per_gene


# --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="legnini23", choices=sorted(SPECS), help="Which dataset's seed run to analyse.")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--arrays-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-perm", type=int, default=200, help="Gene-permutation draws for the geometry nulls.")
    ap.add_argument("--n-shuffle", type=int, default=5, help="Random dimension pairings per seed pair.")
    ap.add_argument("--k-clusters", type=int, default=8, help="Gene clusters for the co-membership check.")
    ap.add_argument("--top-n", type=int, default=20, help="Genes with the largest |observed contrast| to focus on.")
    ap.add_argument("--top-k-genes", type=int, default=20, help="k for the top-k Jaccard diagnostic.")
    ap.add_argument("--n-label-perm", type=int, default=50,
                    help="Group-label shuffles for the ensemble-contrast null.")
    args = ap.parse_args()

    spec = SPECS[args.spec]
    seeds = args.seeds or spec["seeds"]
    arrays_dir = Path(args.arrays_dir or spec["arrays_dir"])
    out_dir = Path(args.out_dir or spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    print(f"=== {args.spec}: basis-free cross-seed stability ===")
    print(f"arrays {arrays_dir}\nout    {out_dir}")

    data = {s: load_seed(s, arrays_dir) for s in seeds}
    var_names = np.asarray(data[seeds[0]][0]["var_names"], dtype=object).astype(str)
    for s in seeds:
        assert list(np.asarray(data[s][0]["var_names"]).astype(str)) == list(var_names), (
            f"seed {s}: gene order differs"
        )

    y_true, groups, h5_var, n_cells = load_h5ad_parts(Path(spec["h5ad"]), spec["truth"], spec["group_key"])
    assert list(h5_var) == list(var_names), "gene order differs between the h5ad and the stored arrays"
    assert n_cells == data[seeds[0]][0]["Z_local"].shape[0], (
        f"cell count mismatch: h5ad filtered to {n_cells}, arrays have "
        f"{data[seeds[0]][0]['Z_local'].shape[0]} -- the zero-expression filter did not reproduce"
    )
    print(f"{n_cells} cells, {len(var_names)} genes, {len(seeds)} seeds, "
          f"groups({spec['group_key']}): {dict(zip(*np.unique(groups, return_counts=True)))}")

    # The self-check that licenses rebuilding predictions as Z @ W.T at all.
    for which in COMPONENTS:
        r = columnwise_pearson(y_true.astype(np.float64), predictions(data[seeds[0]][0], which))
        stored = data[seeds[0]][0][f"pearson_{which}"]
        ok = np.isfinite(r) & np.isfinite(stored)
        dev = float(np.max(np.abs(r[ok] - stored[ok])))
        assert dev < 1e-5, f"{which}: Z @ W.T does not reproduce the stored per-gene Pearson (max dev {dev:.2e})"
    print("Z @ W.T reproduces the stored per-gene Pearson for both decoders (< 1e-5)")

    # --- (1)+(2) geometry ------------------------------------------------------------------
    geom = pd.DataFrame(geometry_rows(data, seeds, rng, args.n_perm))
    geom.to_csv(out_dir / "geom_gene_gram.csv", index=False)
    print("\n--- (1)+(2) gene-gene co-loading geometry (rotation/sign/permutation invariant) ---")
    print(
        geom.groupby("component")[
            ["spearman_gene_gram", "spearman_gene_gram_null", "z_vs_null",
             "cka_S", "cka_S_null", "cka_S_rownorm", "cka_S_rownorm_null", "cka_W", "cka_W_null"]
        ]
        .mean()
        .to_string(float_format=lambda v: f"{v:+.4f}")
    )

    # --- (3) predicted patterns ------------------------------------------------------------
    pred_rows, pred_per_gene = prediction_rows(data, seeds, y_true, var_names)
    pred = pd.DataFrame(pred_rows)
    pred.to_csv(out_dir / "geom_prediction_agreement.csv", index=False)
    pred_per_gene.to_csv(out_dir / "geom_prediction_agreement_per_gene.csv", index=False)
    print("\n--- (3) per-gene agreement of PREDICTED expression across seeds (no matching) ---")
    print(
        pred.groupby("component")[
            ["median_abs_r", "median_r", "frac_abs_r_gt_0.5", "median_abs_r_gene_shifted_null"]
        ]
        .mean()
        .to_string(float_format=lambda v: f"{v:+.4f}")
    )

    # --- (4) up/down regulation ------------------------------------------------------------
    c_pair, c_truth, c_gene = contrast_rows(
        data, seeds, y_true, groups, var_names, args.top_n, rng, args.n_label_perm
    )
    cp, ct = pd.DataFrame(c_pair), pd.DataFrame(c_truth)
    cp.to_csv(out_dir / "geom_contrast_cross_seed.csv", index=False)
    ct.to_csv(out_dir / "geom_contrast_vs_observed.csv", index=False)
    c_gene.to_csv(out_dir / "geom_contrast_per_gene.csv", index=False)
    print(f"\n--- (4a) cross-seed agreement of the predicted {spec['group_key']} contrast ---")
    print(cp.groupby(["component", "group"])[
        ["spearman_lfc", "sign_agreement_all", f"sign_agreement_top{args.top_n}_true"]
    ].mean().to_string(float_format=lambda v: f"{v:+.4f}"))
    cols = ["spearman_vs_observed", "sign_agreement_vs_observed_all",
            f"sign_agreement_vs_observed_top{args.top_n}"]
    seed_str = ct["seed"].astype(str)
    is_ens = seed_str == "ensemble5"
    is_null = seed_str == "ensemble5_shuffled_labels"
    print("\n--- (4b) agreement of that contrast with the OBSERVED data: single seed vs 5-seed mean ---")
    merged = (
        ct[~is_ens & ~is_null].groupby(["component", "group"])[cols].mean()
        .join(ct[is_ens].set_index(["component", "group"])[cols], lsuffix="_1seed", rsuffix="_ens5")
        .join(ct[is_null].set_index(["component", "group"])[cols].add_suffix("_null"))
    )
    print(merged[[f"{c}{sfx}" for c in cols for sfx in ("_1seed", "_ens5", "_null")]]
          .to_string(float_format=lambda v: f"{v:+.4f}"))

    # --- (5) pairings ----------------------------------------------------------------------
    mem = pd.DataFrame(membership_rows(data, seeds, args.k_clusters, rng, max(20, args.n_perm // 10)))
    mem.to_csv(out_dir / "geom_gene_cluster_agreement.csv", index=False)
    print(f"\n--- (5) do the same gene PAIRS cluster together (k={args.k_clusters}) ---")
    print(mem.groupby("component")[["ari", "ari_null", "pair_jaccard", "pair_jaccard_null"]]
          .mean().to_string(float_format=lambda v: f"{v:+.4f}"))

    # --- (6) the artifact ------------------------------------------------------------------
    art = pd.DataFrame(topk_artifact_rows(data, seeds, args.top_k_genes, rng, args.n_shuffle))
    art.to_csv(out_dir / "geom_topk_artifact.csv", index=False)
    print(f"\n--- (6) how much of top-{args.top_k_genes} Jaccard is the 1/std(x_g) denominator ---")
    print(art.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # --- (8) per-gene loading stability ----------------------------------------------------
    ls = loading_stability_per_gene(data, seeds, var_names)
    ls.to_csv(out_dir / "geom_loading_stability_per_gene.csv", index=False)
    print("\n--- (8) per-gene loading stability (gene's co-loading profile, across seeds) ---")
    print(
        ls.groupby("component")[["profile_rho_mean", "profile_rho_null_mean", "excess_over_null"]]
        .mean()
        .to_string(float_format=lambda v: f"{v:+.4f}")
    )
    for which in COMPONENTS:
        sub = ls[ls["component"] == which]
        frac = float((sub["excess_over_null"] > 0.1).mean())
        print(f"  {which:6s} genes with excess over null > 0.10: {frac * 100:.1f}% "
              f"({int((sub['excess_over_null'] > 0.1).sum())}/{len(sub)}) | "
              f"most stable: {', '.join(sub.nlargest(6, 'profile_rho_mean')['gene'])}")

    # --- (7) the local-vs-global assignment ------------------------------------------------
    a_pair, a_loo, a_gene = attribution_rows(data, seeds, var_names)
    ap_df, al_df = pd.DataFrame(a_pair), pd.DataFrame(a_loo)
    ap_df.to_csv(out_dir / "geom_attribution_cross_seed.csv", index=False)
    a_gene.to_csv(out_dir / "geom_attribution_per_gene.csv", index=False)
    print("\n--- (7) is a gene's local-vs-global assignment reproducible (Fig 3E/3F claim) ---")
    print(f"cross-seed spearman of r_local - r_global : {ap_df['spearman_delta_pearson'].mean():+.4f} "
          f"(sd {ap_df['spearman_delta_pearson'].std(ddof=1):.3f})")
    print(f"cross-seed sign agreement                 : {ap_df['sign_agreement'].mean():.4f} "
          f"(chance 0.5)")
    print(f"leave-one-out: rho(mean of 4, held-out)   : "
          f"{al_df['spearman_mean_of_others_vs_held_out'].mean():+.4f}")
    print(f"genes where all {len(seeds)} seeds agree which component wins: "
          f"{a_gene['all_seeds_agree'].mean() * 100:.0f}% "
          f"({int(a_gene['all_seeds_agree'].sum())}/{len(a_gene)})")
    if a_gene["all_seeds_agree"].any():
        top = a_gene[a_gene["all_seeds_agree"]].reindex(
            a_gene[a_gene["all_seeds_agree"]]["delta_pearson_mean"].abs().sort_values(ascending=False).index
        )
        print("unanimous genes, largest |delta| first:")
        print(top.head(12).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # --- the one-line answers --------------------------------------------------------------
    print("\n=== headline ===")
    for which in COMPONENTS:
        g = geom[geom["component"] == which]
        p = pred[pred["component"] == which]
        print(
            f"{which:6s} gene-gram rho {g['spearman_gene_gram'].mean():+.3f} "
            f"(null {g['spearman_gene_gram_null'].mean():+.3f}, z {g['z_vs_null'].mean():+.1f}) | "
            f"CKA(rownorm) {g['cka_S_rownorm'].mean():.3f} (null {g['cka_S_rownorm_null'].mean():.3f}) | "
            f"pred |r| {p['median_abs_r'].mean():.3f} (null {p['median_abs_r_gene_shifted_null'].mean():.3f})"
        )
    unan = c_gene[c_gene["component"] == "global"]
    if len(unan):
        agree = unan.groupby("group").apply(
            lambda d: pd.Series({
                "unanimous": d["unanimous"].mean(),
                "unanimous_and_right": (d["unanimous"] & d["direction_matches_observed"]).mean(),
            }),
            include_groups=False,
        )
        print("\nglobal decoder, per group: fraction of genes where all seeds agree on the "
              "direction, and where that direction also matches the observed data")
        print(agree.to_string(float_format=lambda v: f"{v:.3f}"))

    print(f"\nwrote 10 csv files to {out_dir}")


if __name__ == "__main__":
    main()
