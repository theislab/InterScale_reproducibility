"""Cross-seed stability of things that are NOT the gene loadings.

`4d_loading_geometry_stability.py` establishes that the standardized loadings do not reproduce
across seeds beyond a small high-variance minority. This script asks whether the model
nevertheless learns something common, at three levels that sidestep the loading vector entirely:

  1. `clusters`   Cluster the local and global embeddings so every cell gets a label, then score
                  the agreement of those labellings across seeds (AMI / ARI / NMI / Fowlkes-
                  Mallows). A latent basis is defined only up to rotation, but a PARTITION of the
                  cells is not -- it is basis-free by construction, so this needs no matching and
                  no sign convention. Also scored against the biological grouping
                  (`cell_type_coarse` / `SHH_zone`), because two seeds agreeing on a partition
                  that has nothing to do with biology would not be worth much.
  2. `dim_panels` Take the most important dimensions (importance-ordered `dims_left` from
                  `calculate_dim_importance`), collect their top genes, and intersect those panels
                  across seeds. This is the Fig. 3F object, so it is the intersection a reader of
                  that figure would care about. Reported against a gene-permutation null, and
                  additionally on row-normalised loadings so the 1/std(x_g) artifact cannot
                  inflate it.
  3. `gene_ranks` Rank genes by how well each decoder reconstructs them (`calculate_gene_ranks`'
                  R2 rank; rank G = best) and sweep the size N of the top-N set, reporting what
                  fraction of it is shared by ALL five seeds. The point of the sweep is to find
                  the N at which the shared fraction clears a target (default 50%) -- and to show
                  the chance curve next to it, because the shared fraction rises to 1 at N =
                  n_genes for free. Also intersects the top-N by LOCAL rank against the top-N by
                  GLOBAL rank, which says whether the two decoders even prefer different genes.

Everything reads the arrays `4a_seed_stability_extract.py` already wrote. Clustering is the only
expensive part (386727 x 128 on melton25); the rest is seconds.

Usage
-----
    python scripts/legnini_23/4e_embedding_and_rank_stability.py --spec melton25
    python scripts/legnini_23/4e_embedding_and_rank_stability.py --spec melton25 --k 5 9 15 25
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path

# MUST precede the numpy import. These nodes report >100 cores, and OpenBLAS is built for a
# maximum of 128 threads: KMeans with several restarts then allocates a thread pool per restart
# and the process dies with "Program is Terminated. Because you tried to allocate too many memory
# regions." before printing anything useful. setdefault, so an explicit outer setting still wins.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent


def _load_geom():
    """Import 4d by file path -- its leading digit makes it an invalid module name to import."""
    spec = importlib.util.spec_from_file_location(
        "geom4d", _HERE / "4d_loading_geometry_stability.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


geom = _load_geom()
SPECS, COMPONENTS = geom.SPECS, geom.COMPONENTS


# --------------------------------------------------------------------------------------------
# 1. embedding cluster coherence
# --------------------------------------------------------------------------------------------
def cluster_embeddings(data, seeds, ks, groups, rng_seed=0, batch_size=4096, n_init=3):
    """Cluster each seed's embedding at several k, and score cross-seed label agreement.

    KMeans is itself stochastic, so `random_state` is pinned to the SAME value for every seed:
    any disagreement then comes from the embeddings rather than from the clusterer. Embeddings are
    clustered as-is, not standardized -- the dimension scales are part of the representation being
    tested, and z-scoring them would hand every dead dimension equal weight.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        fowlkes_mallows_score,
        normalized_mutual_info_score,
    )

    rng = np.random.default_rng(rng_seed)
    pair_rows, bio_rows = [], []

    for which in COMPONENTS:
        # Read each embedding ONCE. np.load on an npz is lazy and decompresses on every access,
        # so indexing Z_{which} inside the k loop would re-inflate a 200 MB array per k.
        Zs = {s: np.asarray(data[s][0][f"Z_{which}"], dtype=np.float32) for s in seeds}
        oks = {s: np.isfinite(Z).all(axis=1) for s, Z in Zs.items()}

        for k in ks:
            labels = {}
            for s in seeds:
                Z, ok = Zs[s], oks[s]
                lab = np.full(Z.shape[0], -1, dtype=np.int32)
                km = MiniBatchKMeans(
                    n_clusters=k, random_state=rng_seed, batch_size=batch_size, n_init=n_init
                )
                lab[ok] = km.fit_predict(Z[ok])
                labels[s] = lab

            # Cross-seed agreement, on the cells both seeds actually embedded.
            for a, b in itertools.combinations(seeds, 2):
                m = (labels[a] >= 0) & (labels[b] >= 0)
                la, lb = labels[a][m], labels[b][m]
                perm = lb[rng.permutation(len(lb))]
                pair_rows.append(
                    {
                        "component": which,
                        "k": k,
                        "seed_a": a,
                        "seed_b": b,
                        "n_cells": int(m.sum()),
                        "ami": adjusted_mutual_info_score(la, lb),
                        "nmi": normalized_mutual_info_score(la, lb),
                        "ari": adjusted_rand_score(la, lb),
                        "fowlkes_mallows": fowlkes_mallows_score(la, lb),
                        # AMI/ARI are already chance-corrected; the shuffled column is carried so
                        # the correction can be seen to work rather than trusted.
                        "ami_shuffled": adjusted_mutual_info_score(la, perm),
                        "ari_shuffled": adjusted_rand_score(la, perm),
                        "nmi_shuffled": normalized_mutual_info_score(la, perm),
                    }
                )

            # Agreement with biology. A partition that is reproducible but unrelated to cell
            # identity would be a much weaker claim, so this is reported next to it.
            for s in seeds:
                m = labels[s] >= 0
                bio_rows.append(
                    {
                        "component": which,
                        "k": k,
                        "seed": s,
                        "ami_vs_biology": adjusted_mutual_info_score(groups[m], labels[s][m]),
                        "ari_vs_biology": adjusted_rand_score(groups[m], labels[s][m]),
                        "nmi_vs_biology": normalized_mutual_info_score(groups[m], labels[s][m]),
                    }
                )

        del Zs, oks

    return pd.DataFrame(pair_rows), pd.DataFrame(bio_rows)


# --------------------------------------------------------------------------------------------
# 2. top genes of the top-importance dimensions
# --------------------------------------------------------------------------------------------
def dim_panel_overlap(data, seeds, var_names, n_dims_list, n_top, rng, n_perm):
    """Intersect the top-gene panels of the most important dimensions, across seeds.

    Two sources, deliberately:
      * `meta` -- `top_genes_per_dim`, i.e. what `get_genes_dim` produced for Fig. 3F, including
        its specificity filter. This is the figure's own object.
      * `rownorm` -- top |S| per dimension after row-normalising the loadings, which removes the
        1/std(x_g) factor. If the two disagree, the figure's panel is carrying the artifact.
    """
    n_genes = len(var_names)
    gene_ix = {g: i for i, g in enumerate(var_names)}
    rows = []

    for which in COMPONENTS:
        # dims_left is importance-ordered (dim_sorted[:n]), so [:T] is the top-T dimensions.
        dims = {s: np.asarray(data[s][1][f"dims_{which}"], dtype=int) for s in seeds}
        tgpd = {s: data[s][1]["top_genes_per_dim"][which] for s in seeds}
        S = {s: geom.row_normalize(np.asarray(data[s][0][f"S_{which}"], dtype=np.float64))
             for s in seeds}

        for T in n_dims_list:
            panels = {"meta": {}, "rownorm": {}}
            for s in seeds:
                top_dims = dims[s][:T]
                meta_set = set()
                for d in top_dims:
                    meta_set |= set(tgpd[s].get(str(int(d)), [])[:n_top])
                panels["meta"][s] = {gene_ix[g] for g in meta_set if g in gene_ix}
                rn = set()
                for d in top_dims:
                    rn |= geom.top_k(S[s][:, int(d)], n_top)
                panels["rownorm"][s] = rn

            for source, P in panels.items():
                sizes = [len(P[s]) for s in seeds]
                inter_all = set.intersection(*(P[s] for s in seeds))
                union_all = set.union(*(P[s] for s in seeds))
                jac = [geom.jaccard(P[a], P[b]) for a, b in itertools.combinations(seeds, 2)]

                # Null: same panel sizes, genes drawn at random.
                null_j, null_i = [], []
                for _ in range(n_perm):
                    draws = [set(rng.choice(n_genes, size=len(P[s]), replace=False).tolist())
                             for s in seeds]
                    null_j.append(np.mean([geom.jaccard(draws[i], draws[j])
                                           for i, j in itertools.combinations(range(len(seeds)), 2)]))
                    null_i.append(len(set.intersection(*draws)))

                rows.append(
                    {
                        "component": which,
                        "source": source,
                        "n_top_dims": T,
                        "n_top_genes_per_dim": n_top,
                        "panel_size_mean": float(np.mean(sizes)),
                        "panel_size_min": int(min(sizes)),
                        "panel_size_max": int(max(sizes)),
                        "n_in_all_5_seeds": len(inter_all),
                        "n_in_union": len(union_all),
                        "frac_of_mean_panel_in_all_5": len(inter_all) / np.mean(sizes) if np.mean(sizes) else np.nan,
                        "mean_pairwise_jaccard": float(np.mean(jac)),
                        "mean_pairwise_jaccard_null": float(np.mean(null_j)),
                        "n_in_all_5_null": float(np.mean(null_i)),
                        "genes_in_all_5": ";".join(sorted(var_names[i] for i in inter_all)),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# 3. gene-rank overlap, swept over N
# --------------------------------------------------------------------------------------------
def rank_overlap(data, seeds, var_names, target, step):
    """Sweep the top-N size and report what fraction of it all five seeds share.

    `calculate_gene_ranks` ranks by per-gene R2 with rank = n_genes being the BEST reconstructed
    gene, so "top-N" is the N largest ranks.

    The shared fraction reaches 1 at N = n_genes for nothing, so the chance curve is reported
    alongside: for independent seeds, the expected 5-way shared fraction of a top-N set is
    (N/G)^4. `excess` is the difference, and `n_at_max_excess` is the principled choice of N,
    while `n_first_over_target` answers the "tune it to >50%" question directly.
    """
    n_genes = len(var_names)
    ranks = {}
    for s in seeds:
        z = data[s][0]
        ranks[(s, "local")] = np.asarray(z["local_rank"], dtype=float)
        ranks[(s, "global")] = np.asarray(z["global_rank"], dtype=float)
        ranks[(s, "avg")] = (ranks[(s, "local")] + ranks[(s, "global")]) / 2.0

    Ns = list(range(step, n_genes + 1, step))
    if Ns[-1] != n_genes:
        Ns.append(n_genes)

    curve, cross = [], []
    for key in ("avg", "local", "global"):
        for N in Ns:
            tops = [set(np.argsort(-ranks[(s, key)])[:N].tolist()) for s in seeds]
            inter = set.intersection(*tops)
            jac = [geom.jaccard(tops[i], tops[j])
                   for i, j in itertools.combinations(range(len(seeds)), 2)]
            chance = (N / n_genes) ** (len(seeds) - 1)
            curve.append(
                {
                    "ranking": key,
                    "N": N,
                    "frac_N": N / n_genes,
                    "n_in_all_5": len(inter),
                    "frac_in_all_5": len(inter) / N,
                    "frac_in_all_5_chance": chance,
                    "excess_over_chance": len(inter) / N - chance,
                    "mean_pairwise_jaccard": float(np.mean(jac)),
                }
            )
    curve = pd.DataFrame(curve)

    for key, sub in curve.groupby("ranking"):
        over = sub[sub["frac_in_all_5"] >= target]
        best = sub.loc[sub["excess_over_chance"].idxmax()]
        row = {
            "ranking": key,
            "target": target,
            "n_first_over_target": int(over["N"].min()) if len(over) else None,
            "frac_at_that_N": float(over.loc[over["N"].idxmin(), "frac_in_all_5"]) if len(over) else np.nan,
            "chance_at_that_N": float(over.loc[over["N"].idxmin(), "frac_in_all_5_chance"]) if len(over) else np.nan,
            "n_at_max_excess": int(best["N"]),
            "frac_at_max_excess": float(best["frac_in_all_5"]),
            "chance_at_max_excess": float(best["frac_in_all_5_chance"]),
            "max_excess": float(best["excess_over_chance"]),
        }
        cross.append(row)

    # Do the two decoders even prefer different genes? Within a seed, and across seeds.
    lg = []
    for N in Ns:
        within, across = [], []
        for s in seeds:
            tl = set(np.argsort(-ranks[(s, "local")])[:N].tolist())
            tg = set(np.argsort(-ranks[(s, "global")])[:N].tolist())
            within.append(geom.jaccard(tl, tg))
        for a, b in itertools.combinations(seeds, 2):
            tl = set(np.argsort(-ranks[(a, "local")])[:N].tolist())
            tg = set(np.argsort(-ranks[(b, "global")])[:N].tolist())
            across.append(geom.jaccard(tl, tg))
        lg.append(
            {
                "N": N,
                "jaccard_local_vs_global_same_seed": float(np.mean(within)),
                "jaccard_local_vs_global_across_seeds": float(np.mean(across)),
                "jaccard_chance": (N / n_genes) ** 2 / (2 * N / n_genes - (N / n_genes) ** 2),
            }
        )

    return curve, pd.DataFrame(cross), pd.DataFrame(lg)


def rank_agreement(data, seeds):
    """Cross-seed agreement of the gene ranking, before and after removing what the DATA explains.

    A gene's reconstruction rank is strongly predicted by how variably it is expressed, and
    expression variance is identical in every seed -- so some cross-seed rank agreement is
    inherited from the data rather than learned. This regresses each seed's rank on the per-gene
    expression sd (in rank space, so the fit is monotone-robust) and re-scores the agreement of
    the residuals. Agreement that survives is reproducible structure the MODEL contributes.
    """
    from scipy.stats import rankdata, spearmanr

    x_std = np.mean([data[s][0]["x_std"] for s in seeds], axis=0)
    rx = rankdata(x_std)
    rows = []
    for key in ("local_rank", "global_rank"):
        R = {s: np.asarray(data[s][0][key], dtype=float) for s in seeds}
        resid = {}
        for s in seeds:
            ry = rankdata(R[s])
            resid[s] = ry - np.polyval(np.polyfit(rx, ry, 1), rx)
        raw = [spearmanr(R[a], R[b]).statistic for a, b in itertools.combinations(seeds, 2)]
        res = [spearmanr(resid[a], resid[b]).statistic for a, b in itertools.combinations(seeds, 2)]
        rows.append(
            {
                "ranking": key.replace("_rank", ""),
                "rho_rank_vs_expression_sd": float(np.mean([spearmanr(R[s], x_std).statistic for s in seeds])),
                "cross_seed_rho_raw": float(np.mean(raw)),
                "cross_seed_rho_raw_sd": float(np.std(raw, ddof=1)),
                "cross_seed_rho_residual": float(np.mean(res)),
                "cross_seed_rho_residual_sd": float(np.std(res, ddof=1)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="legnini23", choices=sorted(SPECS))
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--arrays-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--k", type=int, nargs="+", default=None,
                    help="Cluster counts to try. Default: the number of biological groups, plus 5/15/25.")
    ap.add_argument("--n-top-dims", type=int, nargs="+", default=[1, 2, 3, 5, 10],
                    help="How many of the most important dimensions to pool genes from.")
    ap.add_argument("--n-top-genes", type=int, default=20, help="Top genes per dimension.")
    ap.add_argument("--rank-target", type=float, default=0.5,
                    help="Shared-fraction target the top-N sweep is tuned to.")
    ap.add_argument("--rank-step", type=int, default=5, help="Step of the top-N sweep.")
    ap.add_argument("--n-perm", type=int, default=200, help="Draws for the panel-overlap null.")
    ap.add_argument("--skip-clusters", action="store_true", help="Skip the expensive stage.")
    args = ap.parse_args()

    spec = SPECS[args.spec]
    seeds = args.seeds or spec["seeds"]
    arrays_dir = Path(args.arrays_dir or spec["arrays_dir"])
    out_dir = Path(args.out_dir or spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    print(f"=== {args.spec}: embedding / panel / rank stability ===")
    print(f"arrays {arrays_dir}\nout    {out_dir}", flush=True)

    data = {s: geom.load_seed(s, arrays_dir) for s in seeds}
    var_names = np.asarray(data[seeds[0]][0]["var_names"], dtype=object).astype(str)
    n_genes = len(var_names)
    print(f"{len(seeds)} seeds, {n_genes} genes, n_embed {data[seeds[0]][1]['n_embed']}", flush=True)

    # --- 1. embedding cluster coherence -----------------------------------------------------
    if not args.skip_clusters:
        _, groups, _, n_cells = geom.load_h5ad_parts(
            Path(spec["h5ad"]), spec["truth"], spec["group_key"]
        )
        n_bio = len(np.unique(groups))
        ks = args.k or sorted({5, n_bio, 15, 25})
        print(f"\nclustering {n_cells} cells at k={ks} "
              f"(biological groups: {n_bio} levels of {spec['group_key']})", flush=True)

        cp, cb = cluster_embeddings(data, seeds, ks, groups)
        cp.to_csv(out_dir / "emb_cluster_cross_seed.csv", index=False)
        cb.to_csv(out_dir / "emb_cluster_vs_biology.csv", index=False)

        print("\n--- (1) cross-seed agreement of the embedding CLUSTERINGS ---")
        print(
            cp.groupby(["component", "k"])[["ami", "ari", "nmi", "fowlkes_mallows",
                                           "ami_shuffled", "ari_shuffled"]]
            .mean().to_string(float_format=lambda v: f"{v:+.4f}")
        )
        print(f"\n--- (1b) do those clusters correspond to {spec['group_key']}? ---")
        print(
            cb.groupby(["component", "k"])[["ami_vs_biology", "ari_vs_biology", "nmi_vs_biology"]]
            .mean().to_string(float_format=lambda v: f"{v:+.4f}")
        )
    else:
        print("\n(1) clustering skipped")

    # --- 2. top-dimension gene panels -------------------------------------------------------
    dp = dim_panel_overlap(data, seeds, var_names, args.n_top_dims, args.n_top_genes, rng, args.n_perm)
    dp.to_csv(out_dir / "emb_dim_panel_overlap.csv", index=False)
    print("\n--- (2) top genes of the most important dimensions, intersected across seeds ---")
    show = ["component", "source", "n_top_dims", "panel_size_mean", "n_in_all_5_seeds",
            "n_in_all_5_null", "frac_of_mean_panel_in_all_5", "mean_pairwise_jaccard",
            "mean_pairwise_jaccard_null"]
    print(dp[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for _, r in dp[(dp["source"] == "meta") & (dp["n_in_all_5_seeds"] > 0)].iterrows():
        print(f"  {r['component']:6s} top-{int(r['n_top_dims'])} dims, in all 5 seeds: {r['genes_in_all_5']}")

    # --- 3. gene-rank overlap ---------------------------------------------------------------
    curve, cross, lg = rank_overlap(data, seeds, var_names, args.rank_target, args.rank_step)
    curve.to_csv(out_dir / "emb_rank_overlap_curve.csv", index=False)
    cross.to_csv(out_dir / "emb_rank_overlap_tuned.csv", index=False)
    lg.to_csv(out_dir / "emb_rank_local_vs_global.csv", index=False)

    print(f"\n--- (3) top-N gene-rank overlap, tuned to a shared fraction of {args.rank_target} ---")
    print(cross.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  curve at selected N (ranking = avg):")
    sel = curve[(curve["ranking"] == "avg")]
    keep = sel[sel["N"].isin([n for n in sel["N"] if n % max(args.rank_step, n_genes // 12) == 0 or n == n_genes])]
    print(keep[["N", "frac_N", "n_in_all_5", "frac_in_all_5", "frac_in_all_5_chance",
                "excess_over_chance", "mean_pairwise_jaccard"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    ra = rank_agreement(data, seeds)
    ra.to_csv(out_dir / "emb_rank_agreement.csv", index=False)
    print("\n--- (3c) is the rank agreement learned, or inherited from expression variance? ---")
    print(ra.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("  (residual = after regressing each seed's rank on per-gene expression sd, which is")
    print("   identical across seeds; agreement surviving that is contributed by the model)")

    print("\n--- (3b) do the two decoders prefer different genes? (top-N Jaccard) ---")
    print(lg[lg["N"].isin(keep["N"])].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nwrote 7 csv files to {out_dir}")


if __name__ == "__main__":
    main()
