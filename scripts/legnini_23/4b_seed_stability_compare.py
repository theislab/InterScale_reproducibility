"""Legnini - cross-seed stability of the standardized decoder loadings.

Reviewer question this answers (part 2 of 2): "there is no assessment of whether these
assignments are stable across different random seeds, model initializations, or hyperparameter
choices."

Reads the per-seed arrays written by `4a_seed_stability_extract.py` and reports, for the local
and global decoder separately:

  * **Sanity first.** Reconstruction quality (per-gene R2 and Pearson r) and the *number* of
    selected dimensions, per seed. If these are not themselves stable, no loading-stability
    number is interpretable, so they are printed and written before anything else.
  * **Dimension alignment.** Embedding dimensions are only defined up to permutation and sign
    across independently initialized runs, so every seed pair is aligned by Hungarian matching
    on |corr(z_i^a, z_j^b)| before loadings are compared. Matches below `--min-r` are discarded
    and the discard rate is reported -- a high discard rate is itself the finding (dimensions
    that do not recur across seeds).
  * **Loading stability.** Spearman correlation of the full 88-gene loading vector for each
    matched dimension pair (sign-corrected), plus Jaccard overlap of the top-10 and top-20 genes
    by |standardized loading|.
  * **Program stability.** Jaccard of the local and global gene *panels* Fig. 3F actually plots
    (`get_genes_dim` over the selected dims), and the agreement of the local-only / global-only /
    shared labelling across seeds. This is the figure's claim stated directly.
  * **A matching-free check.** Spearman of the per-gene local-minus-global reconstruction rank
    difference, and how often its sign agrees across seeds. This needs no dimension matching at
    all, so it cannot be inflated or deflated by the alignment step.

Every Jaccard is reported against a **permutation null**: with only 88 genes in this panel, two
random top-20 sets already overlap at J ~ 0.13, so raw Jaccard read against 0 would overstate
stability.

Usage
-----
    python scripts/legnini_23/4b_seed_stability_compare.py
    python scripts/legnini_23/4b_seed_stability_compare.py --sweep-id zmnw1x9c   # + wandb metrics
"""

# %%
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

# %%
ARRAYS_DIR = Path(
    "/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/results/legnini23_revision/seed_stability/analysis"
)
REPO = Path("/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility")
FIG_DIR = REPO / "figures" / "legnini23" / "seed_stability"

SEEDS = [40, 41, 42, 43, 44]
COMPONENTS = ("local", "global")


# %%
def load_seed(seed: int, arrays_dir: Path):
    npz = np.load(arrays_dir / f"seed{seed}_arrays.npz", allow_pickle=True)
    with open(arrays_dir / f"seed{seed}_meta.json") as f:
        meta = json.load(f)
    return npz, meta


def top_k_genes(loadings: np.ndarray, k: int) -> set[int]:
    """Indices of the k genes with the largest |standardized loading| on one dimension."""
    return set(np.argsort(np.abs(loadings))[::-1][:k].tolist())


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else np.nan


def jaccard_null(n_total: int, k_a: int, k_b: int, n_draws: int, rng) -> float:
    """Expected Jaccard of two independent random gene sets of the given sizes."""
    if min(k_a, k_b) == 0:
        return np.nan
    vals = np.empty(n_draws)
    for i in range(n_draws):
        a = set(rng.choice(n_total, size=k_a, replace=False).tolist())
        b = set(rng.choice(n_total, size=k_b, replace=False).tolist())
        vals[i] = jaccard(a, b)
    return float(vals.mean())


def column_corr(Za: np.ndarray, Zb: np.ndarray) -> np.ndarray:
    """Correlation between every column of Za and every column of Zb, over shared finite rows."""
    ok = np.isfinite(Za).all(axis=1) & np.isfinite(Zb).all(axis=1)
    A = Za[ok].astype(np.float64)
    B = Zb[ok].astype(np.float64)
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    sa = A.std(axis=0, ddof=1)
    sb = B.std(axis=0, ddof=1)
    # A constant dimension has no correlation with anything; guard the divide and let the
    # resulting 0 fall below --min-r so the match is discarded rather than silently kept.
    sa[sa == 0] = np.inf
    sb[sb == 0] = np.inf
    return (A.T @ B) / (A.shape[0] - 1) / np.outer(sa, sb)


# %%
def match_dims(Za: np.ndarray, Zb: np.ndarray, min_r: float):
    """Align dimensions of two runs by Hungarian matching on |correlation|.

    Returns one record per matched pair, with the sign needed to bring b's loading vector into
    a's orientation (dimensions are identifiable only up to sign across runs).
    """
    C = column_corr(Za, Zb)
    rows, cols = linear_sum_assignment(-np.abs(C))
    return [
        {
            "dim_a": int(i),
            "dim_b": int(j),
            "r": float(C[i, j]),
            "abs_r": float(abs(C[i, j])),
            "sign": float(np.sign(C[i, j])) or 1.0,
            "kept": bool(abs(C[i, j]) >= min_r),
        }
        for i, j in zip(rows, cols, strict=True)
    ]


def shuffled_pairing_rows(Sa, Sb, matches, n_shuffle, rng, seed_a, seed_b, which):
    """Same statistics, but pairing dimensions at random instead of by Hungarian matching.

    This is the control that matters for the top-gene Jaccard. A standardized loading is
    `W * std(z) / std(x_gene)`, and the `1/std(x_gene)` factor is a property of the *data*, so it
    is byte-identical across seeds. Genes with small expression sd therefore land in the top-|S|
    set of essentially every dimension of every run, which inflates Jaccard far above a
    gene-permutation null even when the decoder weights are unrelated. Only Jaccard above THIS
    null is evidence that a dimension recurs across seeds.
    """
    n_dims = Sa.shape[1]
    matched_of = {m["dim_a"]: m["dim_b"] for m in matches}
    rows = []
    for _ in range(n_shuffle):
        for dim_a in range(n_dims):
            choices = [d for d in range(n_dims) if d != matched_of.get(dim_a)]
            dim_b = int(rng.choice(choices))
            la = Sa[:, dim_a]
            lb = Sb[:, dim_b]
            # Sign is not identifiable, and the matched case gets it from the embedding
            # correlation; here take the orientation that favours the null.
            sp = float(spearmanr(la, lb).statistic)
            rows.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "component": which,
                    "spearman_loadings": abs(sp),
                    "jaccard_top10": jaccard(top_k_genes(la, 10), top_k_genes(lb, 10)),
                    "jaccard_top20": jaccard(top_k_genes(la, 20), top_k_genes(lb, 20)),
                }
            )
    return rows


def compare_pair(seed_a, seed_b, data, args, rng):
    """All cross-seed statistics for one seed pair, for both components."""
    npz_a, meta_a = data[seed_a]
    npz_b, meta_b = data[seed_b]

    n_genes = len(npz_a["var_names"])
    assert list(npz_a["var_names"]) == list(npz_b["var_names"]), "gene order differs between seeds"
    assert npz_a["Z_global"].shape[0] == npz_b["Z_global"].shape[0], "cell count differs between seeds"

    dim_rows, pair_rows, null_rows = [], [], []

    for which in COMPONENTS:
        Za, Zb = npz_a[f"Z_{which}"], npz_b[f"Z_{which}"]
        Sa, Sb = npz_a[f"S_{which}"], npz_b[f"S_{which}"]
        selected_a = set(npz_a[f"dims_{which}"].tolist())
        selected_b = set(npz_b[f"dims_{which}"].tolist())

        matches = match_dims(Za, Zb, args.min_r)
        for m in matches:
            row = {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "component": which,
                **m,
                "selected_a": m["dim_a"] in selected_a,
                "selected_b": m["dim_b"] in selected_b,
            }
            if m["kept"]:
                la = Sa[:, m["dim_a"]]
                lb = Sb[:, m["dim_b"]] * m["sign"]
                row["spearman_loadings"] = float(spearmanr(la, lb).statistic)
                for k in (10, 20):
                    row[f"jaccard_top{k}"] = jaccard(top_k_genes(la, k), top_k_genes(lb, k))
            dim_rows.append(row)

        null_rows += shuffled_pairing_rows(
            Sa, Sb, matches, args.n_shuffle, rng, seed_a, seed_b, which
        )

        kept = [m for m in matches if m["kept"]]
        pair_rows.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "component": which,
                "n_dims": len(matches),
                "n_kept": len(kept),
                "discard_rate": 1.0 - len(kept) / len(matches) if matches else np.nan,
                "mean_abs_r": float(np.mean([m["abs_r"] for m in matches])),
                # Restricted to dimensions the selection rule actually kept in run a: those are
                # the ones Fig. 3E/3F reads genes off.
                "n_selected_a_kept": sum(1 for m in kept if m["dim_a"] in selected_a),
                "n_selected_a": len(selected_a),
            }
        )

        # --- program-level: the gene panels the figure plots -------------------------------
        pa, pb = set(meta_a[f"program_{which}"]), set(meta_b[f"program_{which}"])
        pair_rows[-1]["jaccard_program"] = jaccard(pa, pb)
        pair_rows[-1]["jaccard_program_null"] = jaccard_null(n_genes, len(pa), len(pb), args.n_permutations, rng)

    # --- local-vs-global labelling of genes, and the matching-free rank check --------------
    labels = {}
    for seed, (_, meta) in ((seed_a, data[seed_a]), (seed_b, data[seed_b])):
        loc, glo = set(meta["program_local"]), set(meta["program_global"])
        labels[seed] = {
            **{g: "local_only" for g in loc - glo},
            **{g: "global_only" for g in glo - loc},
            **{g: "shared" for g in loc & glo},
        }
    shared_genes = set(labels[seed_a]) & set(labels[seed_b])
    label_agreement = (
        np.mean([labels[seed_a][g] == labels[seed_b][g] for g in shared_genes]) if shared_genes else np.nan
    )

    rd_a = npz_a["rank_difference"]
    rd_b = npz_b["rank_difference"]
    ok = np.isfinite(rd_a) & np.isfinite(rd_b)

    pair_rows.append(
        {
            "seed_a": seed_a,
            "seed_b": seed_b,
            "component": "gene_level",
            "label_agreement": float(label_agreement),
            "n_genes_labelled_both": len(shared_genes),
            "spearman_rank_difference": float(spearmanr(rd_a[ok], rd_b[ok]).statistic),
            "sign_agreement_rank_difference": float(np.mean(np.sign(rd_a[ok]) == np.sign(rd_b[ok]))),
        }
    )

    return dim_rows, pair_rows, null_rows


# %%
def sanity_table(data, sweep_id=None):
    """Per-seed reconstruction quality and dimension counts, plus wandb metrics if available."""
    rows = []
    for seed, (npz, meta) in data.items():
        rows.append(
            {
                "seed": seed,
                "n_obs": meta["n_obs"],
                "n_vars": meta["n_vars"],
                "n_dims_local": meta["n_dims_local"],
                "n_dims_global": meta["n_dims_global"],
                "program_size_local": len(meta["program_local"]),
                "program_size_global": len(meta["program_global"]),
                "median_gene_r2_local": meta["median_gene_r2_local"],
                "median_gene_r2_global": meta["median_gene_r2_global"],
                "median_gene_r_local": meta["median_gene_pearson_local"],
                "median_gene_r_global": meta["median_gene_pearson_global"],
                "q90_gene_r_global": meta["q90_gene_pearson_global"],
                "n_genes_r_gt_0.2_local": meta["n_genes_pearson_gt_0.2_local"],
                "n_genes_r_gt_0.2_global": meta["n_genes_pearson_gt_0.2_global"],
                "pred_true_sd_ratio_global": meta["median_pred_true_sd_ratio_global"],
            }
        )
    df = pd.DataFrame(rows).sort_values("seed")

    if sweep_id:
        try:
            import wandb

            api = wandb.Api(timeout=60)
            metrics = {}
            for run in api.sweep(f"francesca-drummer/InterScale_hyperparameter_sweep/{sweep_id}").runs:
                seed = run.config.get("optim", {}).get("seed")
                # The trial's applied value, not the base config's: every run in this sweep
                # carries the same *name*, because main_sweep.py builds the run name before the
                # sweep values are applied. optim.seed is the only reliable identifier.
                seed = run.config.get("optim.seed", seed)
                if seed is not None:
                    metrics[int(seed)] = {
                        "wandb_state": run.state,
                        "wandb_val_r2": run.summary.get("val_r2"),
                        "wandb_test_r2": run.summary.get("test_r2"),
                        "wandb_val_pearson": run.summary.get("val_pearson_corr"),
                        "wandb_epochs": run.summary.get("epoch"),
                    }
            wandb_df = pd.DataFrame(metrics).T.rename_axis("seed").reset_index()
            df = df.merge(wandb_df, on="seed", how="left")
        except Exception as e:  # noqa: BLE001 - wandb is optional here, never fatal
            print(f"could not fetch wandb metrics for sweep {sweep_id}: {type(e).__name__}: {e}")

    return df


def summarize(dim_df, pair_df, null_df, data, args, rng):
    """Collapse the per-pair records into the reportable summary table."""
    n_genes = len(next(iter(data.values()))[0]["var_names"])
    rows = []

    def add(component, metric, values, null=np.nan, level="per_matched_dim"):
        v = pd.Series(values, dtype=float).dropna()
        rows.append(
            {
                "component": component,
                "metric": metric,
                "level": level,
                "n": len(v),
                "mean": v.mean(),
                "sd": v.std(ddof=1) if len(v) > 1 else np.nan,
                "median": v.median(),
                "min": v.min(),
                "max": v.max(),
                "permutation_null": null,
            }
        )

    for which in COMPONENTS:
        kept = dim_df[(dim_df["component"] == which) & dim_df["kept"]]
        kept_sel = kept[kept["selected_a"]]
        pairs = pair_df[pair_df["component"] == which]

        null10 = jaccard_null(n_genes, 10, 10, args.n_permutations, rng)
        null20 = jaccard_null(n_genes, 20, 20, args.n_permutations, rng)

        add(which, "spearman_loadings", kept["spearman_loadings"])
        # The shuffled-pairing control reports |rho| (sign is not identifiable when dimensions are
        # paired at random), so the matched case needs |rho| too or the two are not comparable:
        # a signed mean near zero can hide substantial per-dimension agreement of either sign.
        add(which, "abs_spearman_loadings", kept["spearman_loadings"].abs())
        add(which, "jaccard_top10", kept["jaccard_top10"], null10)
        add(which, "jaccard_top20", kept["jaccard_top20"], null20)
        add(which, "spearman_loadings", kept_sel["spearman_loadings"], level="per_selected_dim")
        add(which, "abs_spearman_loadings", kept_sel["spearman_loadings"].abs(), level="per_selected_dim")
        add(which, "jaccard_top10", kept_sel["jaccard_top10"], null10, level="per_selected_dim")
        add(which, "jaccard_top20", kept_sel["jaccard_top20"], null20, level="per_selected_dim")
        # The control: what these same statistics look like when dimensions are paired at random.
        shuffled = null_df[null_df["component"] == which]
        for metric in ("abs_spearman_loadings", "jaccard_top10", "jaccard_top20"):
            source = "spearman_loadings" if metric == "abs_spearman_loadings" else metric
            add(which, metric, shuffled[source], level="shuffled_dim_pairing")

        add(which, "abs_r_matched_dims", dim_df[dim_df["component"] == which]["abs_r"], level="per_matched_dim")
        add(which, "discard_rate", pairs["discard_rate"], level="per_seed_pair")
        add(which, "jaccard_program", pairs["jaccard_program"], pairs["jaccard_program_null"].mean(), "per_seed_pair")

    gene = pair_df[pair_df["component"] == "gene_level"]
    add("gene_level", "label_agreement", gene["label_agreement"], level="per_seed_pair")
    add("gene_level", "spearman_rank_difference", gene["spearman_rank_difference"], level="per_seed_pair")
    add("gene_level", "sign_agreement_rank_difference", gene["sign_agreement_rank_difference"], level="per_seed_pair")

    return pd.DataFrame(rows)


# %%
def plot_stability(dim_df, null_df, summary, out_path: Path):
    """Matched-dimension stability against the shuffled-pairing control, per component.

    Each panel shows the observed distribution (Hungarian-matched dimensions) next to the same
    statistic under random dimension pairing. The comparison that matters is between the two
    boxes, not between the observed box and zero.
    """
    metrics = ["abs_spearman_loadings", "jaccard_top10", "jaccard_top20"]
    # `shuffled_pairing_rows` already stores |rho| (no identifiable sign under random pairing),
    # so the observed side is taken in absolute value too.
    observed = dim_df[dim_df["kept"]].assign(
        pairing="matched", abs_spearman_loadings=lambda d: d["spearman_loadings"].abs()
    )
    control = null_df.assign(pairing="shuffled dims", abs_spearman_loadings=lambda d: d["spearman_loadings"])
    long = (
        pd.concat([observed, control], ignore_index=True)
        .melt(id_vars=["component", "pairing"], value_vars=metrics, var_name="metric", value_name="value")
        .dropna(subset=["value"])
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    palette = {"matched": "#C1272D", "shuffled dims": "0.75"}

    for ax, metric in zip(axes, metrics, strict=True):
        sub = long[long["metric"] == metric]
        sns.boxplot(
            data=sub, x="component", y="value", hue="pairing", palette=palette, ax=ax,
            showfliers=False, legend=(metric == metrics[0]),
        )
        gene_null = summary[(summary["metric"] == metric) & (summary["level"] == "per_matched_dim")][
            "permutation_null"
        ].dropna()
        if not gene_null.empty:
            ax.axhline(float(gene_null.iloc[0]), ls=":", lw=1.2, color="0.35")
            ax.text(0.99, float(gene_null.iloc[0]), " random gene sets", ha="right", va="bottom",
                    fontsize=8, color="0.35", transform=ax.get_yaxis_transform())
        ax.set_title(metric.replace("_", " "), fontsize=10)
        ax.set_xlabel("")
        ax.axhline(0, lw=0.5, color="0.7")

    axes[0].set_ylabel("cross-seed agreement (10 seed pairs)")
    fig.suptitle(
        "Stability of standardized decoder loadings across seeds 40-44 "
        "(matched vs. randomly paired dimensions)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"wrote {out_path}")
    return fig


# %%
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--arrays-dir", default=str(ARRAYS_DIR))
    parser.add_argument("--out-dir", default=str(FIG_DIR))
    parser.add_argument("--min-r", type=float, default=0.3, help="Discard dimension matches below this |correlation|.")
    parser.add_argument("--n-permutations", type=int, default=2000, help="Draws for the Jaccard permutation null.")
    parser.add_argument(
        "--n-shuffle",
        type=int,
        default=20,
        help="Random dimension pairings per seed pair, for the shuffled-pairing control.",
    )
    parser.add_argument("--sweep-id", default=None, help="wandb sweep id, to add per-seed training metrics.")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {seed: load_seed(seed, Path(args.arrays_dir)) for seed in args.seeds}

    # --- sanity checks first: these gate the interpretation of everything below ------------
    sanity = sanity_table(data, args.sweep_id)
    sanity.to_csv(out_dir / "sanity_checks.csv", index=False)
    print("\n=== per-seed sanity checks ===")
    print(sanity.to_string(index=False))

    dim_rows, pair_rows, null_rows = [], [], []
    for seed_a, seed_b in itertools.combinations(args.seeds, 2):
        d, p, n = compare_pair(seed_a, seed_b, data, args, rng)
        dim_rows += d
        pair_rows += p
        null_rows += n

    dim_df = pd.DataFrame(dim_rows)
    pair_df = pd.DataFrame(pair_rows)
    null_df = pd.DataFrame(null_rows)
    summary = summarize(dim_df, pair_df, null_df, data, args, rng)

    dim_df.to_csv(out_dir / "stability_matched_dims.csv", index=False)
    pair_df.to_csv(out_dir / "stability_pairs.csv", index=False)
    null_df.to_csv(out_dir / "stability_shuffled_pairing.csv", index=False)
    summary.to_csv(out_dir / "stability_summary.csv", index=False)

    print(f"\n=== cross-seed stability ({len(list(itertools.combinations(args.seeds, 2)))} seed pairs) ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    plot_stability(dim_df, null_df, summary, out_dir / "seed_stability_boxplot.png")
    print(f"\nwrote {out_dir / 'stability_summary.csv'} and {out_dir / 'stability_pairs.csv'}")


if __name__ == "__main__":
    main()
