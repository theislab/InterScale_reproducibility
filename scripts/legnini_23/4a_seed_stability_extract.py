"""Legnini - per-seed extraction for the decoder-loading stability analysis.

Reviewer question this answers (part 1 of 2): are the standardized decoder loadings -- and the
local/global gene assignments read off them -- stable across random seeds?

This script runs the Fig. 3E/3F gene-level pipeline **once per seed** and writes the raw
ingredients to disk. `4b_seed_stability_compare.py` then does the cross-seed comparison; nothing
here compares seeds, so a failed/missing seed can be re-extracted on its own.

The five models come from the seed sweep registered as
    config_files/sweeps/seed.yaml  (interscale repo)
    wandb sweep zmnw1x9c, project InterScale_hyperparameter_sweep
which varies `optim.seed` over [40..44] and NOTHING else -- radius 200, pct_mask_nodes 0.3,
n_embed 16, GCN 2x256, transformer 2 layers/4 heads are all fixed at the values the
`legnini23 / node_reg` registry pair resolves to.

Three details are load-bearing, and all three were wrong in the original notebook pipeline:

1. The adata must go through `remove_zero_expression_cells` before inference, exactly as
   training did (43762 -> 43114 cells on this dataset). Otherwise the graphs fed to the model
   here are not the graphs it was trained on.
2. Local loadings are standardized by std(`_local_emb`), not std(`_global_emb`). The dual-decoder
   module decodes the LOCAL embedding through the local decoder
   (`DualDecoderCombinedModule.predict_local`), so using the global embedding rescales every
   local dimension by the wrong constant -- which changes dimension ranking, and therefore which
   dimensions Fig. 3E/3F selects.
3. `calculate_gene_ranks` compares predictions against `adata.X`, but `legnini23_pp.h5ad` keeps
   raw counts in X (max ~101) while the model decodes `layers["log1p_norm"]` (max ~4.6). X is
   therefore repointed at the log layer before ranking. (interscale now asserts on this rather
   than silently returning nonsense R2, so this is belt and braces.)

Dimension selection is automatic (`calculate_dim_importance` -> `dims_left` at
`--cumulative-cutoff`), not the hand-picked dim lists the notebook used for the figure: a
stability claim needs a selection rule that can be re-run per seed.

Usage
-----
    python scripts/legnini_23/4a_seed_stability_extract.py            # all five seeds
    python scripts/legnini_23/4a_seed_stability_extract.py --seeds 42 --overwrite

Run it inside the interscale apptainer container (see run_seed_stability.sh).
"""

# %%
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import scanpy as sc

warnings.filterwarnings("ignore", category=FutureWarning, message=r".*spatial_neighbors.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*n_neighs.*")

# %%
INTERSCALE_REPO = Path("/home/icb/francesca.drummer/1-Projects/A3_InterScale/interscale")
LUSTRE = Path("/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer")
SEEDS = [40, 41, 42, 43, 44]

# The CosMx (melton25) seed models were NOT trained at the registry defaults. They come from
# `celltype_reconstruction_mse.py`, which retrained the lowest-val_mse trial of sweep sio2fsjm
# once per seed into `best_recon/`, so the config has to be reapplied here or the state dict will
# not match the model that gets built (load() uses strict=False and would silently return a
# garbage model). Keep this in sync with that script's BEST dict.
MELTON25_BEST = {
    "optim.loss": "GaussianNLL",
    "model.n_embed": 128,
    "model.local_component.parameters.hidden_dim": 256,
    "model.local_component.parameters.num_layers": 1,
    "model.local_component.parameters.dropout_local": 0.0,
    "model.global_component.parameters.num_layers": 2,
    "model.global_component.parameters.n_heads": 2,
    "model.global_component.parameters.dim_feedforward": 256,
    "model.global_component.parameters.dropout_global": 0.0,
    "dataset.batch_size": 8,
    "dataset.mask_percentage": 0.1,
    "optim.lr": 0.005,
    "optim.wd": 0.01,
    "optim.lr_warmup": 20,
    "optim.monitor": "val_concordance_corr",
}

# Per-seed arrays are a few MB each; they live next to the checkpoints, not in the git repo.
SPECS = {
    "legnini23": {
        "dataset": "legnini23",
        "task": "node_reg",
        "models_dir": LUSTRE / "results" / "legnini23_revision" / "seed_stability",
        "overrides": {},
        # legnini23 pins model.save with a trailing separator, so the checkpoints sit INSIDE
        # models_dir and the filename carries no extra prefix.
        "save_is_dir": True,
    },
    "melton25": {
        "dataset": "melton25",
        "task": "node_reg",
        "models_dir": LUSTRE / "results" / "melton25_revision" / "best_recon",
        "overrides": MELTON25_BEST,
        "save_is_dir": True,
    },
    # Masking-granularity pair, matched on everything but the masking. Same five seeds, same 300
    # epochs, same architecture, both trained with MASK_VALUE = -1 -- so a difference in how stable
    # the gene loadings are across seeds is attributable to cell-vs-gene masking. These mirror the
    # MASKING switch in notebooks/legnini_23/4_decoder_stability.ipynb; keep the two in step.
    #
    # Distinct from "legnini23" above, which is the PUBLISHED arm: 100 epochs, MASK_VALUE = 0,
    # checkpoints in seed_stability/. Left untouched.
    #
    # 300 epochs because at the 100 default no trial on this dataset ever early-stops
    # (optim.min_delta is 0.0, so only the n_epochs cap stops a run) and 100 -> 300 moves
    # val_concordance_corr by +200% to +286%, with seed spread falling from 2-18% to 1-4%. A
    # stability analysis at 100 would mostly measure how far short of its schedule each run was.
    #
    # config_files/sweeps/seed_cell_masking.yaml -- wandb sweep ze40fzaz
    "legnini23_mask_cell": {
        "dataset": "legnini23",
        "task": "node_reg",
        "models_dir": LUSTRE / "results" / "legnini23_revision" / "seed_stability_cell",
        "overrides": {"optim.n_epochs": 300},
        "save_is_dir": True,
    },
    # config_files/sweeps/seed_gene_masking.yaml -- wandb sweep dty3dlk3
    #
    # mask_percentage 0.25, not the node_reg_genemask task file's 0.5: the sweep pinned it, and
    # 0.25 is where gene masking measurably wins (val_concordance_corr 0.1947 +/- 0.0041 against
    # cell 0.30's 0.1634 +/- 0.0084). It changes no tensor shape, so a mismatch here would load
    # cleanly and only be wrong in the recorded config.
    "legnini23_mask_gene": {
        "dataset": "legnini23",
        "task": "node_reg_genemask",
        "models_dir": LUSTRE / "results" / "legnini23_revision" / "seed_stability_gene",
        "overrides": {"optim.n_epochs": 300, "dataset.mask_percentage": 0.25},
        "save_is_dir": True,
    },
}
DEFAULT_SPEC = "legnini23"


# %%
def build_cfg(seed: int, models_dir: Path, registry: Path, spec: dict):
    """Resolve the registry config for this run and pin it to one seed / checkpoint directory.

    The seed sweep applied `optim.seed` and `model.save` on top of the resolved pair, so the same
    two keys are applied here -- `get_model_filename_prefix` puts the seed in the filename, which
    is what makes the five checkpoints separable.

    `spec["overrides"]` carries anything else the run was trained with that the registry does not
    resolve to. It is empty for legnini23 (whose seed sweep varied nothing but the seed), the
    winning hyperparameter trial for melton25, and the epoch/rate pins for the two
    legnini23_mask_* arms.
    """
    from interscale.config.registry import resolve_config
    from interscale.config.sweep import apply_sweep_config

    cfg = resolve_config(spec["dataset"], spec["task"], registry_path=str(registry))
    if spec["overrides"]:
        cfg, _ = apply_sweep_config(
            cfg, "hyperparmeter", spec["overrides"], model_type="CombinedModel",
            sweep_params=list(spec["overrides"]),
        )
    cfg.defrost()
    cfg.optim.seed = int(seed)
    cfg.model.save = str(models_dir) + os.sep
    cfg.freeze()
    return cfg


def target_layer(adata, cfg) -> str:
    """Name of the layer holding what the decoders reconstruct, materialising it if needed.

    `cfg.dataset.layer_key` is None for melton25 -- that dataset's expression lives in .X and its
    only layer is raw counts. Several downstream calls (`gene_loadings`, `get_genes_dim`) take a
    layer NAME and raise on None, so .X is exposed under an explicit key rather than branching at
    every call site.
    """
    if cfg.dataset.layer_key:
        return cfg.dataset.layer_key
    if "_target" not in adata.layers:
        adata.layers["_target"] = adata.X.copy()
    return "_target"


def load_adata(cfg):
    """Load the h5ad with the same preprocessing the training run applied."""
    from interscale.tl import remove_zero_expression_cells

    adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    adata = remove_zero_expression_cells(adata)
    return adata


def per_gene_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    """Per-gene reconstruction quality, over the cells where truth and prediction are finite.

    Kept separate from `calculate_gene_ranks` (which only returns ranks) because the reviewer
    response needs the absolute reconstruction quality, and because it is the sanity check that
    has to hold before any loading-stability number is worth interpreting.

    Returns raw R2, *offset-corrected* R2, Pearson r, and the first two moments of both truth and
    prediction per gene. The extras are not padding: raw R2 on this model is dominated by a
    systematic mean offset and by variance shrinkage, so the raw number alone (and especially its
    mean over genes, which low-variance genes drive to -1e2) describes the miscalibration rather
    than the correlation structure the loadings are read from.
    """
    n_genes = y_true.shape[1]
    out = {k: np.full(n_genes, np.nan) for k in ("r2", "r2_offset_corrected", "pearson", "true_mean", "true_std", "pred_mean", "pred_std")}

    for g in range(n_genes):
        t = y_true[:, g]
        p = y_pred[:, g]
        ok = np.isfinite(t) & np.isfinite(p)
        if ok.sum() < 3:
            continue
        t, p = t[ok], p[ok]
        out["true_mean"][g], out["true_std"][g] = t.mean(), t.std()
        out["pred_mean"][g], out["pred_std"][g] = p.mean(), p.std()

        ss_tot = float(np.sum((t - t.mean()) ** 2))
        if ss_tot > 0:
            out["r2"][g] = 1.0 - float(np.sum((t - p) ** 2)) / ss_tot
            shifted = p - p.mean() + t.mean()
            out["r2_offset_corrected"][g] = 1.0 - float(np.sum((t - shifted) ** 2)) / ss_tot
        if t.std() > 0 and p.std() > 0:
            out["pearson"][g] = float(np.corrcoef(t, p)[0, 1])

    return out


def importance_in_dim_order(result, n_dims: int) -> np.ndarray:
    """Re-index `calculate_dim_importance`'s sorted scores back onto dimension order."""
    y = np.full(n_dims, np.nan)
    y[np.asarray(result["dim_sorted"], dtype=int)] = np.asarray(result["y_sorted"], dtype=float)
    return y


# %%
def extract_seed(seed: int, args) -> dict:
    """Run the full gene-level pipeline for one seed and write its arrays to disk."""
    import interscale
    from interscale.evaluation import calculate_dim_importance, calculate_gene_ranks, gene_loadings, get_genes_dim
    from interscale.tl.utils import get_model_filename_prefix

    print(f"\n{'=' * 70}\nseed {seed}\n{'=' * 70}", flush=True)

    spec = SPECS[args.spec]
    cfg = build_cfg(seed, Path(args.models_dir), Path(args.registry), spec)
    adata = load_adata(cfg)
    layer_key = target_layer(adata, cfg)

    interscale.model.CombinedModel._setup_anndata(
        adata=adata,
        prediction_task=cfg.dataset.prediction_task,
        layer_key=cfg.dataset.layer_key,
        sample_key_list=cfg.dataset.sample_key,
        prediction_obs=cfg.dataset.prediction_obs,
        view_registry=False,
    )
    model = interscale.model.CombinedModel.load(
        str(args.models_dir),
        adata,
        cfg,
        local_component=True,
        global_component=True,
    )

    # Embeddings and both decoders' predictions. prefix="" gives the keys the evaluation
    # functions default to: obsm _local_emb/_global_emb, layers _y_pred_local/_y_pred_global.
    adata = model.get_model_output(adata, prefix="")

    # 43114 x 4318 float32 of attention (~750 MB) that this analysis does not use.
    adata.obsm.pop("_attn_matrix", None)

    # --- standardized loadings (note: local standardized by the LOCAL embedding) -------------
    gene_loadings(
        adata,
        model,
        layer_key=layer_key,
        local_latent_key="_local_emb",
        global_latent_key="_global_emb",
    )
    S_local = np.asarray(adata.varm["_local_std_gene_loadings"], dtype=np.float64)
    S_global = np.asarray(adata.varm["_global_std_gene_loadings"], dtype=np.float64)

    core = model.module
    W_local = core.state_dict()["local_module.decoder.decoder.weight"].detach().cpu().numpy()
    W_global = core.state_dict()["global_module.decoder.decoder.weight"].detach().cpu().numpy()

    Z_local = np.asarray(adata.obsm["_local_emb"], dtype=np.float32)
    Z_global = np.asarray(adata.obsm["_global_emb"], dtype=np.float32)
    n_dims = Z_global.shape[1]

    # --- informative dimensions, per component ----------------------------------------------
    imp = {}
    dims = {}
    for which, s_key, z_key in (
        ("local", "_local_std_gene_loadings", "_local_emb"),
        ("global", "_global_std_gene_loadings", "_global_emb"),
    ):
        res = calculate_dim_importance(
            adata,
            s_key=s_key,
            z_key=z_key,
            mode="full",
            use_ratio=True,
            cumulative_cutoff=args.cumulative_cutoff,
            spacing=2,
        )
        imp[which] = importance_in_dim_order(res, n_dims)
        dims[which] = np.asarray(res["dims_left"], dtype=int)
        print(f"  {which}: {len(dims[which])} dims at cumulative cutoff {args.cumulative_cutoff} -> {dims[which]}")

    # --- top genes per dimension, via the same call Fig. 3F uses -----------------------------
    programs = {}
    top_genes_per_dim = {}
    for which in ("local", "global"):
        df = get_genes_dim(
            adata,
            dims=dims[which].tolist(),
            which=which,
            n_top=args.n_top,
            X_layer=layer_key,
            min_frac=0.1,
            rank_by="loading_x_sd",
            enforce_specificity=True,
            specificity_min=1.5,
            plot=False,
        )
        programs[which] = [str(g) for g in df.index]
        # Per-dim winners: the genes this dim scores highest in absolute value, within the
        # specificity-filtered panel.
        top_genes_per_dim[which] = {
            str(d): [str(g) for g in df[d].abs().sort_values(ascending=False).head(args.n_top).index]
            for d in df.columns
        }
        print(f"  {which}: program panel = {len(programs[which])} genes")

    # --- reconstruction quality + local/global gene ranking ---------------------------------
    _t = adata.layers[layer_key]
    y_true = np.asarray(_t.todense() if hasattr(_t, "todense") else _t, dtype=np.float64)
    y_local = np.asarray(adata.layers["_y_pred_local"], dtype=np.float64)
    y_global = np.asarray(adata.layers["_y_pred_global"], dtype=np.float64)

    q_local = per_gene_metrics(y_true, y_local)
    q_global = per_gene_metrics(y_true, y_global)

    # `calculate_gene_ranks` reads ground truth from .X, which holds raw counts in this h5ad.
    adata.X = adata.layers[layer_key].copy()
    ranks = calculate_gene_ranks(
        adata,
        layers_local_pred="_y_pred_local",
        layers_global_pred="_y_pred_global",
    ).set_index("gene")
    ranks = ranks.loc[adata.var_names]

    var_names = np.asarray(adata.var_names, dtype=object)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out / f"seed{seed}_arrays.npz",
        var_names=var_names,
        W_local=W_local,
        W_global=W_global,
        S_local=S_local,
        S_global=S_global,
        Z_local=Z_local,
        Z_global=Z_global,
        x_std=np.asarray(np.std(y_true, axis=0, ddof=1)),
        importance_local=imp["local"],
        importance_global=imp["global"],
        dims_local=dims["local"],
        dims_global=dims["global"],
        **{f"{k}_local": v for k, v in q_local.items()},
        **{f"{k}_global": v for k, v in q_global.items()},
        rank_difference=ranks["Rank Difference"].to_numpy(dtype=float),
        local_rank=ranks["Local Rank"].to_numpy(dtype=float),
        global_rank=ranks["Global Rank"].to_numpy(dtype=float),
    )

    meta = {
        "seed": int(seed),
        "dataset": spec["dataset"],
        "task": spec["task"],
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_embed": int(n_dims),
        "cumulative_cutoff": float(args.cumulative_cutoff),
        "n_top": int(args.n_top),
        "n_dims_local": int(len(dims["local"])),
        "n_dims_global": int(len(dims["global"])),
        "dims_local": dims["local"].tolist(),
        "dims_global": dims["global"].tolist(),
        "program_local": programs["local"],
        "program_global": programs["global"],
        "top_genes_per_dim": top_genes_per_dim,
        # Medians, not means: a handful of low-variance genes drive mean R2 to -1e2 and make it
        # useless as a stability sanity check.
        **{
            f"{stat}_{which}": float(fn(q[key]))
            for which, q in (("local", q_local), ("global", q_global))
            for stat, key, fn in (
                ("median_gene_r2", "r2", np.nanmedian),
                ("median_gene_r2_offset_corrected", "r2_offset_corrected", np.nanmedian),
                ("median_gene_pearson", "pearson", np.nanmedian),
                ("q90_gene_pearson", "pearson", lambda v: np.nanpercentile(v, 90)),
            )
        },
        **{
            f"n_genes_pearson_gt_0.2_{which}": int(np.nansum(q["pearson"] > 0.2))
            for which, q in (("local", q_local), ("global", q_global))
        },
        # Variance shrinkage: the ratio of predicted to true per-gene sd. Well under 1 means the
        # decoder is regressing toward the mean, which caps every correlation downstream.
        **{
            f"median_pred_true_sd_ratio_{which}": float(np.nanmedian(q["pred_std"] / q["true_std"]))
            for which, q in (("local", q_local), ("global", q_global))
        },
        "checkpoint": str(
            Path(args.models_dir)
            / f"{get_model_filename_prefix(cfg, True, True)}model.pt"
        ),
    }
    with open(out / f"seed{seed}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"  median per-gene R2 local={meta['median_gene_r2_local']:.3f}  global={meta['median_gene_r2_global']:.3f}\n"
        f"  median per-gene r  local={meta['median_gene_pearson_local']:.4f}  "
        f"global={meta['median_gene_pearson_global']:.4f}  "
        f"(genes r>0.2: {meta['n_genes_pearson_gt_0.2_local']} / {meta['n_genes_pearson_gt_0.2_global']})\n"
        f"  pred/true sd ratio local={meta['median_pred_true_sd_ratio_local']:.3f}  "
        f"global={meta['median_pred_true_sd_ratio_global']:.3f}\n"
        f"  wrote {out / f'seed{seed}_arrays.npz'}"
    )
    return meta


# %%
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", default=DEFAULT_SPEC, choices=sorted(SPECS),
                        help="Which dataset's seed run to extract.")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--models-dir", default=None, help="Directory holding the per-seed checkpoints.")
    parser.add_argument("--out-dir", default=None, help="Where the per-seed arrays are written.")
    parser.add_argument(
        "--registry",
        default=str(INTERSCALE_REPO / "config_files" / "registry.yaml"),
        help="Path to the interscale registry (absolute, so the script runs from anywhere).",
    )
    parser.add_argument(
        "--cumulative-cutoff",
        type=float,
        default=0.60,
        help="Cumulative importance kept when selecting informative dimensions (0.60 matches the figure).",
    )
    parser.add_argument("--n-top", type=int, default=20, help="Top genes per dimension.")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract seeds that already have arrays.")
    args = parser.parse_args()
    spec = SPECS[args.spec]
    if args.models_dir is None:
        args.models_dir = str(spec["models_dir"])
    if args.out_dir is None:
        args.out_dir = str(Path(args.models_dir) / "analysis")
    print(f"spec={args.spec} models={args.models_dir} out={args.out_dir}")

    done, skipped, failed = [], [], []
    for seed in args.seeds:
        target = Path(args.out_dir) / f"seed{seed}_arrays.npz"
        if target.exists() and not args.overwrite:
            print(f"seed {seed}: {target.name} exists, skipping (use --overwrite)")
            skipped.append(seed)
            continue
        try:
            extract_seed(seed, args)
            done.append(seed)
        except FileNotFoundError as e:
            # A seed whose sweep trial has not finished yet: report it and keep going, so a
            # partial sweep still yields a partial analysis.
            print(f"seed {seed}: FAILED -- {e}")
            failed.append(seed)

    print(f"\nextracted={done} skipped={skipped} failed={failed}")
    if failed:
        raise SystemExit(f"{len(failed)} seed(s) missing a checkpoint: {failed}")


if __name__ == "__main__":
    main()
