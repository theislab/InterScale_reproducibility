"""Per-cell-type reconstruction MSE for InterScale on CosMx, alongside GITIII.

Reproduces GITIII's figure definition exactly (from its run_cosmx_pancreas.ipynb):

    per-cell MSE = mean over genes of (y_pred - y_true)^2
    grouped by the receiver cell's OWN type, bar = mean, error bar = std

and then adds the panel that makes the two methods actually comparable, because the raw MSEs
are not:

    GITIII's target  y : mean -0.008, sd 0.200  -- the CENTRED cell-state residual
    InterScale's target: adata.X, non-negative log1p expression, mean ~0.3, larger variance

An MSE is only interpretable against the variance of the thing being predicted. GITIII's 0.036
against a target variance of ~0.040 is ~10% of variance explained; the same 0.036 against a
different target would mean something else entirely. Panel B therefore plots the fraction of
variance unexplained (FVU = MSE / Var(y_true), computed within each cell type), which is
scale-free and can be put side by side.

The model is the lowest-val_mse configuration of sweep sio2fsjm (60 trials), retrained here
because every trial in that sweep wrote to the same checkpoint filename and overwrote its
predecessor -- the sweep produces metrics, not a usable model.

    python scripts/melton_jimenez_25/celltype_reconstruction_mse.py
    python scripts/melton_jimenez_25/celltype_reconstruction_mse.py --skip-train   # reuse ckpt
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch

warnings.filterwarnings("ignore", category=FutureWarning, message=r".*spatial_neighbors.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*n_neighs.*")

REPO = Path("/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility")
INTERSCALE_REGISTRY = "/home/icb/francesca.drummer/1-Projects/A3_InterScale/interscale/config_files/registry.yaml"
GITIII = Path("/home/icb/francesca.drummer/1-Projects/A3_InterScale/benchmarks/GITIII/cosmx_pancreas_run")
SAVE_DIR = Path(
    "/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/results/melton25_revision/best_recon/"
)
OUT = REPO / "figures" / "melton25" / "reconstruction_per_celltype"

# Lowest val_mse of sweep sio2fsjm (run ycjcovv2). Note its val_r2 is -0.004: on this dataset the
# best-MSE model is statistically indistinguishable from predicting the per-gene mean, which is
# itself part of the finding rather than a reason to pick a different config.
BEST = {
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

C_IS = "#eb6834"  # reference palette slot 2 - InterScale
C_GI = "#2a78d6"  # slot 1 - GITIII
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

mpl.rcParams.update(
    {
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2, "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    }
)


def build_cfg(seed: int | None = None):
    """melton25/node_reg with the winning trial's values applied, optionally at a given seed.

    The seed reaches the checkpoint filename through get_model_filename_prefix, so runs at
    different seeds coexist in SAVE_DIR instead of overwriting one another.
    """
    from interscale.config.registry import resolve_config
    from interscale.config.sweep import apply_sweep_config

    cfg = resolve_config("melton25", "node_reg", registry_path=INTERSCALE_REGISTRY)
    cfg, _ = apply_sweep_config(cfg, "hyperparmeter", BEST, model_type="CombinedModel", sweep_params=list(BEST))
    cfg.defrost()
    cfg.model.save = str(SAVE_DIR)
    if seed is not None:
        cfg.optim.seed = int(seed)
    cfg.freeze()
    return cfg


def train_or_load(cfg, skip_train: bool):
    """Train the winning config once, or load the checkpoint written by an earlier run."""
    import interscale
    from interscale.geome_dataloader import GraphAnnDataModule
    from interscale.tl import prepare_geome_dataset, remove_zero_expression_cells, set_full_reproducibility

    set_full_reproducibility(cfg.optim.seed)
    adata = remove_zero_expression_cells(sc.read_h5ad(cfg.dataset.h5ad_data))

    interscale.model.CombinedModel._setup_anndata(
        adata=adata,
        prediction_task=cfg.dataset.prediction_task,
        layer_key=cfg.dataset.layer_key,
        sample_key_list=cfg.dataset.sample_key,
        prediction_obs=cfg.dataset.prediction_obs,
        view_registry=False,
    )

    if skip_train:
        model = interscale.model.CombinedModel.load(
            str(SAVE_DIR), adata, cfg, local_component=True, global_component=True
        )
        return model, adata

    model = interscale.model.CombinedModel(adata, cfg=cfg)
    pyg, _ = prepare_geome_dataset(adata, cfg)
    dm = GraphAnnDataModule(
        datas=pyg, num_workers=1, batch_size=int(cfg.dataset.batch_size),
        mask_percentage=cfg.dataset.mask_percentage, mask_strategy=cfg.dataset.mask_strategy,
        learning_type=cfg.dataset.prediction_level,
    )
    model.train(max_epochs=cfg.optim.n_epochs, datamodule=dm, early_stopping=cfg.optim.early_stopping)
    return model, adata


def interscale_per_cell(cfg, model, adata) -> pd.DataFrame:
    """Per-cell MSE of both decoders, plus the per-cell target variance for normalisation."""
    adata = model.get_model_output(adata, prefix="")
    adata.obsm.pop("_attn_matrix", None)  # 387k x 1600 floats this analysis never reads

    X = adata.layers[cfg.dataset.layer_key] if cfg.dataset.layer_key else adata.X
    y_true = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float32)

    # Denominator for the scale-free panel: squared deviation from the PER-GENE mean profile,
    # i.e. the error a constant predictor makes. NOT the per-cell variance across genes -- that
    # denominator would credit InterScale for the shape of the average expression profile (most
    # genes low, a few high), which GITIII's centred target has already removed. Using the same
    # "distance from the mean predictor" for both is what makes MSE/SST comparable at all.
    mean_profile = y_true.mean(axis=0, keepdims=True)
    sst = ((y_true - mean_profile) ** 2).sum(axis=1)

    out = {"cell_type": np.asarray(adata.obs["cell_type_coarse"]).astype(str), "sst": sst}
    for which in ("local", "global"):
        y_pred = np.asarray(adata.layers[f"_y_pred_{which}"], dtype=np.float32)
        ok = np.isfinite(y_pred).all(axis=1)
        sse = np.full(len(y_true), np.nan, dtype=np.float64)
        sse[ok] = ((y_pred[ok] - y_true[ok]) ** 2).sum(axis=1)
        out[f"sse_{which}"] = sse
        out[f"mse_{which}"] = sse / y_true.shape[1]
    return pd.DataFrame(out), y_true


def gitiii_per_cell() -> pd.DataFrame:
    """GITIII per-cell MSE, recomputed from its influence tensors with its own definition."""
    ys, ps, cts = [], [], []
    for section in ("Run5211_S1", "Run5211_S2", "Run5211_S3"):
        x = torch.load(GITIII / "influence_tensor" / f"edges_{section}.pth", weights_only=False)
        ys.append(x["y"])
        ps.append(x["y_pred"])
        cts += [r[0] for r in x["cell_type_name"]]
        del x
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    del ys, ps

    # Same construction as for InterScale, on GITIII's own target: the mean profile is taken
    # across all cells of all three sections, so both methods are scored against "predict the
    # per-gene mean" on their respective targets.
    mean_profile = y.mean(axis=0, keepdims=True)
    return pd.DataFrame(
        {
            "cell_type": cts,
            "sse": ((p - y) ** 2).sum(axis=1),
            "sst": ((y - mean_profile) ** 2).sum(axis=1),
            "mse": ((p - y) ** 2).mean(axis=1),
        }
    )


def summarise(df, mse_col, sse_col):
    """Per-cell-type MSE (GITIII's definition) plus the scale-free FVU = SSE/SST = 1 - R2."""
    g = df.groupby("cell_type")
    return pd.DataFrame(
        {
            "mse_mean": g[mse_col].mean(),
            "mse_std": g[mse_col].std(),
            "n": g[mse_col].size(),
            "fvu": g[sse_col].sum() / g["sst"].sum(),
        }
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-train", action="store_true", help="Reuse the checkpoint in SAVE_DIR.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg()
    model, adata = train_or_load(cfg, args.skip_train)
    per_cell, _ = interscale_per_cell(cfg, model, adata)
    per_cell.to_csv(OUT / "interscale_per_cell_mse.csv", index=False)

    gi = gitiii_per_cell()
    is_sum = summarise(per_cell.dropna(subset=["mse_global"]), "mse_global", "sse_global")
    gi_sum = summarise(gi, "mse", "sse")

    order = is_sum.sort_values("mse_mean").index.tolist()
    is_sum.reindex(order).to_csv(OUT / "interscale_per_celltype.csv")
    gi_sum.to_csv(OUT / "gitiii_per_celltype.csv")
    print("\nInterScale per cell type:\n", is_sum.reindex(order).to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nGITIII per cell type:\n", gi_sum.to_string(float_format=lambda v: f"{v:.4f}"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.bar(order, is_sum.reindex(order)["mse_mean"], yerr=is_sum.reindex(order)["mse_std"],
           capsize=4, color=C_IS, zorder=3, error_kw=dict(lw=1, ecolor=INK_2))
    ax.set_ylabel("Reconstruction MSE (predicted vs. log1p expression)")
    ax.set_xlabel("Cell type (cell_type_coarse)")
    ax.set_title("A  InterScale reconstruction loss per cell type\nCosMx pancreas — GITIII figure definition",
                 fontsize=9.5, pad=8)
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax = axes[1]
    shared = [c for c in order if c in gi_sum.index]
    x = np.arange(len(shared))
    w = 0.38
    ax.bar(x - w / 2, is_sum.reindex(shared)["fvu"], width=w - 0.03, color=C_IS, zorder=3, label="InterScale")
    ax.bar(x + w / 2, gi_sum.reindex(shared)["fvu"], width=w - 0.03, color=C_GI, zorder=3, label="GITIII")
    ax.axhline(1.0, color=INK_2, lw=1.2, ls=(0, (5, 3)), zorder=4)
    ax.text(len(shared) - 0.5, 1.0, " predicting the mean", ha="right", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(x, shared)
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    ax.set_ylabel("fraction of variance unexplained  (MSE / Var)")
    ax.set_xlabel("Cell type (cell_type_coarse)")
    ax.set_title("B  Scale-free comparison — the two targets differ\n"
                 "(GITIII predicts a centred cell-state residual, InterScale log1p expression)",
                 fontsize=9.5, pad=8)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_png = OUT / "reconstruction_mse_per_celltype.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
